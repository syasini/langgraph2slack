"""Durable, size-bounded bridge state (ADR-0006).

The bridge remembers a few things between events: which run produced which Slack message
(for feedback), tool-call details (for "View Full Details"), which Slack threads the agent
has joined, and which events it has already handled. That memory lives behind a small
``StateStore`` interface:

- ``LangGraphStateStore``: the LangGraph store on the deployment (the default, shared
  across replicas and restarts).
- ``InMemoryStateStore``: a bounded in-memory fallback, used with a warning when no store
  is reachable.
- Any object with the same three async methods, passed as ``SlackBot(state_store=...)``.

``BridgeState`` wraps a store with the bridge's namespaces and TTLs, so callers never deal
with keys or expiry.
"""

import asyncio
import logging
import math
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Callable, Optional, Protocol

from .tool_calls import ActiveToolCall

logger = logging.getLogger(__name__)

DAY = 24 * 60 * 60


def _now() -> float:
    """The clock bridge state uses for expiry and probe scheduling (seconds)."""
    return time.monotonic()


class StateStore(Protocol):
    """Namespaced key-value storage with per-item expiry.

    ``ttl`` is in seconds; None means the item doesn't expire. Values are JSON-compatible
    dicts. ``get`` returns None for a missing or expired item.
    """

    async def get(self, namespace: str, key: str) -> Optional[dict]: ...

    async def put(
        self, namespace: str, key: str, value: dict, ttl: Optional[float] = None
    ) -> None: ...

    async def delete(self, namespace: str, key: str) -> None: ...


class InMemoryStateStore:
    """Bounded in-memory store. Oldest items are evicted beyond ``max_items``.

    State is lost on restart and isn't shared between replicas.
    """

    def __init__(self, max_items: int = 10_000, clock: Optional[Callable[[], float]] = None):
        self._items: OrderedDict[tuple[str, str], tuple[dict, Optional[float]]] = OrderedDict()
        self._max_items = max_items
        self._clock = clock

    def _time(self) -> float:
        return self._clock() if self._clock else _now()

    def _expired(self, expires_at: Optional[float]) -> bool:
        return expires_at is not None and self._time() >= expires_at

    async def get(self, namespace: str, key: str) -> Optional[dict]:
        entry = self._items.get((namespace, key))
        if entry is None:
            return None
        value, expires_at = entry
        if self._expired(expires_at):
            del self._items[(namespace, key)]
            return None
        return value

    async def put(self, namespace: str, key: str, value: dict, ttl: Optional[float] = None) -> None:
        expires_at = self._time() + ttl if ttl is not None else None
        self._items[(namespace, key)] = (value, expires_at)
        self._items.move_to_end((namespace, key))
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)

    async def delete(self, namespace: str, key: str) -> None:
        self._items.pop((namespace, key), None)

    def live_items(self) -> list[tuple[str, str, dict, Optional[float]]]:
        """Unexpired items as ``(namespace, key, value, remaining_ttl)``, oldest first.

        ``remaining_ttl`` is in seconds, or None for items that don't expire.
        """
        now = self._time()
        return [
            (namespace, key, value, expires_at - now if expires_at is not None else None)
            for (namespace, key), (value, expires_at) in self._items.items()
            if not self._expired(expires_at)
        ]


class LangGraphStateStore:
    """State kept in the LangGraph store of the deployment, via the SDK client.

    Items live under ``("langgraph2slack", <scope>, <namespace>)``. Give each bridge on a
    deployment its own ``scope`` (``SlackBot`` uses its assistant ID) so two bridges never
    read each other's records. Items are never indexed for semantic search.
    """

    PREFIX = "langgraph2slack"

    def __init__(self, langgraph_client, scope: str = "default", supports_ttl: bool = True):
        self._store = langgraph_client.store
        # Namespace labels may not contain periods
        self._scope = scope.replace(".", "_")
        self.supports_ttl = supports_ttl

    def _namespace(self, namespace: str) -> list[str]:
        return [self.PREFIX, self._scope, namespace]

    async def get(self, namespace: str, key: str) -> Optional[dict]:
        try:
            item = await self._store.get_item(self._namespace(namespace), key=key)
        except Exception as e:
            if getattr(e, "status_code", None) == 404:
                return None
            raise
        return item["value"] if item else None

    async def put(self, namespace: str, key: str, value: dict, ttl: Optional[float] = None) -> None:
        # The LangGraph store's TTL is in minutes
        ttl_minutes = math.ceil(ttl / 60) if ttl is not None and self.supports_ttl else None
        await self._store.put_item(
            self._namespace(namespace), key=key, value=value, index=False, ttl=ttl_minutes
        )

    async def delete(self, namespace: str, key: str) -> None:
        await self._store.delete_item(self._namespace(namespace), key=key)


def _rejected(error: Exception) -> bool:
    """True if the store rejected one request (a client error such as an oversized value or
    a bad key), as opposed to an outage (connection error, timeout, 5xx, 408 or 429)."""
    status = getattr(error, "status_code", None)
    return status is not None and 400 <= status < 500 and status not in (408, 429)


def _ttl_unsupported(error: Exception) -> Optional[bool]:
    """Classify a failed TTL write: True if it proves the store doesn't support TTLs,
    None if it might (an opaque server error), False if it's an operational failure.

    In loopback mode (the bridge running inside the Agent Server) the store's own
    ``NotImplementedError("TTL is not supported by ...")`` reaches the client. Over HTTP the
    server turns it into a bare 500, indistinguishable from any other server error.
    """
    if isinstance(error, NotImplementedError) and "TTL" in str(error):
        return True
    if getattr(error, "status_code", None) == 500:
        return None
    return False


class DeploymentStateStore:
    """Uses the deployment's LangGraph store, failing over to a bounded in-memory store.

    - The store is probed on first use. If it's unreachable, or any later read or write
      fails for operational reasons, records go to the in-memory fallback and the store is
      re-probed every ``retry_after`` seconds. A missing key is not a failure, and neither
      is a request the store rejects (an oversized value, say): that one write is dropped.
    - On recovery, unexpired fallback records are copied to the store in the background,
      with their remaining TTL, so records created during the outage stay usable and events
      aren't held up meanwhile. A record the store already holds is kept, never overwritten.
    - A store that doesn't support TTLs would keep every record forever, so the bridge uses
      the in-memory store instead, with a warning. TTL support is decided from an error that
      proves it's missing; transient probe failures never decide it. To keep records in such
      a store anyway, pass ``LangGraphStateStore(client, supports_ttl=False)`` explicitly.

    Records written only to the store before an outage aren't readable until it recovers.
    """

    PROBE_NAMESPACE = "probe"
    PROBE_KEY = "reachable"
    PROBE_TTL = 60
    DRAIN_CONCURRENCY = 8

    def __init__(
        self,
        langgraph_client,
        scope: str = "default",
        retry_after: float = 300,
        probe_timeout: float = 5,
        max_fallback_items: int = 10_000,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._client = langgraph_client
        self._scope = scope
        self._retry_after = retry_after
        self._probe_timeout = probe_timeout
        self._clock = clock
        self._fallback = InMemoryStateStore(max_fallback_items, clock=clock)
        self._durable: Optional[LangGraphStateStore] = None
        self._healthy = False
        self._ttl_unsupported = False
        self._next_probe_at = 0.0
        self._lock = asyncio.Lock()
        self._drain_task: Optional[asyncio.Task] = None

    def _time(self) -> float:
        return self._clock() if self._clock else _now()

    # -- health ------------------------------------------------------------

    async def _probe(self) -> Optional[LangGraphStateStore]:
        """Write a probe item, deciding TTL support on the first successful probe.

        Returns None if the store proved it doesn't support TTLs.
        """
        if self._durable is not None:
            await self._durable.put(self.PROBE_NAMESPACE, self.PROBE_KEY, {}, ttl=self.PROBE_TTL)
            return self._durable

        with_ttl = LangGraphStateStore(self._client, self._scope, supports_ttl=True)
        try:
            await with_ttl.put(self.PROBE_NAMESPACE, self.PROBE_KEY, {}, ttl=self.PROBE_TTL)
            return with_ttl
        except Exception as ttl_error:
            verdict = _ttl_unsupported(ttl_error)
            if verdict is False:
                raise
            without_ttl = LangGraphStateStore(self._client, self._scope, supports_ttl=False)
            await without_ttl.put(self.PROBE_NAMESPACE, self.PROBE_KEY, {})
            await without_ttl.delete(self.PROBE_NAMESPACE, self.PROBE_KEY)
            if verdict is None:
                # An opaque 500: TTLs are unsupported only if a TTL write fails the same way
                # again right after a write without one succeeded.
                try:
                    await with_ttl.put(self.PROBE_NAMESPACE, self.PROBE_KEY, {}, ttl=self.PROBE_TTL)
                    return with_ttl
                except Exception as retry_error:
                    if _ttl_unsupported(retry_error) is False:
                        raise
            logger.warning(
                f"Bridge state: the LangGraph store doesn't support TTLs ({ttl_error}), so "
                f"bridge records would never expire there. Using a bounded in-memory store "
                f"instead: records won't survive a restart or be shared between replicas. "
                f"Enable store TTLs in your deployment config, or pass "
                f"state_store=LangGraphStateStore(client, supports_ttl=False) to keep "
                f"records without expiry."
            )
            return None

    def _fail_over(self, error: Exception) -> None:
        if self._healthy:
            logger.warning(
                f"Bridge state: the LangGraph store failed ({error}). Using a bounded in-memory "
                f"store until it recovers."
            )
        self._healthy = False
        self._next_probe_at = self._time() + self._retry_after

    async def _copy_back(self, namespace, key, value, remaining_ttl) -> None:
        """Copy one fallback record to the store, unless the store already has one."""
        try:
            if await self._durable.get(namespace, key) is None:
                await self._durable.put(namespace, key, value, remaining_ttl)
        except Exception as e:
            if not _rejected(e):
                raise
            logger.warning(
                f"Bridge state: the LangGraph store rejected a record kept during the outage "
                f"({namespace}/{key}); dropping it: {e}"
            )
        await self._fallback.delete(namespace, key)

    async def _drain_fallback(self) -> None:
        """Copy unexpired fallback records to the store, a few at a time.

        Runs in the background after a recovery; reads check the fallback meanwhile. Stops
        (leaving the rest for the next recovery) if the store fails again.
        """
        items = self._fallback.live_items()
        for start in range(0, len(items), self.DRAIN_CONCURRENCY):
            if not self._healthy:
                return
            batch = items[start : start + self.DRAIN_CONCURRENCY]
            results = await asyncio.gather(
                *(self._copy_back(*item) for item in batch), return_exceptions=True
            )
            failure = next((r for r in results if isinstance(r, Exception)), None)
            if failure is not None:
                self._fail_over(failure)
                return

    def _start_drain(self) -> None:
        if self._drain_task is None or self._drain_task.done():
            self._drain_task = asyncio.create_task(self._drain_fallback())

    async def _use_durable(self) -> bool:
        """True if operations should go to the LangGraph store right now."""
        if self._healthy:
            return True
        if self._ttl_unsupported or self._time() < self._next_probe_at:
            return False
        async with self._lock:
            if self._healthy:
                return True
            if self._ttl_unsupported or self._time() < self._next_probe_at:
                return False
            first_probe = self._durable is None and self._next_probe_at == 0.0
            try:
                durable = await asyncio.wait_for(self._probe(), timeout=self._probe_timeout)
            except Exception as e:
                logger.warning(
                    f"Bridge state: the LangGraph store is not reachable ({e!r}). Using a "
                    f"bounded in-memory store until it is: records kept meanwhile won't "
                    f"survive a restart or be shared between replicas."
                )
                self._fail_over(e)
                return False
            if durable is None:
                self._ttl_unsupported = True
                return False
            self._durable = durable
            self._healthy = True
            logger.info(
                "Bridge state: using the LangGraph store"
                if first_probe
                else "Bridge state: the LangGraph store recovered"
            )
            if self._fallback.live_items():
                self._start_drain()
            return True

    # -- StateStore --------------------------------------------------------
    #
    # A request the store rejects is raised to the caller (BridgeState logs it) without a
    # failover: the store is up, it just refused that one record.

    async def get(self, namespace: str, key: str) -> Optional[dict]:
        if await self._use_durable():
            try:
                value = await self._durable.get(namespace, key)
            except Exception as e:
                if _rejected(e):
                    raise
                self._fail_over(e)
            else:
                if value is not None:
                    return value
        # Also covers records not yet copied back after a recovery
        return await self._fallback.get(namespace, key)

    async def put(self, namespace: str, key: str, value: dict, ttl: Optional[float] = None) -> None:
        if await self._use_durable():
            try:
                await self._durable.put(namespace, key, value, ttl)
                return
            except Exception as e:
                if _rejected(e):
                    raise
                self._fail_over(e)
        await self._fallback.put(namespace, key, value, ttl)

    async def delete(self, namespace: str, key: str) -> None:
        await self._fallback.delete(namespace, key)
        if await self._use_durable():
            try:
                await self._durable.delete(namespace, key)
            except Exception as e:
                if _rejected(e):
                    raise
                self._fail_over(e)


class BridgeState:
    """The bridge's memory, by kind. Each kind has its own namespace and TTL.

    A failing store never fails a run: reads degrade to "unknown" and writes are logged.
    """

    FEEDBACK = "feedback"
    TOOL_CALLS = "tool_calls"
    JOINED = "joined_conversations"
    DEDUPE = "duplicate_records"
    STARTED = "started_messages"
    LAST_SEEN = "last_seen_markers"
    NAMES = "name_cache"

    TTLS = {
        FEEDBACK: 30 * DAY,
        TOOL_CALLS: 30 * DAY,
        JOINED: 90 * DAY,
        DEDUPE: 1 * DAY,
        STARTED: 30 * DAY,
        LAST_SEEN: 90 * DAY,
        NAMES: 1 * DAY,
    }

    def __init__(self, store: StateStore):
        self.store = store
        # Serializes claims of the same key within this process (the store can't do it for
        # us): key -> (lock, number of claims holding or waiting for it)
        self._claim_locks: dict[str, tuple[asyncio.Lock, int]] = {}

    @staticmethod
    def _message_key(channel_id: str, ts: str) -> str:
        return f"{channel_id}:{ts}"

    async def _get(self, namespace: str, key: str) -> Optional[dict]:
        try:
            return await self.store.get(namespace, key)
        except Exception as e:
            logger.warning(f"Bridge state read failed ({namespace}/{key}): {e}")
            return None

    async def _put(self, namespace: str, key: str, value: dict) -> None:
        try:
            await self.store.put(namespace, key, value, self.TTLS[namespace])
        except Exception as e:
            logger.warning(f"Bridge state write failed ({namespace}/{key}): {e}")

    # -- feedback mappings -------------------------------------------------

    async def remember_run(self, channel_id: str, message_ts: str, thread_id: str, run_id: str):
        """Record which run produced a Slack message, for feedback buttons."""
        await self._put(
            self.FEEDBACK,
            self._message_key(channel_id, message_ts),
            {"thread_id": thread_id, "run_id": run_id},
        )

    async def run_for_message(self, channel_id: str, message_ts: str) -> Optional[dict]:
        """The ``{"thread_id", "run_id"}`` that produced a Slack message, if known."""
        return await self._get(self.FEEDBACK, self._message_key(channel_id, message_ts))

    # -- tool-call details -------------------------------------------------

    async def save_tool_calls(
        self, channel_id: str, plan_ts: str, calls: list[ActiveToolCall]
    ) -> None:
        """Record the full tool calls behind a plan block, for "View Full Details"."""
        await self._put(
            self.TOOL_CALLS,
            self._message_key(channel_id, plan_ts),
            {"calls": [asdict(c) for c in calls]},
        )

    async def tool_calls(self, channel_id: str, plan_ts: str) -> list[ActiveToolCall]:
        value = await self._get(self.TOOL_CALLS, self._message_key(channel_id, plan_ts))
        return [ActiveToolCall(**c) for c in (value or {}).get("calls", [])]

    # -- joined conversations ----------------------------------------------

    async def mark_joined(self, channel_id: str, thread_ts: str) -> None:
        """Remember that the agent has posted in this Slack thread."""
        await self._put(self.JOINED, self._message_key(channel_id, thread_ts), {})

    async def is_joined(self, channel_id: str, thread_ts: str) -> bool:
        return await self._get(self.JOINED, self._message_key(channel_id, thread_ts)) is not None

    # -- duplicate suppression ---------------------------------------------

    async def claim_trigger(
        self, event_id: Optional[str], channel_id: str, message_ts: str, trigger_kind: str
    ) -> bool:
        """Record that a trigger is being handled. False if it was already recorded.

        The key is the Events API event ID when present, otherwise the message and trigger
        kind. Best effort across replicas: the store has no compare-and-set, so two
        replicas racing on the same redelivery can both claim it.
        """
        key = event_id or f"{self._message_key(channel_id, message_ts)}:{trigger_kind}"
        lock, holders = self._claim_locks.get(key, (asyncio.Lock(), 0))
        self._claim_locks[key] = (lock, holders + 1)
        try:
            async with lock:
                if await self._get(self.DEDUPE, key) is not None:
                    return False
                await self._put(self.DEDUPE, key, {})
                return True
        finally:
            lock, holders = self._claim_locks[key]
            if holders == 1:
                del self._claim_locks[key]
            else:
                self._claim_locks[key] = (lock, holders - 1)

    async def mark_started(self, channel_id: str, message_ts: str) -> None:
        """Remember that a Slack message started a run (edits to it then don't trigger)."""
        await self._put(self.STARTED, self._message_key(channel_id, message_ts), {})

    async def unmark_started(self, channel_id: str, message_ts: str) -> None:
        """Forget that a message started a run (its run failed before completing)."""
        try:
            await self.store.delete(self.STARTED, self._message_key(channel_id, message_ts))
        except Exception as e:
            logger.warning(f"Bridge state delete failed ({self.STARTED}): {e}")

    async def has_started(self, channel_id: str, message_ts: str) -> bool:
        return await self._get(self.STARTED, self._message_key(channel_id, message_ts)) is not None
