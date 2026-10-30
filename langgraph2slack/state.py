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


def is_not_found(error: Exception) -> bool:
    """True if a LangGraph SDK error is a 404 (missing item or thread), across SDK versions."""
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status == 404


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
            if is_not_found(e):
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
      re-probed every ``retry_after`` seconds. A missing key is not a failure.
    - On recovery, unexpired fallback records are copied to the store with their remaining
      TTL, so records created during the outage stay usable. A record the store already
      holds is kept, never overwritten.
    - TTL support is decided once, from an error that proves it's missing; transient probe
      failures never disable expiry.

    Records written only to the store before an outage aren't readable until it recovers.
    """

    PROBE_NAMESPACE = "probe"
    PROBE_KEY = "reachable"
    PROBE_TTL = 60

    def __init__(
        self,
        langgraph_client,
        scope: str = "default",
        retry_after: float = 300,
        max_fallback_items: int = 10_000,
        clock: Optional[Callable[[], float]] = None,
    ):
        self._client = langgraph_client
        self._scope = scope
        self._retry_after = retry_after
        self._clock = clock
        self._fallback = InMemoryStateStore(max_fallback_items, clock=clock)
        self._durable: Optional[LangGraphStateStore] = None
        self._healthy = False
        self._next_probe_at = 0.0
        self._lock = asyncio.Lock()

    def _time(self) -> float:
        return self._clock() if self._clock else _now()

    # -- health ------------------------------------------------------------

    async def _probe(self) -> LangGraphStateStore:
        """Write a probe item, deciding TTL support on the first successful probe."""
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
                f"Bridge state: the LangGraph store doesn't support TTLs ({ttl_error}). "
                f"Using it without expiry; enable store TTLs in your deployment config so "
                f"bridge records expire."
            )
            return without_ttl

    def _fail_over(self, error: Exception) -> None:
        if self._healthy:
            logger.warning(
                f"Bridge state: the LangGraph store failed ({error}). Using a bounded in-memory "
                f"store until it recovers."
            )
        self._healthy = False
        self._next_probe_at = self._time() + self._retry_after

    async def _drain_fallback(self) -> None:
        """Copy unexpired fallback records to the store, keeping any record it already has."""
        for namespace, key, value, remaining_ttl in self._fallback.live_items():
            if await self._durable.get(namespace, key) is None:
                await self._durable.put(namespace, key, value, remaining_ttl)
            await self._fallback.delete(namespace, key)

    async def _use_durable(self) -> bool:
        """True if operations should go to the LangGraph store right now."""
        if self._healthy:
            return True
        if self._time() < self._next_probe_at:
            return False
        async with self._lock:
            if self._healthy:
                return True
            if self._time() < self._next_probe_at:
                return False
            first_probe = self._durable is None and self._next_probe_at == 0.0
            try:
                self._durable = await self._probe()
                await self._drain_fallback()
            except Exception as e:
                logger.warning(
                    f"Bridge state: the LangGraph store is not reachable ({e}). Using a bounded "
                    f"in-memory store until it is: records kept meanwhile won't survive a "
                    f"restart or be shared between replicas."
                )
                self._fail_over(e)
                return False
            self._healthy = True
            logger.info(
                "Bridge state: using the LangGraph store"
                if first_probe
                else "Bridge state: the LangGraph store recovered"
            )
            return True

    # -- StateStore --------------------------------------------------------

    async def get(self, namespace: str, key: str) -> Optional[dict]:
        if await self._use_durable():
            try:
                value = await self._durable.get(namespace, key)
            except Exception as e:
                self._fail_over(e)
            else:
                if value is not None:
                    return value
        # Also covers records written to the fallback while a recovery was in progress
        return await self._fallback.get(namespace, key)

    async def put(self, namespace: str, key: str, value: dict, ttl: Optional[float] = None) -> None:
        if await self._use_durable():
            try:
                await self._durable.put(namespace, key, value, ttl)
                return
            except Exception as e:
                self._fail_over(e)
        await self._fallback.put(namespace, key, value, ttl)

    async def delete(self, namespace: str, key: str) -> None:
        await self._fallback.delete(namespace, key)
        if await self._use_durable():
            try:
                await self._durable.delete(namespace, key)
            except Exception as e:
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
        # Serializes duplicate claims within this process (the store can't do it for us)
        self._claim_lock = asyncio.Lock()

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

    async def save_tool_calls(self, plan_ts: str, calls: list[ActiveToolCall]) -> None:
        await self._put(self.TOOL_CALLS, plan_ts, {"calls": [asdict(c) for c in calls]})

    async def tool_calls(self, plan_ts: str) -> list[ActiveToolCall]:
        value = await self._get(self.TOOL_CALLS, plan_ts)
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
        async with self._claim_lock:
            if await self._get(self.DEDUPE, key) is not None:
                return False
            await self._put(self.DEDUPE, key, {})
            return True

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

    # -- name cache --------------------------------------------------------

    async def cached_name(self, kind: str, slack_id: str) -> Optional[str]:
        """A cached user or channel name (``kind`` is "user" or "channel")."""
        value = await self._get(self.NAMES, f"{kind}:{slack_id}")
        return value.get("name") if value else None

    async def cache_name(self, kind: str, slack_id: str, name: str) -> None:
        await self._put(self.NAMES, f"{kind}:{slack_id}", {"name": name})

    # -- last-seen markers -------------------------------------------------

    async def last_seen(self, channel_id: str, conversation_ts: str) -> Optional[str]:
        """Timestamp of the latest Slack message the agent has seen in a conversation."""
        value = await self._get(self.LAST_SEEN, self._message_key(channel_id, conversation_ts))
        return value.get("ts") if value else None

    async def set_last_seen(self, channel_id: str, conversation_ts: str, ts: str) -> None:
        await self._put(self.LAST_SEEN, self._message_key(channel_id, conversation_ts), {"ts": ts})
