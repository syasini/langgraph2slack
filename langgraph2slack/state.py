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

    def __init__(self, max_items: int = 10_000, clock: Callable[[], float] = time.monotonic):
        self._items: OrderedDict[tuple[str, str], tuple[dict, Optional[float]]] = OrderedDict()
        self._max_items = max_items
        self._clock = clock

    async def get(self, namespace: str, key: str) -> Optional[dict]:
        entry = self._items.get((namespace, key))
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and self._clock() >= expires_at:
            del self._items[(namespace, key)]
            return None
        return value

    async def put(self, namespace: str, key: str, value: dict, ttl: Optional[float] = None) -> None:
        expires_at = self._clock() + ttl if ttl is not None else None
        self._items[(namespace, key)] = (value, expires_at)
        self._items.move_to_end((namespace, key))
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)

    async def delete(self, namespace: str, key: str) -> None:
        self._items.pop((namespace, key), None)


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


class DeploymentStateStore:
    """Uses the deployment's LangGraph store when reachable, otherwise the in-memory store.

    Reachability is probed on first use. After a failed probe the in-memory store is used,
    and the probe is retried every ``retry_after`` seconds so a transient failure doesn't
    keep the bridge off the shared store for good.
    """

    PROBE_NAMESPACE = "probe"
    PROBE_KEY = "reachable"

    def __init__(
        self,
        langgraph_client,
        scope: str = "default",
        retry_after: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._client = langgraph_client
        self._scope = scope
        self._retry_after = retry_after
        self._clock = clock
        self._store: Optional[StateStore] = None
        self._fallback: Optional[InMemoryStateStore] = None
        self._next_probe_at = 0.0
        self._lock = asyncio.Lock()

    async def _probe(self, supports_ttl: bool) -> LangGraphStateStore:
        store = LangGraphStateStore(self._client, self._scope, supports_ttl=supports_ttl)
        await store.put(self.PROBE_NAMESPACE, self.PROBE_KEY, {}, ttl=60)
        return store

    async def _resolve(self) -> StateStore:
        if self._store is not None:
            return self._store
        if self._fallback is not None and self._clock() < self._next_probe_at:
            return self._fallback
        async with self._lock:
            if self._store is not None:
                return self._store
            try:
                self._store = await self._probe(supports_ttl=True)
                logger.info("Bridge state: using the LangGraph store")
                return self._store
            except Exception as ttl_error:
                try:
                    self._store = await self._probe(supports_ttl=False)
                    logger.warning(
                        f"Bridge state: the LangGraph store rejected a TTL ({ttl_error}). Using "
                        f"it without expiry; enable store TTLs in your deployment config so "
                        f"bridge records expire."
                    )
                    return self._store
                except Exception as e:
                    logger.warning(
                        f"Bridge state: the LangGraph store is not reachable ({e}). Using a "
                        f"bounded in-memory store until it is: feedback buttons, tool-call "
                        f"details, joined threads and duplicate suppression won't survive "
                        f"restarts or be shared between replicas."
                    )
                    self._fallback = self._fallback or InMemoryStateStore()
                    self._next_probe_at = self._clock() + self._retry_after
                    return self._fallback

    async def get(self, namespace: str, key: str) -> Optional[dict]:
        return await (await self._resolve()).get(namespace, key)

    async def put(self, namespace: str, key: str, value: dict, ttl: Optional[float] = None) -> None:
        await (await self._resolve()).put(namespace, key, value, ttl)

    async def delete(self, namespace: str, key: str) -> None:
        await (await self._resolve()).delete(namespace, key)


class BridgeState:
    """The bridge's memory, by kind. Each kind has its own namespace and TTL.

    A failing store never fails a run: reads degrade to "unknown" and writes are logged.
    """

    FEEDBACK = "feedback"
    TOOL_CALLS = "tool_calls"
    JOINED = "joined_conversations"
    DEDUPE = "duplicate_records"
    LAST_SEEN = "last_seen_markers"
    NAMES = "name_cache"

    TTLS = {
        FEEDBACK: 30 * DAY,
        TOOL_CALLS: 30 * DAY,
        JOINED: 90 * DAY,
        DEDUPE: 1 * DAY,
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
