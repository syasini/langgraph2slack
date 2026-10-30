"""Trigger gate: decides whether an incoming Slack payload starts a run.

A **trigger** is a Slack event that makes the bridge start a run. The gate classifies it by
kind (mention, direct message, joined thread, ...) and records its **provenance**: who
authored it, human or bot. Provenance is passed along to the agent; the bridge enforces
nothing based on it (ADR-0001).
"""

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional

from .state import BridgeState
from .utils import is_bot_mention, is_dm

logger = logging.getLogger(__name__)

TriggerKind = Literal["mention", "direct_message", "joined_thread", "shortcut", "edit"]
AuthorType = Literal["human", "bot"]

# Message subtypes that can carry a trigger. Everything else (joins, topic changes,
# deletions, ...) is ignored.
TRIGGER_SUBTYPES = {None, "bot_message", "file_share", "thread_broadcast"}

# How many thread messages to scan when checking whether the agent has joined a thread
THREAD_HISTORY_LIMIT = 100


@dataclass(frozen=True)
class Provenance:
    """Who authored a trigger, and which human (if any) asked for the run.

    The requester is usually the human author, but not always: a human who points the
    agent at someone else's message is the requester of that run.
    """

    author_type: AuthorType
    author_id: Optional[str] = None
    bot_id: Optional[str] = None
    app_id: Optional[str] = None
    requester_id: Optional[str] = None

    @classmethod
    def from_message(cls, message: dict) -> "Provenance":
        """Read provenance from a Slack message. A message with a bot ID is a bot's;
        a human author is also the requester."""
        bot_id = message.get("bot_id")
        author_type: AuthorType = "bot" if bot_id else "human"
        return cls(
            author_type=author_type,
            author_id=message.get("user"),
            bot_id=bot_id,
            app_id=message.get("app_id") or (message.get("bot_profile") or {}).get("app_id"),
            requester_id=message.get("user") if author_type == "human" else None,
        )


@dataclass(frozen=True)
class Trigger:
    """A Slack event the bridge will start a run for: its kind and its provenance."""

    kind: TriggerKind
    provenance: Provenance


@dataclass(frozen=True)
class BridgeIdentity:
    """The bridge's own Slack identity, from Slack's auth check."""

    user_id: str
    bot_id: Optional[str] = None

    def authored(self, message: dict) -> bool:
        if self.bot_id and message.get("bot_id") == self.bot_id:
            return True
        return bool(message.get("user")) and message.get("user") == self.user_id


class TriggerGate:
    """Decides whether a Slack message event is a trigger, and of what kind."""

    def __init__(
        self,
        slack_client,
        get_identity: Callable[[], Awaitable[BridgeIdentity]],
        state: BridgeState,
    ):
        self._slack = slack_client
        self._get_identity = get_identity
        self._state = state

    async def evaluate(self, event: dict, event_id: Optional[str] = None) -> Optional[Trigger]:
        """Return the trigger for ``event``, or None if it should not start a run.

        A trigger is claimed in bridge state as it's recognized, so a redelivery of the same
        event (same ``event_id``, or same message and trigger kind) returns None.
        """
        trigger = await self._classify(event)
        if trigger is None:
            return None
        if not await self._state.claim_trigger(
            event_id, event.get("channel"), event.get("ts"), trigger.kind
        ):
            logger.info(f"Ignoring duplicate delivery of {event_id or event.get('ts')}")
            return None
        return trigger

    async def _classify(self, event: dict) -> Optional[Trigger]:
        if event.get("subtype") not in TRIGGER_SUBTYPES:
            return None

        identity = await self._get_identity()
        if identity.authored(event):
            return None

        provenance = Provenance.from_message(event)
        if provenance.author_type == "bot":
            return None
        if not provenance.author_id:
            return None

        if is_dm(event):
            return Trigger("direct_message", provenance)

        if is_bot_mention(event.get("text", ""), identity.user_id):
            return Trigger("mention", provenance)

        thread_ts = event.get("thread_ts")
        if thread_ts and await self._is_joined_thread(event.get("channel"), thread_ts, identity):
            return Trigger("joined_thread", provenance)

        return None

    async def _is_joined_thread(
        self, channel_id: str, thread_ts: str, identity: BridgeIdentity
    ) -> bool:
        """True if the agent has already posted in this Slack thread.

        Remembered threads are answered from bridge state; Slack's thread history is read
        only when there's no record, and a hit there is remembered.
        """
        if await self._state.is_joined(channel_id, thread_ts):
            return True
        try:
            history = await self._slack.conversations_replies(
                channel=channel_id, ts=thread_ts, limit=THREAD_HISTORY_LIMIT
            )
        except Exception as e:
            logger.warning(f"Could not check thread history: {e}")
            return False
        if any(identity.authored(m) for m in history.get("messages", [])):
            await self._state.mark_joined(channel_id, thread_ts)
            return True
        return False
