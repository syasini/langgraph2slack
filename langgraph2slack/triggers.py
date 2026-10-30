"""Trigger gate: decides whether an incoming Slack payload starts a run.

A **trigger** is a Slack event that makes the bridge start a run. The gate classifies it by
kind (mention, direct message, joined thread, ...) and records its **provenance**: who
authored it, human or bot. Provenance is passed along to the agent; the bridge enforces
nothing based on it (ADR-0001).
"""

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional

from .normalizer import message_markup
from .state import BridgeState
from .utils import is_bot_mention, is_dm

logger = logging.getLogger(__name__)

TriggerKind = Literal["mention", "direct_message", "joined_thread", "shortcut", "edit"]
AuthorType = Literal["human", "bot"]

# Message subtypes that can carry a trigger. Everything else (joins, topic changes,
# deletions, ...) is ignored. Edits (message_changed) are handled by the edit rule.
TRIGGER_SUBTYPES = {None, "bot_message", "file_share", "thread_broadcast"}
EDIT_SUBTYPE = "message_changed"

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
    """A Slack event the bridge will start a run for.

    ``message`` is the Slack message the trigger is, with its channel: the event itself, or
    for an edit the message as edited. ``occurred_ts`` is when the trigger happened, when
    that's later than the message (an edit, a shortcut).

    A trigger can instead ask about another message, its ``subject`` (the message shortcut):
    then ``request`` is the request text, and the subject is handed over as context.
    """

    kind: TriggerKind
    provenance: Provenance
    message: dict
    occurred_ts: Optional[str] = None
    request: Optional[str] = None
    subject: Optional[dict] = None


@dataclass(frozen=True)
class BotTriggerPolicy:
    """Which bots may trigger the agent (ADR-0007). By default, none.

    Bots must always mention the agent, except allowlisted bots in joined threads when
    ``allowlisted_in_joined_threads`` is set.
    """

    allow: bool = False
    allowlist: frozenset[str] = frozenset()
    allowlisted_in_joined_threads: bool = False

    def __post_init__(self):
        if self.allowlisted_in_joined_threads and not self.allowlist:
            raise ValueError(
                "allowlisted_bots_in_joined_threads requires a bot_trigger_allowlist: it only "
                "applies to allowlisted bots"
            )
        if self.allowlist and not self.allow:
            logger.warning(
                "bot_trigger_allowlist is set but allow_bot_triggers is False: no bot can "
                "trigger the agent"
            )

    def admits(self, provenance: Provenance) -> bool:
        """True if this bot may trigger at all."""
        return self.allow and (not self.allowlist or self.is_allowlisted(provenance))

    def is_allowlisted(self, provenance: Provenance) -> bool:
        return bool({provenance.bot_id, provenance.app_id} & self.allowlist)

    def may_skip_mention_in_joined_threads(self, provenance: Provenance) -> bool:
        return self.allowlisted_in_joined_threads and self.is_allowlisted(provenance)


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
        bot_policy: BotTriggerPolicy = BotTriggerPolicy(),
    ):
        self._slack = slack_client
        self._get_identity = get_identity
        self._state = state
        self._bots = bot_policy

    async def evaluate(self, event: dict, event_id: Optional[str] = None) -> Optional[Trigger]:
        """Return the trigger for ``event``, or None if it should not start a run.

        A trigger is claimed in bridge state as it's recognized, so a redelivery of the same
        event (same ``event_id``, or same message and trigger kind) returns None. The message
        is also recorded as having started a run, which the edit rule checks.
        """
        if event.get("subtype") == EDIT_SUBTYPE:
            trigger = await self._classify_edit(event)
        elif event.get("subtype") in TRIGGER_SUBTYPES:
            trigger = await self._classify_message(event)
        else:
            trigger = None
        if trigger is None:
            return None
        return trigger if await self._claim(trigger, event_id) else None

    async def evaluate_shortcut(
        self, payload: dict, instruction: str, channel_type: str
    ) -> Optional[Trigger]:
        """The trigger for the message shortcut used on a message.

        The shortcut is a human's action: that person authored the trigger and is the
        requester. The message they used it on (any message, a bot's included) is the
        trigger's subject, handed to the agent as context. The run lands in that message's
        thread (a new thread under it if it's top-level), and ``instruction`` is the request.
        """
        subject = payload.get("message") or {}
        channel_id = (payload.get("channel") or {}).get("id")
        requester_id = (payload.get("user") or {}).get("id")
        if not (subject.get("ts") and channel_id and requester_id):
            return None
        subject = {**subject, "channel": channel_id, "channel_type": channel_type}
        message = {
            "type": "message",
            "user": requester_id,
            "channel": channel_id,
            "channel_type": channel_type,
            "ts": subject["ts"],
            "thread_ts": subject.get("thread_ts") or subject["ts"],
        }
        trigger = Trigger(
            "shortcut",
            Provenance.from_message(message),
            message,
            occurred_ts=payload.get("action_ts"),
            request=instruction,
            subject=subject,
        )
        # Slack doesn't retry interactivity payloads; the trigger ID still guards against a
        # double delivery. Without one there's nothing safe to key on, so nothing is claimed.
        trigger_id = payload.get("trigger_id")
        if trigger_id and not await self._claim(trigger, f"shortcut:{trigger_id}"):
            return None
        return trigger

    async def _claim(self, trigger: Trigger, event_id: Optional[str]) -> bool:
        """Claim a trigger (False for a redelivery). A trigger whose own message asks is
        recorded as having started a run (the edit rule checks this); the message a
        shortcut was used on isn't, since someone else wrote it."""
        channel_id, message_ts = trigger.message.get("channel"), trigger.message.get("ts")
        if not await self._state.claim_trigger(event_id, channel_id, message_ts, trigger.kind):
            logger.info(f"Ignoring duplicate delivery of {event_id or message_ts}")
            return False
        if trigger.subject is None:
            await self._state.mark_started(channel_id, message_ts)
        return True

    async def _admit_author(self, message: dict, identity: BridgeIdentity) -> Optional[Provenance]:
        """The message author's provenance if they may trigger the agent at all, else None.

        Shared by new messages and edits, so edits can't bypass author rules.
        """
        if message.get("subtype") not in TRIGGER_SUBTYPES:
            return None
        if identity.authored(message):
            return None
        provenance = Provenance.from_message(message)
        if provenance.author_type == "bot":
            return provenance if self._bots.admits(provenance) else None
        return provenance if provenance.author_id else None

    async def _classify_message(self, message: dict) -> Optional[Trigger]:
        identity = await self._get_identity()
        provenance = await self._admit_author(message, identity)
        if provenance is None:
            return None

        mentioned = is_bot_mention(message_markup(message), identity.user_id)
        thread_ts = message.get("thread_ts")
        channel_id = message.get("channel")

        if provenance.author_type == "bot":
            # Bots must mention the agent (ADR-0007), except allowlisted bots that may
            # converse in joined threads when the developer opted in
            if mentioned:
                return Trigger("mention", provenance, message)
            if (
                thread_ts
                and self._bots.may_skip_mention_in_joined_threads(provenance)
                and await self._is_joined_thread(channel_id, thread_ts, identity)
            ):
                return Trigger("joined_thread", provenance, message)
            return None

        if is_dm(message):
            return Trigger("direct_message", provenance, message)
        if mentioned:
            return Trigger("mention", provenance, message)
        if thread_ts and await self._is_joined_thread(channel_id, thread_ts, identity):
            return Trigger("joined_thread", provenance, message)
        return None

    async def _classify_edit(self, event: dict) -> Optional[Trigger]:
        """An edit triggers only when it adds a mention of the agent to a message that
        hasn't already started a run. Its author is gated like a new message's."""
        message = {
            **(event.get("message") or {}),
            "channel": event.get("channel"),
            "channel_type": event.get("channel_type"),
        }
        previous = event.get("previous_message") or {}
        identity = await self._get_identity()
        provenance = await self._admit_author(message, identity)
        if provenance is None:
            return None
        if not is_bot_mention(message_markup(message), identity.user_id):
            return None
        if is_bot_mention(message_markup(previous), identity.user_id):
            return None
        if await self._state.has_started(message["channel"], message.get("ts")):
            return None
        return Trigger("edit", provenance, message, event.get("event_ts") or event.get("ts"))

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
