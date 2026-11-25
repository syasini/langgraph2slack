"""Collects backfill: earlier Slack thread messages handed to the agent as untrusted context.

- **First trigger in a conversation** (no LangGraph thread exists yet): the thread's earlier
  messages. The thread's parent (often the alert being discussed) is always kept; the rest
  are the most recent messages that fit the message-count and character budgets.
- **Later triggers:** only messages posted since the agent last saw the conversation (its
  last-seen marker, or else its last reply in the thread), excluding the agent's own
  messages.
- Either way, messages that themselves started runs are left out (except the parent): the
  agent's conversation memory holds those, or will once their run starts.
- **After each bridge-started run** the conversation's last-seen marker advances to the
  trigger.
"""

import asyncio
import dataclasses
import logging
from typing import Optional

from .files import REFERENCE_COST
from .normalizer import NormalizedMessage, SlackMessageNormalizer
from .state import BridgeState, KeyedLocks, is_not_found
from .triggers import BridgeIdentity

logger = logging.getLogger(__name__)

# Thread reads: up to this many pages of up to 1000 messages each
MAX_THREAD_PAGES = 5
THREAD_PAGE_SIZE = 1000

# Per-message overhead counted against the character budget (time, name, ID labels)
LINE_OVERHEAD = 40
# A message is truncated to fit the remaining budget only if at least this much fits
MIN_TRUNCATED_CHARS = 200
TRUNCATION_MARK = " … [truncated]"


def ts_key(ts: Optional[str]) -> tuple[int, int]:
    """A Slack timestamp as a sortable value ("1700000000.000100" -> (1700000000, 100)).

    One that doesn't parse (a corrupt marker, say) sorts as the oldest possible.
    """
    seconds, _, micros = (ts or "0").partition(".")
    try:
        return int(seconds or 0), int(micros or 0)
    except ValueError:
        return 0, 0


class BackfillCollector:
    def __init__(
        self,
        slack_client,
        langgraph_client,
        state: BridgeState,
        normalizer: SlackMessageNormalizer,
        *,
        enabled: bool = True,
        max_messages: int = 20,
        max_chars: int = 8000,
    ):
        self._slack = slack_client
        self._langgraph = langgraph_client
        self._state = state
        self._normalizer = normalizer
        self.enabled = enabled
        self.max_messages = max_messages
        self.max_chars = max_chars
        # Marker updates for one conversation happen one at a time, so it never moves back
        self._marker_locks = KeyedLocks()

    async def collect(
        self,
        *,
        channel_id: str,
        conversation_ts: str,
        trigger_ts: str,
        thread_id: str,
        identity: BridgeIdentity,
        until_ts: Optional[str] = None,
        subject: Optional[dict] = None,
    ) -> list[NormalizedMessage]:
        """The backfill for a trigger, oldest first (empty when backfill is off).

        Args:
            channel_id: Slack channel of the conversation
            conversation_ts: Root of the Slack thread the conversation maps to
            trigger_ts: Timestamp of the message the trigger is about
            thread_id: The conversation's LangGraph thread ID
            identity: The bridge's own Slack identity
            until_ts: When the trigger happened, if later than ``trigger_ts`` (an edit):
                messages up to then count as earlier
            subject: The message the trigger asks about (the message shortcut's target). It's
                always part of the backfill, kept within the budget first.
        """
        if not self.enabled:
            # A shortcut's message is the context the agent is asked about: always sent
            return await self.subject_only(subject) if subject else []
        until = ts_key(until_ts or trigger_ts)
        first_contact = not await self._conversation_exists(thread_id)
        subject_ts = subject.get("ts") if subject else None
        parent_ts = conversation_ts if first_contact else None

        if first_contact and conversation_ts == trigger_ts and until_ts is None and not subject:
            return []  # a top-level trigger starts its thread: nothing came before it

        def wanted(message: dict) -> bool:
            if message.get("ts") == subject_ts:
                return True
            if message.get("ts") == trigger_ts:
                return False
            return ts_key(message.get("ts")) < until

        marker = None if first_contact else await self._state.last_seen(channel_id, conversation_ts)
        # With a marker, only messages after it need reading
        messages = await self._thread_messages(channel_id, conversation_ts, oldest=marker)
        if subject and not any(m.get("ts") == subject_ts for m in messages):
            # Not read from the thread (unreadable, or before the marker): use the payload's copy
            messages = sorted([*messages, subject], key=lambda m: ts_key(m.get("ts")))
        if first_contact:
            earlier = [m for m in messages if wanted(m)]
        else:
            if marker is None:
                marker = self._last_reply_ts(messages, identity)
            earlier = [
                m
                for m in messages
                if wanted(m)
                and (
                    m.get("ts") == subject_ts
                    or (
                        (marker is None or ts_key(m.get("ts")) > ts_key(marker))
                        and not identity.authored(m)
                    )
                )
            ]

        # The thread's parent (first contact) and the subject, once each, always stay; other
        # messages that started runs of their own are left out
        # The subject first: it's what the agent is asked about
        keep = list(dict.fromkeys(ts for ts in (subject_ts, parent_ts) if ts))
        checked = [m for m in earlier if m.get("ts") not in keep]
        started = await asyncio.gather(
            *(self._state.has_started(channel_id, m.get("ts")) for m in checked)
        )
        started_ts = {m.get("ts") for m, was_started in zip(checked, started) if was_started}
        earlier = [m for m in earlier if m.get("ts") not in started_ts]
        return await self._within_budget(earlier, keep=keep)

    async def subject_only(self, subject: dict) -> list[NormalizedMessage]:
        """Just the message a shortcut was used on, within the budget (backfill off, or
        collecting it failed)."""
        return await self._within_budget([subject], keep=[subject.get("ts")])

    async def advance(self, channel_id: str, conversation_ts: str, seen_ts: str) -> None:
        """Move the conversation's last-seen marker forward to ``seen_ts``."""
        async with self._marker_locks.hold(f"{channel_id}:{conversation_ts}"):
            current = await self._state.last_seen(channel_id, conversation_ts)
            if current is None or ts_key(seen_ts) > ts_key(current):
                await self._state.set_last_seen(channel_id, conversation_ts, seen_ts)

    async def _within_budget(
        self, messages: list[dict], keep: list[str] = ()
    ) -> list[NormalizedMessage]:
        """The most recent messages that fit both budgets, normalized, oldest first.

        The messages at the ``keep`` timestamps (the thread's parent, the subject) are kept
        first. A message too long
        for the remaining budget is truncated to fit when enough room is left. Messages are
        normalized newest first and only until the budget is spent, so files on messages
        that won't fit are never downloaded.
        """
        kept: list[NormalizedMessage] = []
        remaining = self.max_chars
        # Images past the per-run limit are never downloaded
        image_slots = self._normalizer.file_limits.max_images

        async def take(raw: dict) -> bool:
            """Add a message if it fits; False once the budget is spent."""
            nonlocal remaining, image_slots
            if len(kept) >= self.max_messages:
                return False
            message = await self._normalizer.normalize(raw, read_files=False)
            if not (message.text or raw.get("files")):
                return True  # nothing to show: skip it, keep going
            overhead = len(message.display_name) + LINE_OVERHEAD
            text_room = remaining - overhead
            if text_room <= 0:
                return False
            # Every file costs at least a reference line: leave room for those
            references = REFERENCE_COST * len(raw.get("files") or [])
            if len(message.text) + references > text_room:
                if text_room - references < MIN_TRUNCATED_CHARS:
                    return False
                cut = text_room - references - len(TRUNCATION_MARK)
                message = dataclasses.replace(message, text=message.text[:cut] + TRUNCATION_MARK)
            # Files are read only now, with what's left of the budget: files that can't fit
            # are listed by reference without being downloaded
            room = text_room - len(message.text)
            files = await self._normalizer.read_files(
                raw, budget_chars=room, image_slots=image_slots
            )
            # Inline what fits, in order, keeping room for the later files' references
            fitted = []
            for index, part in enumerate(files):
                later_references = REFERENCE_COST * (len(files) - index - 1)
                if part.is_inlined and part.cost() > room - later_references:
                    part = part.as_reference("not inlined: backfill budget")
                fitted.append(part)
                room -= part.cost()
            files = tuple(fitted)
            files_cost = sum(f.cost() for f in files)
            image_slots -= sum(1 for f in files if f.kind == "image")
            message = dataclasses.replace(message, files=files)
            kept.append(message)
            remaining -= overhead + len(message.text) + files_cost
            return True

        kept_first = [m for ts in keep for m in messages if m.get("ts") == ts]
        for message in kept_first:
            await take(message)
        for message in reversed(messages):
            if any(message is k for k in kept_first):
                continue
            if not await take(message):
                break
        return sorted(kept, key=lambda m: ts_key(m.ts))

    @staticmethod
    def _last_reply_ts(messages: list[dict], identity: BridgeIdentity) -> Optional[str]:
        replies = [m.get("ts") for m in messages if identity.authored(m)]
        return max(replies, key=ts_key) if replies else None

    async def _conversation_exists(self, thread_id: str) -> bool:
        try:
            await self._langgraph.threads.get(thread_id)
            return True
        except Exception as e:
            if is_not_found(e):
                return False
            # Unknown: don't hand over the full thread again
            logger.warning(f"Could not check LangGraph thread {thread_id}: {e}")
            return True

    async def _thread_messages(
        self, channel_id: str, thread_ts: str, oldest: Optional[str] = None
    ) -> list[dict]:
        """A Slack thread's messages (parent included; after ``oldest`` if given), oldest
        first."""
        by_ts: dict[str, dict] = {}
        cursor = None
        for _ in range(MAX_THREAD_PAGES):
            kwargs = {"channel": channel_id, "ts": thread_ts, "limit": THREAD_PAGE_SIZE}
            if oldest:
                kwargs["oldest"] = oldest
            if cursor:
                kwargs["cursor"] = cursor
            try:
                page = await self._slack.conversations_replies(**kwargs)
            except Exception as e:
                logger.warning(f"Could not read thread {channel_id}:{thread_ts} for backfill: {e}")
                break
            # Slack repeats the parent message on every page
            for message in page.get("messages", []):
                by_ts[message.get("ts")] = message
            cursor = (page.get("response_metadata") or {}).get("next_cursor")
            if not page.get("has_more") or not cursor:
                break
        return sorted(by_ts.values(), key=lambda m: ts_key(m.get("ts")))
