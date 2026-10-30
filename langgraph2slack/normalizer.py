"""Turns Slack messages into clean markdown the agent can read.

Slack messages arrive as Slack markup: mentions look like ``<@U123>``, links like
``<https://…|label>``, channels like ``<#C123>``, HTML entities stay escaped, and many bot
posts keep their content in attachments or blocks rather than the top-level text. The
normalizer reads all of those and produces a ``NormalizedMessage``.

User and channel names are looked up through Slack and cached in bridge state. A lookup
that fails (a missing scope, a deleted user) falls back to the raw ID and never blocks a run.
"""

import html
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from .state import BridgeState

logger = logging.getLogger(__name__)

# <@U123> or <@U123|name>
_USER_MENTION = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>]*)?>")
# <#C123> or <#C123|name>
_CHANNEL_REF = re.compile(r"<#([CGD][A-Z0-9]+)(?:\|([^>]*))?>")
# <!here>, <!channel>, <!everyone>, <!subteam^S123|@team>, <!date^…|fallback>
_SPECIAL = re.compile(r"<!([^>|]+)(?:\|([^>]*))?>")
# <https://example.com|label>, <https://example.com>, <mailto:a@b.c|a@b.c>
_LINK = re.compile(r"<((?:https?|mailto|ftp|tel|slack):[^>|]+)(?:\|([^>]*))?>")

# Block types whose text is a human-readable rendering of the message
_TEXT_BLOCKS = {"section", "header", "context", "rich_text"}

# How long a failed name lookup is remembered (in this process) before Slack is asked again
FAILED_LOOKUP_TTL = 600


@dataclass(frozen=True)
class NormalizedMessage:
    """A Slack message as clean markdown, with who wrote it."""

    ts: str
    author_type: Literal["human", "bot"]
    author_id: Optional[str]  # user ID, or bot ID for classic integrations
    display_name: str
    text: str
    files: tuple = field(default_factory=tuple)

    @property
    def is_bot(self) -> bool:
        return self.author_type == "bot"


class SlackMessageNormalizer:
    """Normalizes Slack messages, resolving user and channel names (cached)."""

    def __init__(self, slack_client, state: BridgeState):
        self._slack = slack_client
        self._state = state
        # IDs whose lookup failed recently -> when to try again (bounded)
        self._failed_lookups: dict[str, float] = {}

    def _recently_failed(self, key: str) -> bool:
        retry_at = self._failed_lookups.get(key)
        return retry_at is not None and time.monotonic() < retry_at

    def _remember_failure(self, key: str) -> None:
        if len(self._failed_lookups) >= 1000:
            self._failed_lookups.clear()
        self._failed_lookups[key] = time.monotonic() + FAILED_LOOKUP_TTL

    async def normalize(
        self, message: dict, *, strip_mention_of: Optional[str] = None
    ) -> NormalizedMessage:
        """Normalize one Slack message.

        Args:
            message: A Slack message (from an event or read back from the thread)
            strip_mention_of: A user ID whose mentions are removed (the bridge's own, for
                the request text)
        """
        raw = message_markup(message)
        if strip_mention_of:
            raw = re.sub(rf"<@{strip_mention_of}(?:\|[^>]*)?>", "", raw)
        text = await self._convert(raw)

        bot_id = message.get("bot_id")
        author_type = "bot" if bot_id else "human"
        author_id = message.get("user") or bot_id
        return NormalizedMessage(
            ts=message.get("ts", ""),
            author_type=author_type,
            author_id=author_id,
            display_name=await self._author_name(message),
            text=_tidy(text),
        )

    # -- markup conversion -------------------------------------------------

    async def _convert(self, text: str) -> str:
        text = await _sub_async(_USER_MENTION, text, self._user_mention)
        text = await _sub_async(_CHANNEL_REF, text, self._channel_ref)
        text = _SPECIAL.sub(_special, text)
        text = _LINK.sub(_link, text)
        return html.unescape(text)

    async def _user_mention(self, match: re.Match) -> str:
        user_id = match.group(1)
        name = await self.user_name(user_id)
        return f"@{name} ({user_id})" if name != user_id else f"@{user_id}"

    async def _channel_ref(self, match: re.Match) -> str:
        channel_id, label = match.group(1), match.group(2)
        return f"#{label or await self.channel_name(channel_id)}"

    # -- names (cached) ----------------------------------------------------

    async def user_name(self, user_id: str) -> str:
        """A user's display name, or the raw ID if it can't be looked up."""
        cached = await self._state.cached_name("user", user_id)
        if cached:
            return cached
        if self._recently_failed(f"user:{user_id}"):
            return user_id
        try:
            user = (await self._slack.users_info(user=user_id))["user"]
        except Exception as e:
            logger.debug(f"Could not look up user {user_id}: {e}")
            self._remember_failure(f"user:{user_id}")
            return user_id
        profile = user.get("profile") or {}
        name = (
            profile.get("display_name")
            or profile.get("real_name")
            or user.get("real_name")
            or user.get("name")
            or user_id
        )
        await self._state.cache_name("user", user_id, name)
        return name

    async def channel_name(self, channel_id: str) -> str:
        """A channel's name, or the raw ID if it can't be looked up."""
        cached = await self._state.cached_name("channel", channel_id)
        if cached:
            return cached
        if self._recently_failed(f"channel:{channel_id}"):
            return channel_id
        try:
            name = (await self._slack.conversations_info(channel=channel_id))["channel"]["name"]
        except Exception as e:
            logger.debug(f"Could not look up channel {channel_id}: {e}")
            self._remember_failure(f"channel:{channel_id}")
            return channel_id
        await self._state.cache_name("channel", channel_id, name)
        return name

    async def _author_name(self, message: dict) -> str:
        if message.get("bot_id"):
            profile = message.get("bot_profile") or {}
            return (
                profile.get("name")
                or message.get("username")
                or (await self.user_name(message["user"]) if message.get("user") else None)
                or message["bot_id"]
            )
        user_id = message.get("user")
        return await self.user_name(user_id) if user_id else "unknown"


def message_markup(message: dict) -> str:
    """A message's content as Slack markup: top-level text, text-bearing blocks and
    attachments.

    Parts that repeat one already included (top-level text usually mirrors the blocks or
    the attachment fallback) are skipped. Link unfurls are skipped too: Slack adds them to
    show a linked page, so they're not something the author wrote.
    """
    parts: list[str] = []
    seen: set[str] = set()

    def add(text: Optional[str]) -> None:
        key = _dedupe_key(text or "")
        if key and key not in seen:
            seen.add(key)
            parts.append(text)

    add(message.get("text"))
    for block in message.get("blocks") or []:
        # Rich text blocks are the rendering of a typed message: the same content as the
        # top-level text, so they're only read when that's missing
        if block.get("type") == "rich_text" and message.get("text"):
            continue
        for text in _block_texts(block):
            add(text)
    for attachment in message.get("attachments") or []:
        if _is_unfurl(attachment):
            continue
        add(attachment.get("pretext"))
        add(attachment.get("title"))
        # The fallback summarizes the attachment; use it only when there's no text
        add(attachment.get("text") or attachment.get("fallback"))
        for f in attachment.get("fields") or []:
            title, value = f.get("title"), f.get("value")
            add(f"{title}: {value}" if title and value else (title or value))
        add(attachment.get("footer"))
    return "\n".join(parts)


def _dedupe_key(text: str) -> str:
    """Text reduced for duplicate detection, ignoring user mentions and spacing (top-level
    text often repeats an attachment fallback with a mention added)."""
    return " ".join(_USER_MENTION.sub("", text).split())


def _is_unfurl(attachment: dict) -> bool:
    """True for a link preview Slack attached (a page, or a quoted Slack message)."""
    return bool(
        attachment.get("from_url")
        or attachment.get("original_url")
        or attachment.get("is_msg_unfurl")
        or attachment.get("is_app_unfurl")
    )


def _block_texts(block: dict) -> list[str]:
    """Readable text from a Slack block (section, header, context, rich text)."""
    kind = block.get("type")
    if kind not in _TEXT_BLOCKS:
        return []
    if kind == "rich_text":
        return [_rich_text(block)]
    texts = []
    if isinstance(block.get("text"), dict):
        texts.append(block["text"].get("text", ""))
    for f in block.get("fields") or []:
        texts.append(f.get("text", ""))
    for element in block.get("elements") or []:
        if isinstance(element, dict) and element.get("type") in ("mrkdwn", "plain_text"):
            texts.append(element.get("text", ""))
    return [t for t in texts if t]


def _rich_text(block: dict) -> str:
    """Flatten a rich_text block back to Slack markup (mentions and links as <...>)."""

    def render(element: dict) -> str:
        kind = element.get("type")
        if kind == "text":
            return element.get("text", "")
        if kind == "user":
            return f"<@{element.get('user_id')}>"
        if kind == "channel":
            return f"<#{element.get('channel_id')}>"
        if kind == "link":
            label = element.get("text")
            return f"<{element.get('url')}|{label}>" if label else f"<{element.get('url')}>"
        if kind == "emoji":
            return f":{element.get('name')}:"
        if kind == "broadcast":
            return f"<!{element.get('range')}>"
        if kind in ("rich_text_section", "rich_text_quote", "rich_text_preformatted"):
            inner = "".join(render(e) for e in element.get("elements") or [])
            if kind == "rich_text_preformatted":
                return f"```\n{inner}\n```"
            if kind == "rich_text_quote":
                return "> " + inner
            return inner
        if kind == "rich_text_list":
            return "\n".join(f"- {render(item)}" for item in element.get("elements") or [])
        return ""

    return "\n".join(render(e) for e in block.get("elements") or [])


def _special(match: re.Match) -> str:
    command, label = match.group(1), match.group(2)
    if label:
        return label
    if command in ("here", "channel", "everyone"):
        return f"@{command}"
    return command


def _link(match: re.Match) -> str:
    url, label = match.group(1), match.group(2)
    if url.startswith("mailto:"):
        return label or url[len("mailto:") :]
    if not label or label == url:
        return url
    return f"[{label}]({url})"


def _tidy(text: str) -> str:
    """Collapse runs of spaces left behind by stripped mentions and trim the ends."""
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


async def _sub_async(pattern: re.Pattern, text: str, replace) -> str:
    """re.sub with an async replacement function."""
    out, last = [], 0
    for match in pattern.finditer(text):
        out.append(text[last : match.start()])
        out.append(await replace(match))
        last = match.end()
    out.append(text[last:])
    return "".join(out)
