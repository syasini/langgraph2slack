"""Configuration management using pydantic-settings.

This module handles all configuration loading from environment variables
with validation and type safety.
"""

import logging
from typing import TYPE_CHECKING, Optional

from dotenv import find_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .triggers import Provenance, Trigger

if TYPE_CHECKING:
    from .framing import RequestedBy
    from .normalizer import NormalizedMessage

logger = logging.getLogger(__name__)


class BotConfig(BaseSettings):
    """Bot configuration loaded from environment variables.

    All settings can be provided via environment variables or .env file.

    Required:
        SLACK_BOT_TOKEN: Slack bot token (xoxb-...)
        SLACK_SIGNING_SECRET: Slack signing secret for request verification
        ASSISTANT_ID: LangGraph assistant ID

    Optional:
        LANGGRAPH_URL: LangGraph deployment URL (None = loopback, default)

    Example .env:
        SLACK_BOT_TOKEN=xoxb-123-456-abc
        SLACK_SIGNING_SECRET=abc123def456
        ASSISTANT_ID=my-assistant
    """

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),  # Automatically find .env file
        env_ignore_empty=True,  # Ignore empty env vars
        extra="ignore",  # Ignore extra env vars we don't care about
    )

    # Slack credentials (required)
    SLACK_BOT_TOKEN: SecretStr
    SLACK_SIGNING_SECRET: SecretStr

    # LangGraph settings (required)
    ASSISTANT_ID: str

    # Optional settings
    LANGGRAPH_URL: Optional[str] = None  # None = use loopback (on platform)

    # Multitask strategy for concurrent runs on the same thread.
    # One of: "enqueue" (finish current, then process next), "interrupt" (cancel in-flight run),
    # "reject", "rollback". Defaults to "enqueue" for safety (especially with tool-using agents).
    MULTITASK_STRATEGY: str = "enqueue"

    def get_slack_bot_token(self) -> str:
        """Get Slack bot token as plain string."""
        return self.SLACK_BOT_TOKEN.get_secret_value()

    def get_slack_signing_secret(self) -> str:
        """Get Slack signing secret as plain string."""
        return self.SLACK_SIGNING_SECRET.get_secret_value()


class MessageContext:
    """Context information passed to transformers.

    Contains metadata about the current Slack message, user, and channel.
    Used by input/output transformers to make context-aware decisions.

    Attributes:
        event: Raw Slack event dict
        user_id: Slack user ID
        channel_id: Slack channel ID
        message_ts: Message timestamp
        thread_ts: Thread timestamp (None if not in thread)
        channel_type: Channel type ('im', 'channel', 'group')
        trigger_kind: How the run was triggered ('mention', 'direct_message', ...)
        author_type: 'human' or 'bot'
        requester_id: The human who triggered the run (None for bot triggers)
    """

    def __init__(self, event: dict, trigger: Optional[Trigger] = None):
        """Initialize context from Slack event.

        Args:
            event: Slack event dict from Slack API
            trigger: The trigger the gate recognized. When omitted, provenance is read
                from the event itself and ``trigger_kind`` is None.
        """
        self.event = event
        self._trigger_kind = trigger.kind if trigger else None
        self._provenance = trigger.provenance if trigger else Provenance.from_message(event)

        # Filled in by SlackBot before the run: who the request line names, and the earlier
        # Slack messages handed over as backfill (oldest first)
        self.requested_by: Optional["RequestedBy"] = None
        self.backfill: tuple["NormalizedMessage", ...] = ()

        # Extract basic info from event (no API calls needed)
        self._user_id = event.get("user", "")
        self._channel_id = event.get("channel", "")
        self._message_ts = event.get("ts", "")
        self._thread_ts = event.get("thread_ts")
        self._channel_type = event.get("channel_type", "channel")

    @property
    def user_id(self) -> str:
        """Slack user ID (e.g., U123ABC456)."""
        return self._user_id

    @property
    def channel_id(self) -> str:
        """Slack channel ID (e.g., C123ABC456)."""
        return self._channel_id

    @property
    def message_ts(self) -> str:
        """Message timestamp (e.g., '1234567890.123456')."""
        return self._message_ts

    @property
    def thread_ts(self) -> Optional[str]:
        """Thread timestamp if message is in a thread, None otherwise."""
        return self._thread_ts

    @property
    def channel_type(self) -> str:
        """Channel type: 'im' (DM), 'channel', 'group', or 'mpim'."""
        return self._channel_type

    @property
    def is_dm(self) -> bool:
        """True if this is a direct message."""
        return self._channel_type == "im"

    @property
    def is_thread(self) -> bool:
        """True if this message is in a thread."""
        return self._thread_ts is not None

    @property
    def conversation_ts(self) -> str:
        """Root of the Slack thread this conversation maps to: the thread if the message is
        a reply, otherwise the message itself."""
        return self._thread_ts or self._message_ts

    @property
    def trigger_kind(self) -> Optional[str]:
        """How the run was triggered: 'mention', 'direct_message', 'joined_thread',
        'shortcut' or 'edit'."""
        return self._trigger_kind

    @property
    def provenance(self) -> Provenance:
        """Who authored the trigger, and the requester's ID."""
        return self._provenance

    @property
    def author_type(self) -> str:
        """'human' or 'bot': who authored the trigger."""
        return self._provenance.author_type

    @property
    def author_id(self) -> Optional[str]:
        """Slack user ID of the trigger's author (a bot user for modern apps; None for
        classic integrations)."""
        return self._provenance.author_id

    @property
    def bot_id(self) -> Optional[str]:
        """Bot ID of the trigger's author, if a bot authored it."""
        return self._provenance.bot_id

    @property
    def app_id(self) -> Optional[str]:
        """App ID of the trigger's author, if a Slack app authored it."""
        return self._provenance.app_id

    @property
    def requester_id(self) -> Optional[str]:
        """The human who triggered the run, or None for bot triggers."""
        return self._provenance.requester_id

    def slack_run_config(self) -> dict:
        """The ``configurable["slack"]`` entry of the run config: the trigger kind, its
        provenance, and the channel and thread identifiers.

        ``thread_ts`` is Slack's raw thread timestamp (None for a top-level message), as in
        metadata. ``conversation_ts`` is the root of the Slack thread this conversation maps
        to: the thread if the trigger is a reply, otherwise the trigger message itself.
        """
        return {
            "trigger_kind": self.trigger_kind,
            "author_type": self.author_type,
            "author_id": self.author_id,
            "bot_id": self.bot_id,
            "app_id": self.app_id,
            "requester_id": self.requester_id,
            "channel_id": self.channel_id,
            "channel_type": self.channel_type,
            "message_ts": self.message_ts,
            "thread_ts": self.thread_ts,
            "conversation_ts": self.conversation_ts,
        }
