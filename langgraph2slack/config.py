"""Configuration management using pydantic-settings.

This module handles all configuration loading from environment variables
with validation and type safety.
"""

import logging
from typing import Optional

from dotenv import find_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    """

    def __init__(self, event: dict):
        """Initialize context from Slack event.

        Args:
            event: Slack event dict from Slack API
        """
        self.event = event

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
