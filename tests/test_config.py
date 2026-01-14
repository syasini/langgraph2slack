"""Unit tests for langgraph2slack.config module.

Tests configuration management including:
- MessageContext properties and state
- BotConfig validation and environment variable loading
"""

import pytest
from pydantic import ValidationError
from langgraph2slack.config import MessageContext, BotConfig


# ============================================================================
# Tests for MessageContext
# ============================================================================


class TestMessageContext:
    """Tests for MessageContext initialization and properties."""

    # Happy path tests - Basic property extraction
    # ------------------------------------------------------------------------

    def test_init_from_dm_event(self, sample_dm_event):
        """MessageContext should extract all fields from DM event."""
        context = MessageContext(sample_dm_event)

        assert context.user_id == "U456USER"
        assert context.channel_id == "D789DM"
        assert context.message_ts == "1234567.890"
        assert context.thread_ts is None  # No thread in DM
        assert context.channel_type == "im"

    def test_init_from_channel_event(self, sample_channel_event):
        """MessageContext should extract all fields from channel event."""
        context = MessageContext(sample_channel_event)

        assert context.user_id == "U456USER"
        assert context.channel_id == "C123CHANNEL"
        assert context.message_ts == "1234567.890"
        assert context.thread_ts is None  # Not in thread
        assert context.channel_type == "channel"

    def test_init_from_thread_event(self, sample_thread_event):
        """MessageContext should extract thread_ts from threaded message."""
        context = MessageContext(sample_thread_event)

        assert context.user_id == "U456USER"
        assert context.channel_id == "C123CHANNEL"
        assert context.message_ts == "1234567.999"
        assert context.thread_ts == "1234567.890"  # Parent message timestamp
        assert context.channel_type == "channel"

    # Property tests - is_dm
    # ------------------------------------------------------------------------

    def test_is_dm_true(self, sample_dm_event):
        """MessageContext.is_dm should return True for DMs."""
        context = MessageContext(sample_dm_event)
        assert context.is_dm is True

    def test_is_dm_false_for_channel(self, sample_channel_event):
        """MessageContext.is_dm should return False for channels."""
        context = MessageContext(sample_channel_event)
        assert context.is_dm is False

    def test_is_dm_false_for_group(self, sample_group_event):
        """MessageContext.is_dm should return False for groups."""
        context = MessageContext(sample_group_event)
        assert context.is_dm is False

    @pytest.mark.parametrize("channel_type,expected_is_dm", [
        ("im", True),           # Direct message
        ("channel", False),     # Public channel
        ("group", False),       # Private group
        ("mpim", False),        # Multi-person IM
    ])
    def test_is_dm_various_channel_types(self, channel_type, expected_is_dm):
        """MessageContext.is_dm should correctly identify DMs across channel types."""
        event = {
            "user": "U123",
            "channel": "C456",
            "channel_type": channel_type,
            "ts": "12345.678"
        }
        context = MessageContext(event)
        assert context.is_dm is expected_is_dm

    # Property tests - is_thread
    # ------------------------------------------------------------------------

    def test_is_thread_true(self, sample_thread_event):
        """MessageContext.is_thread should return True when thread_ts present."""
        context = MessageContext(sample_thread_event)
        assert context.is_thread is True

    def test_is_thread_false_no_thread(self, sample_dm_event):
        """MessageContext.is_thread should return False when no thread_ts."""
        context = MessageContext(sample_dm_event)
        assert context.is_thread is False

    def test_is_thread_false_for_parent_message(self):
        """MessageContext.is_thread should return False for thread parent message.

        The parent message in a thread does not have thread_ts set.
        """
        event = {
            "user": "U123",
            "channel": "C456",
            "channel_type": "channel",
            "text": "Starting a thread",
            "ts": "1234567.890"
            # No thread_ts
        }
        context = MessageContext(event)
        assert context.is_thread is False

    # Critical negative tests - Missing or malformed data
    # ------------------------------------------------------------------------

    def test_missing_user_field(self):
        """Missing user field should default to empty string (graceful handling)."""
        event = {
            "channel": "C123",
            "channel_type": "channel",
            "ts": "12345.678"
            # No 'user' field
        }
        context = MessageContext(event)
        assert context.user_id == ""

    def test_missing_channel_field(self):
        """Missing channel field should default to empty string."""
        event = {
            "user": "U123",
            "channel_type": "channel",
            "ts": "12345.678"
            # No 'channel' field
        }
        context = MessageContext(event)
        assert context.channel_id == ""

    def test_missing_ts_field(self):
        """Missing ts field should default to empty string."""
        event = {
            "user": "U123",
            "channel": "C456",
            "channel_type": "channel"
            # No 'ts' field
        }
        context = MessageContext(event)
        assert context.message_ts == ""

    def test_missing_channel_type_defaults_to_channel(self):
        """Missing channel_type should default to 'channel'."""
        event = {
            "user": "U123",
            "channel": "C456",
            "ts": "12345.678"
            # No 'channel_type' field
        }
        context = MessageContext(event)
        assert context.channel_type == "channel"
        assert context.is_dm is False

    def test_empty_event_dict(self):
        """Empty event dict should create context with default values (graceful)."""
        event = {}
        context = MessageContext(event)

        assert context.user_id == ""
        assert context.channel_id == ""
        assert context.message_ts == ""
        assert context.thread_ts is None
        assert context.channel_type == "channel"
        assert context.is_dm is False
        assert context.is_thread is False

    def test_none_values_in_event(self):
        """None values in event should be handled gracefully."""
        event = {
            "user": None,
            "channel": None,
            "ts": None,
            "channel_type": None,
            "thread_ts": None
        }
        context = MessageContext(event)

        # get() returns None, but we should handle it gracefully
        # The current implementation will return None or ""
        # Let's verify it doesn't crash
        assert context.user_id == "" or context.user_id is None
        assert context.channel_id == "" or context.channel_id is None


# ============================================================================
# Tests for BotConfig
# ============================================================================


class TestBotConfig:
    """Tests for BotConfig validation and environment variable loading."""

    # Happy path tests
    # ------------------------------------------------------------------------

    def test_load_valid_config(self, valid_env_vars):
        """Valid environment variables should load successfully."""
        config = BotConfig()

        assert config.ASSISTANT_ID == "test-assistant"
        assert config.LANGGRAPH_URL == "http://localhost:8123"

    def test_get_slack_bot_token(self, valid_env_vars):
        """get_slack_bot_token() should return plain string."""
        config = BotConfig()
        token = config.get_slack_bot_token()

        assert token == "xoxb-test-token-123"
        assert isinstance(token, str)

    def test_get_slack_signing_secret(self, valid_env_vars):
        """get_slack_signing_secret() should return plain string."""
        config = BotConfig()
        secret = config.get_slack_signing_secret()

        assert secret == "test-signing-secret-456"
        assert isinstance(secret, str)

    def test_optional_langgraph_url_none(self, monkeypatch):
        """LANGGRAPH_URL should default to None when not provided."""
        # Set only required fields
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret-test")
        monkeypatch.setenv("ASSISTANT_ID", "test-assistant")

        config = BotConfig()
        assert config.LANGGRAPH_URL is None

    def test_custom_langgraph_url(self, monkeypatch):
        """Custom LANGGRAPH_URL should be loaded correctly."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret-test")
        monkeypatch.setenv("ASSISTANT_ID", "test-assistant")
        monkeypatch.setenv("LANGGRAPH_URL", "https://my-deployment.langgraph.app")

        config = BotConfig()
        assert config.LANGGRAPH_URL == "https://my-deployment.langgraph.app"

    # Critical negative tests - Validation errors
    # ------------------------------------------------------------------------

    def test_missing_slack_bot_token(self, monkeypatch):
        """Missing SLACK_BOT_TOKEN should raise ValidationError."""
        # Set all except SLACK_BOT_TOKEN
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret-test")
        monkeypatch.setenv("ASSISTANT_ID", "test-assistant")

        with pytest.raises(ValidationError) as exc_info:
            BotConfig()

        # Verify error mentions the missing field
        assert "SLACK_BOT_TOKEN" in str(exc_info.value)

    def test_missing_slack_signing_secret(self, monkeypatch):
        """Missing SLACK_SIGNING_SECRET should raise ValidationError."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("ASSISTANT_ID", "test-assistant")

        with pytest.raises(ValidationError) as exc_info:
            BotConfig()

        assert "SLACK_SIGNING_SECRET" in str(exc_info.value)

    def test_missing_assistant_id(self, monkeypatch):
        """Missing ASSISTANT_ID should raise ValidationError."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret-test")

        with pytest.raises(ValidationError) as exc_info:
            BotConfig()

        assert "ASSISTANT_ID" in str(exc_info.value)

    def test_all_missing_raises_validation_error(self):
        """Missing all required fields should raise ValidationError."""
        # Don't set any environment variables
        with pytest.raises(ValidationError) as exc_info:
            BotConfig()

        error_str = str(exc_info.value)
        # Should mention all required fields
        assert "SLACK_BOT_TOKEN" in error_str
        assert "SLACK_SIGNING_SECRET" in error_str
        assert "ASSISTANT_ID" in error_str

    def test_empty_string_values_fail_validation(self, monkeypatch):
        """Empty string values should fail validation (not allowed)."""
        # Pydantic's SecretStr and regular str should reject empty values
        monkeypatch.setenv("SLACK_BOT_TOKEN", "")  # Empty string
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret-test")
        monkeypatch.setenv("ASSISTANT_ID", "test-assistant")

        # Empty string should fail validation
        # Note: Depending on pydantic settings, this might pass or fail
        # The config has env_ignore_empty=True, so empty values are ignored
        # and treated as missing, which should raise ValidationError
        with pytest.raises(ValidationError):
            BotConfig()

    def test_secret_str_not_exposed_in_repr(self, valid_env_vars):
        """SecretStr fields should not expose values in repr/str."""
        config = BotConfig()

        # Convert to string (repr) and verify secrets are hidden
        config_str = str(config)
        config_repr = repr(config)

        # Secret values should NOT appear in string representation
        assert "xoxb-test-token-123" not in config_str
        assert "test-signing-secret-456" not in config_str
        assert "xoxb-test-token-123" not in config_repr
        assert "test-signing-secret-456" not in config_repr

        # But ASSISTANT_ID (not secret) might appear
        # This is OK as it's not sensitive
