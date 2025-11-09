"""Unit tests for SlackBot core functionality (Tier 1).

Tests the core SlackBot methods that are essential for bot operation:
- Decorator methods (@transform_input, @transform_output, @transform_metadata)
- Metadata building (_build_metadata)
- Reaction helpers (_add_reaction, _remove_reaction)

This completes testing of the core orchestration logic.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from lg2slack.bot import SlackBot
from lg2slack.config import MessageContext


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def minimal_bot():
    """Create a minimal SlackBot instance for testing.

    Uses patch to avoid full initialization while keeping core functionality.
    """
    with patch.object(SlackBot, '_setup_slack_handlers'), \
         patch.object(SlackBot, '_create_fastapi_app'):

        with patch.dict('os.environ', {
            'SLACK_BOT_TOKEN': 'xoxb-test-token',
            'SLACK_SIGNING_SECRET': 'test-secret',
            'ASSISTANT_ID': 'test-assistant',
        }):
            bot = SlackBot()
            return bot


@pytest.fixture
def sample_context():
    """Create a sample MessageContext for testing."""
    return MessageContext({
        "user": "U123USER",
        "channel": "C456CHANNEL",
        "channel_type": "channel",
        "ts": "1234567890.123456"
    })


# ============================================================================
# Tests: Decorator Methods
# ============================================================================


class TestDecoratorMethods:
    """Tests for @transform_input, @transform_output, @transform_metadata decorators."""

    def test_transform_input_adds_to_chain(self, minimal_bot):
        """Decorating with @transform_input should add function to input transformer chain."""
        initial_count = len(minimal_bot._input_transformers)

        @minimal_bot.transform_input
        async def my_transformer(message: str) -> str:
            return message.upper()

        # Should have added one transformer
        assert len(minimal_bot._input_transformers) == initial_count + 1

    def test_transform_output_adds_to_chain(self, minimal_bot):
        """Decorating with @transform_output should add function to output transformer chain."""
        initial_count = len(minimal_bot._output_transformers)

        @minimal_bot.transform_output
        async def add_footer(message: str) -> str:
            return f"{message}\n_Footer_"

        # Should have added one transformer
        assert len(minimal_bot._output_transformers) == initial_count + 1

    def test_transform_metadata_adds_to_chain(self, minimal_bot):
        """Decorating with @transform_metadata should add function to metadata transformer chain."""
        initial_count = len(minimal_bot._metadata_transformers)

        @minimal_bot.transform_metadata
        async def custom_metadata(context: MessageContext) -> dict:
            return {"user_id": context.user_id}

        # Should have added one transformer
        assert len(minimal_bot._metadata_transformers) == initial_count + 1

    def test_multiple_input_transformers(self, minimal_bot):
        """Multiple @transform_input decorators should all be added."""
        initial_count = len(minimal_bot._input_transformers)

        @minimal_bot.transform_input
        async def first(message: str) -> str:
            return f"[1] {message}"

        @minimal_bot.transform_input
        async def second(message: str) -> str:
            return f"[2] {message}"

        @minimal_bot.transform_input
        async def third(message: str) -> str:
            return f"[3] {message}"

        # Should have added 3 transformers
        assert len(minimal_bot._input_transformers) == initial_count + 3

    @pytest.mark.asyncio
    async def test_decorated_input_transformer_is_called(self, minimal_bot, sample_context):
        """Decorated input transformer should be called when applying transforms."""
        @minimal_bot.transform_input
        async def uppercase_transform(message: str) -> str:
            return message.upper()

        result = await minimal_bot._input_transformers.apply("hello world", sample_context)

        assert result == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_decorated_output_transformer_is_called(self, minimal_bot, sample_context):
        """Decorated output transformer should be called when applying transforms."""
        @minimal_bot.transform_output
        async def add_footer(message: str) -> str:
            return f"{message}\n\n_Powered by AI_"

        result = await minimal_bot._output_transformers.apply("Response", sample_context)

        assert "Response" in result
        assert "_Powered by AI_" in result

    @pytest.mark.asyncio
    async def test_decorated_transformers_execute_in_order(self, minimal_bot, sample_context):
        """Multiple decorated transformers should execute in registration order."""
        @minimal_bot.transform_input
        async def add_prefix(message: str) -> str:
            return f"[PREFIX] {message}"

        @minimal_bot.transform_input
        async def add_suffix(message: str) -> str:
            return f"{message} [SUFFIX]"

        result = await minimal_bot._input_transformers.apply("test", sample_context)

        # Order: prefix -> suffix
        assert result == "[PREFIX] test [SUFFIX]"

    @pytest.mark.asyncio
    async def test_decorator_with_context_parameter(self, minimal_bot, sample_context):
        """Decorated transformer with context parameter should receive it."""
        @minimal_bot.transform_input
        async def add_user_context(message: str, context: MessageContext) -> str:
            return f"User {context.user_id}: {message}"

        result = await minimal_bot._input_transformers.apply("hello", sample_context)

        assert result == "User U123USER: hello"


# ============================================================================
# Tests: _build_metadata()
# ============================================================================


class TestBuildMetadata:
    """Tests for metadata building from Slack context."""

    @pytest.mark.asyncio
    async def test_build_metadata_default_fields(self, minimal_bot, sample_context):
        """Should include all default Slack fields in metadata."""
        metadata = await minimal_bot._build_metadata(sample_context)

        # Verify all default fields are present
        assert metadata["slack_user_id"] == "U123USER"
        assert metadata["slack_channel_id"] == "C456CHANNEL"
        assert metadata["slack_message_ts"] == "1234567890.123456"
        assert metadata["slack_thread_ts"] is None
        assert metadata["slack_channel_type"] == "channel"
        assert metadata["slack_is_dm"] is False
        assert metadata["slack_is_thread"] is False

    @pytest.mark.asyncio
    async def test_build_metadata_with_thread(self, minimal_bot):
        """Should include thread information when message is in a thread."""
        thread_context = MessageContext({
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "ts": "9999999.999",
            "thread_ts": "1234567.890"  # In a thread
        })

        metadata = await minimal_bot._build_metadata(thread_context)

        assert metadata["slack_thread_ts"] == "1234567.890"
        assert metadata["slack_is_thread"] is True

    @pytest.mark.asyncio
    async def test_build_metadata_dm_context(self, minimal_bot):
        """Should correctly identify DM in metadata."""
        dm_context = MessageContext({
            "user": "U123USER",
            "channel": "D789DM",
            "channel_type": "im",
            "ts": "1234567.890"
        })

        metadata = await minimal_bot._build_metadata(dm_context)

        assert metadata["slack_is_dm"] is True
        assert metadata["slack_channel_type"] == "im"

    @pytest.mark.asyncio
    async def test_build_metadata_include_false_returns_empty(self, minimal_bot, sample_context):
        """When include_metadata=False, should return empty dict."""
        # Create bot with include_metadata=False
        with patch.object(SlackBot, '_setup_slack_handlers'), \
             patch.object(SlackBot, '_create_fastapi_app'):

            with patch.dict('os.environ', {
                'SLACK_BOT_TOKEN': 'xoxb-test',
                'SLACK_SIGNING_SECRET': 'test-secret',
                'ASSISTANT_ID': 'test-assistant',
            }):
                bot = SlackBot(include_metadata=False)

        metadata = await bot._build_metadata(sample_context)

        # Should return empty dict
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_build_metadata_with_custom_transformer(self, minimal_bot, sample_context):
        """Should use custom metadata transformer when provided."""
        @minimal_bot.transform_metadata
        async def custom_metadata(data: dict, context: MessageContext) -> dict:
            # Metadata transformers get (data, context) signature
            return {
                "custom_user": context.user_id,
                "custom_channel": context.channel_id,
                "custom_field": "test_value"
            }

        metadata = await minimal_bot._build_metadata(sample_context)

        # Should use custom transformer output
        assert metadata["custom_user"] == "U123USER"
        assert metadata["custom_channel"] == "C456CHANNEL"
        assert metadata["custom_field"] == "test_value"

        # Should NOT have default fields
        assert "slack_user_id" not in metadata

    @pytest.mark.asyncio
    async def test_build_metadata_multiple_transformers(self, minimal_bot, sample_context):
        """Multiple metadata transformers should be applied in order."""
        @minimal_bot.transform_metadata
        async def first_transform(data: dict, context: MessageContext) -> dict:
            # First transformer receives empty dict and adds fields
            return {"step": "first", "user_id": context.user_id}

        @minimal_bot.transform_metadata
        async def second_transform(data: dict, context: MessageContext) -> dict:
            # Second transformer receives output from first and can modify it
            data["step"] = "second"
            data["channel_id"] = context.channel_id
            return data

        metadata = await minimal_bot._build_metadata(sample_context)

        # Both transformers should have been applied
        assert metadata["step"] == "second"  # Modified by second transformer
        assert metadata["user_id"] == "U123USER"  # From first transformer
        assert metadata["channel_id"] == "C456CHANNEL"  # Added by second transformer


# ============================================================================
# Tests: Reaction Helpers
# ============================================================================


class TestReactionHelpers:
    """Tests for _add_reaction and _remove_reaction methods."""

    @pytest.mark.asyncio
    async def test_add_reaction_success(self, minimal_bot):
        """Should successfully add emoji reaction to message."""
        # Mock Slack client
        mock_slack_client = MagicMock()
        mock_slack_client.client.reactions_add = AsyncMock()
        minimal_bot.slack_app = mock_slack_client

        await minimal_bot._add_reaction(
            channel_id="C123CHANNEL",
            message_ts="1234567.890",
            emoji="eyes"
        )

        # Verify reactions_add was called correctly
        mock_slack_client.client.reactions_add.assert_called_once_with(
            channel="C123CHANNEL",
            timestamp="1234567.890",
            name="eyes"
        )

    @pytest.mark.asyncio
    async def test_add_reaction_failure_does_not_raise(self, minimal_bot):
        """Should log warning but NOT raise if adding reaction fails."""
        # Mock Slack client to fail
        mock_slack_client = MagicMock()
        mock_slack_client.client.reactions_add = AsyncMock(
            side_effect=Exception("Slack API error")
        )
        minimal_bot.slack_app = mock_slack_client

        # Should NOT raise - just logs warning
        await minimal_bot._add_reaction(
            channel_id="C123",
            message_ts="1234567.890",
            emoji="hourglass"
        )

        # If we get here without exception, test passes

    @pytest.mark.asyncio
    async def test_remove_reaction_success(self, minimal_bot):
        """Should successfully remove emoji reaction from message."""
        # Mock Slack client
        mock_slack_client = MagicMock()
        mock_slack_client.client.reactions_remove = AsyncMock()
        minimal_bot.slack_app = mock_slack_client

        await minimal_bot._remove_reaction(
            channel_id="C123CHANNEL",
            message_ts="1234567.890",
            emoji="eyes"
        )

        # Verify reactions_remove was called correctly
        mock_slack_client.client.reactions_remove.assert_called_once_with(
            channel="C123CHANNEL",
            timestamp="1234567.890",
            name="eyes"
        )

    @pytest.mark.asyncio
    async def test_remove_reaction_failure_does_not_raise(self, minimal_bot):
        """Should log warning but NOT raise if removing reaction fails."""
        # Mock Slack client to fail
        mock_slack_client = MagicMock()
        mock_slack_client.client.reactions_remove = AsyncMock(
            side_effect=Exception("Slack API error")
        )
        minimal_bot.slack_app = mock_slack_client

        # Should NOT raise - just logs warning
        await minimal_bot._remove_reaction(
            channel_id="C123",
            message_ts="1234567.890",
            emoji="hourglass"
        )

        # If we get here without exception, test passes

    @pytest.mark.asyncio
    async def test_reaction_helpers_with_different_emojis(self, minimal_bot):
        """Should handle various emoji names correctly."""
        mock_slack_client = MagicMock()
        mock_slack_client.client.reactions_add = AsyncMock()
        minimal_bot.slack_app = mock_slack_client

        emojis = ["eyes", "hourglass", "robot_face", "white_check_mark", "x"]

        for emoji in emojis:
            await minimal_bot._add_reaction("C123", "1234567.890", emoji)

            # Verify emoji name was passed correctly (without colons)
            call_kwargs = mock_slack_client.client.reactions_add.call_args.kwargs
            assert call_kwargs["name"] == emoji
