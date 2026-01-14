"""Unit tests for SlackBot._should_process_message() method.

Tests the message filtering logic that determines whether the bot should
respond to a Slack message based on:
- DM detection
- Bot mentions
- Thread participation
- Bot message filtering
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from langgraph2slack.bot import SlackBot


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_slack_client():
    """Create a mock Slack client with common API responses.

    This mocks the async Slack API calls used in _should_process_message:
    - auth_test(): Returns bot's user ID
    - conversations_replies(): Returns thread message history
    """
    mock = MagicMock()

    # Default auth_test response (bot user ID)
    mock.auth_test = AsyncMock(return_value={
        "ok": True,
        "url": "https://test-workspace.slack.com/",
        "team": "Test Team",
        "user": "test_bot",
        "team_id": "T123TEAM",
        "user_id": "B123BOT",  # Bot's user ID
    })

    # Default conversations_replies response (empty thread)
    mock.conversations_replies = AsyncMock(return_value={
        "ok": True,
        "messages": [],
    })

    return mock


@pytest.fixture
def mock_slack_app(mock_slack_client):
    """Create a mock Slack app with the mocked client."""
    mock_app = MagicMock()
    mock_app.client = mock_slack_client
    return mock_app


@pytest.fixture
def minimal_bot():
    """Create a minimal SlackBot instance for testing.

    Uses patch to avoid initializing FastAPI, Slack handlers, etc.
    We only need the _should_process_message method to work.
    """
    # Patch the initialization to avoid setting up full infrastructure
    with patch.object(SlackBot, '_setup_slack_handlers'), \
         patch.object(SlackBot, '_create_fastapi_app'):

        # Mock environment variables
        with patch.dict('os.environ', {
            'SLACK_BOT_TOKEN': 'xoxb-test-token',
            'SLACK_SIGNING_SECRET': 'test-secret',
            'ASSISTANT_ID': 'test-assistant',
        }):
            bot = SlackBot()

            # We'll inject the mock slack_app in individual tests
            bot._bot_user_id = None  # Reset to ensure lazy loading is tested

            return bot


@pytest.fixture
def bot_with_mock_slack(minimal_bot, mock_slack_app):
    """SlackBot with mocked Slack client ready for testing."""
    minimal_bot.slack_app = mock_slack_app
    return minimal_bot


# ============================================================================
# Tests: Basic Filtering (Skip Invalid Messages)
# ============================================================================


class TestBasicFiltering:
    """Tests for filtering out invalid messages."""

    @pytest.mark.asyncio
    async def test_skip_message_without_user(self, bot_with_mock_slack):
        """Messages without user field should be skipped (system messages)."""
        event = {
            "channel": "C123CHANNEL",
            "text": "System notification",
            "ts": "1234567.890",
            # No 'user' field
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_skip_message_from_bot(self, bot_with_mock_slack):
        """Messages with bot_id should be skipped (from other bots)."""
        event = {
            "user": "U123USER",
            "bot_id": "B456OTHERBOT",  # From a bot
            "channel": "C123CHANNEL",
            "text": "Bot message",
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_skip_message_with_empty_user(self, bot_with_mock_slack):
        """Messages with empty string user should be skipped."""
        event = {
            "user": "",  # Empty user
            "channel": "C123CHANNEL",
            "text": "Test",
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_skip_message_with_none_user(self, bot_with_mock_slack):
        """Messages with None user should be skipped."""
        event = {
            "user": None,  # None user
            "channel": "C123CHANNEL",
            "text": "Test",
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False


# ============================================================================
# Tests: Direct Messages (Always Process)
# ============================================================================


class TestDirectMessages:
    """Tests for DM detection and processing."""

    @pytest.mark.asyncio
    async def test_process_dm_with_im_channel_type(self, bot_with_mock_slack):
        """DMs (channel_type='im') should always be processed."""
        event = {
            "user": "U123USER",
            "channel": "D456DM",
            "channel_type": "im",  # Direct message
            "text": "Hello bot",
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_dm_loads_bot_user_id(self, bot_with_mock_slack, mock_slack_client):
        """Processing DM should trigger lazy loading of bot_user_id."""
        event = {
            "user": "U123USER",
            "channel": "D456DM",
            "channel_type": "im",
            "text": "Hello",
            "ts": "1234567.890",
        }

        # bot_user_id should be None initially
        assert bot_with_mock_slack._bot_user_id is None

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

        # Should have called auth_test to get bot user ID
        mock_slack_client.auth_test.assert_called_once()

        # Should have cached the bot_user_id
        assert bot_with_mock_slack._bot_user_id == "B123BOT"

    @pytest.mark.asyncio
    async def test_dm_uses_cached_bot_user_id(self, bot_with_mock_slack, mock_slack_client):
        """Second call should use cached bot_user_id, not call auth_test again."""
        event = {
            "user": "U123USER",
            "channel": "D456DM",
            "channel_type": "im",
            "text": "Hello",
            "ts": "1234567.890",
        }

        # First call
        await bot_with_mock_slack._should_process_message(event)
        assert mock_slack_client.auth_test.call_count == 1

        # Second call - should NOT call auth_test again
        await bot_with_mock_slack._should_process_message(event)
        assert mock_slack_client.auth_test.call_count == 1  # Still 1, not 2


# ============================================================================
# Tests: Bot Mentions (Process When Mentioned)
# ============================================================================


class TestBotMentions:
    """Tests for bot mention detection."""

    @pytest.mark.asyncio
    async def test_process_channel_message_with_mention(self, bot_with_mock_slack):
        """Channel messages with @bot mention should be processed."""
        # Pre-set bot_user_id to avoid auth_test call
        bot_with_mock_slack._bot_user_id = "B123BOT"

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "text": "<@B123BOT> hello there!",  # Bot mentioned
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_process_mention_at_end(self, bot_with_mock_slack):
        """Bot mention at end of message should be detected."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "text": "Thanks for the help <@B123BOT>",  # Mention at end
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_process_mention_in_middle(self, bot_with_mock_slack):
        """Bot mention in middle of message should be detected."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "text": "Hey <@B123BOT> can you help?",  # Mention in middle
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_skip_channel_message_without_mention(self, bot_with_mock_slack):
        """Channel messages WITHOUT bot mention should be skipped."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "text": "Just a regular message",  # No mention
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_skip_mention_of_different_user(self, bot_with_mock_slack):
        """Mentioning a different user should not trigger processing."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "text": "<@U999OTHER> hello",  # Different user mentioned
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False


# ============================================================================
# Tests: Thread Participation
# ============================================================================


class TestThreadParticipation:
    """Tests for processing messages in threads where bot has participated."""

    @pytest.mark.asyncio
    async def test_process_thread_where_bot_participated(self, bot_with_mock_slack, mock_slack_client):
        """Messages in threads where bot has replied should be processed."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        # Mock thread history showing bot has participated
        mock_slack_client.conversations_replies = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {"user": "U123USER", "text": "Original question", "ts": "1234567.890"},
                {"user": "B123BOT", "text": "Bot response", "ts": "1234567.891"},  # Bot participated!
                {"user": "U123USER", "text": "Follow up", "ts": "1234567.892"},
            ]
        })

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "text": "Another follow up",  # No mention needed
            "ts": "1234567.893",
            "thread_ts": "1234567.890",  # In a thread
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

        # Should have checked thread history
        mock_slack_client.conversations_replies.assert_called_once_with(
            channel="C456CHANNEL",
            ts="1234567.890",
            limit=100
        )

    @pytest.mark.asyncio
    async def test_skip_thread_where_bot_not_participated(self, bot_with_mock_slack, mock_slack_client):
        """Messages in threads where bot hasn't replied should be skipped."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        # Mock thread history WITHOUT bot participation
        mock_slack_client.conversations_replies = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {"user": "U123USER", "text": "Original question", "ts": "1234567.890"},
                {"user": "U456USER", "text": "Someone else responds", "ts": "1234567.891"},
            ]
        })

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "text": "Follow up",  # No mention
            "ts": "1234567.892",
            "thread_ts": "1234567.890",  # In a thread
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_process_thread_without_participation_but_mentioned(self, bot_with_mock_slack, mock_slack_client):
        """Even if bot hasn't participated, @mention should trigger processing."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        # Mock thread history WITHOUT bot participation
        mock_slack_client.conversations_replies = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {"user": "U123USER", "text": "Original question", "ts": "1234567.890"},
            ]
        })

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            "text": "<@B123BOT> can you help?",  # Bot mentioned!
            "ts": "1234567.891",
            "thread_ts": "1234567.890",  # In a thread
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_thread_history_with_bot_message_first(self, bot_with_mock_slack, mock_slack_client):
        """Bot's message at beginning of thread should be detected."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        mock_slack_client.conversations_replies = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {"user": "B123BOT", "text": "Bot started thread", "ts": "1234567.890"},  # Bot first
                {"user": "U123USER", "text": "User reply", "ts": "1234567.891"},
            ]
        })

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "text": "Another reply",
            "ts": "1234567.892",
            "thread_ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_thread_history_with_bot_message_last(self, bot_with_mock_slack, mock_slack_client):
        """Bot's message at end of thread should be detected."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        mock_slack_client.conversations_replies = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {"user": "U123USER", "text": "Question", "ts": "1234567.890"},
                {"user": "U456USER", "text": "Comment", "ts": "1234567.891"},
                {"user": "B123BOT", "text": "Bot answer", "ts": "1234567.892"},  # Bot last
            ]
        })

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "text": "Thanks!",
            "ts": "1234567.893",
            "thread_ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_thread_history_lookup_fails_falls_back_to_mention(self, bot_with_mock_slack, mock_slack_client):
        """If thread history lookup fails, should fall back to mention check."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        # Make conversations_replies raise an exception
        mock_slack_client.conversations_replies = AsyncMock(
            side_effect=Exception("Slack API error")
        )

        # Message WITH mention (should process due to fallback)
        event_with_mention = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "text": "<@B123BOT> help",
            "ts": "1234567.892",
            "thread_ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event_with_mention)
        assert result is True  # Mention check succeeds

        # Message WITHOUT mention (should skip)
        event_without_mention = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "text": "Just a message",
            "ts": "1234567.893",
            "thread_ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event_without_mention)
        assert result is False  # No mention, and history lookup failed

    @pytest.mark.asyncio
    async def test_thread_with_other_bot_messages(self, bot_with_mock_slack, mock_slack_client):
        """Thread with other bots but not THIS bot should be skipped."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        # Thread has a different bot
        mock_slack_client.conversations_replies = AsyncMock(return_value={
            "ok": True,
            "messages": [
                {"user": "U123USER", "text": "Question", "ts": "1234567.890"},
                {"bot_id": "B999OTHER", "text": "Other bot response", "ts": "1234567.891"},  # Different bot
            ]
        })

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "text": "Follow up",
            "ts": "1234567.892",
            "thread_ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        # Should skip - only OTHER bot participated, not THIS bot
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_thread_history(self, bot_with_mock_slack, mock_slack_client):
        """Thread with empty message history should skip."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        # Empty thread
        mock_slack_client.conversations_replies = AsyncMock(return_value={
            "ok": True,
            "messages": []  # Empty
        })

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "text": "Message",
            "ts": "1234567.891",
            "thread_ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False


# ============================================================================
# Tests: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_message_with_empty_text(self, bot_with_mock_slack):
        """Message with empty text should still be evaluated (DM check, etc)."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        # DM with empty text - should still process (it's a DM)
        event = {
            "user": "U123USER",
            "channel": "D456DM",
            "channel_type": "im",
            "text": "",  # Empty text
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True  # DM should be processed regardless of text

    @pytest.mark.asyncio
    async def test_message_without_text_field(self, bot_with_mock_slack):
        """Message without text field should be handled gracefully."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        event = {
            "user": "U123USER",
            "channel": "C456CHANNEL",
            "channel_type": "channel",
            # No 'text' field
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        # No text = no mention = skip (not a DM either)
        assert result is False

    @pytest.mark.asyncio
    async def test_group_message_without_mention(self, bot_with_mock_slack):
        """Group messages (mpim) without mention should be skipped."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        event = {
            "user": "U123USER",
            "channel": "G789GROUP",
            "channel_type": "group",  # Private group
            "text": "Group discussion",
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_group_message_with_mention(self, bot_with_mock_slack):
        """Group messages with @mention should be processed."""
        bot_with_mock_slack._bot_user_id = "B123BOT"

        event = {
            "user": "U123USER",
            "channel": "G789GROUP",
            "channel_type": "group",
            "text": "<@B123BOT> help needed",  # Mentioned
            "ts": "1234567.890",
        }

        result = await bot_with_mock_slack._should_process_message(event)

        assert result is True
