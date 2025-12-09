"""Shared pytest fixtures for langgraph2slack tests.

This module provides reusable fixtures for testing langgraph2slack components.
Fixtures are automatically discovered by pytest and available to all test files.
"""

import pytest
from langgraph2slack.config import MessageContext


# ============================================================================
# Slack Event Fixtures
# ============================================================================


@pytest.fixture
def sample_dm_event():
    """Sample direct message (DM) event from Slack.

    Used for testing DM detection and message context in private channels.

    Returns:
        dict: Slack event with channel_type='im' (instant message)
    """
    return {
        "user": "U456USER",
        "channel": "D789DM",
        "channel_type": "im",
        "text": "hello bot",
        "ts": "1234567.890"
    }


@pytest.fixture
def sample_channel_event():
    """Sample channel message event from Slack.

    Used for testing public channel messages with bot mentions.

    Returns:
        dict: Slack event with channel_type='channel' and bot mention
    """
    return {
        "user": "U456USER",
        "channel": "C123CHANNEL",
        "channel_type": "channel",
        "text": "<@U123BOT> hello",
        "ts": "1234567.890"
    }


@pytest.fixture
def sample_thread_event():
    """Sample threaded message event from Slack.

    Used for testing thread detection and thread_ts handling.

    Returns:
        dict: Slack event with thread_ts (reply in a thread)
    """
    return {
        "user": "U456USER",
        "channel": "C123CHANNEL",
        "channel_type": "channel",
        "text": "reply in thread",
        "ts": "1234567.999",
        "thread_ts": "1234567.890"  # Parent message timestamp
    }


@pytest.fixture
def sample_group_event():
    """Sample private group message event from Slack.

    Used for testing private group (multi-person DM) handling.

    Returns:
        dict: Slack event with channel_type='group'
    """
    return {
        "user": "U456USER",
        "channel": "G789GROUP",
        "channel_type": "group",
        "text": "hello team",
        "ts": "1234567.890"
    }


# ============================================================================
# MessageContext Fixtures
# ============================================================================


@pytest.fixture
def message_context_dm(sample_dm_event):
    """MessageContext instance for a DM.

    Convenient fixture for testing transformers and handlers with DM context.

    Returns:
        MessageContext: Context with is_dm=True, is_thread=False
    """
    return MessageContext(sample_dm_event)


@pytest.fixture
def message_context_channel(sample_channel_event):
    """MessageContext instance for a channel message.

    Returns:
        MessageContext: Context with is_dm=False, is_thread=False
    """
    return MessageContext(sample_channel_event)


@pytest.fixture
def message_context_thread(sample_thread_event):
    """MessageContext instance for a threaded message.

    Returns:
        MessageContext: Context with is_dm=False, is_thread=True
    """
    return MessageContext(sample_thread_event)


# ============================================================================
# Bot User ID Fixture
# ============================================================================


@pytest.fixture
def bot_user_id():
    """Standard bot user ID for testing mentions.

    Returns:
        str: Bot's Slack user ID (e.g., 'U123BOT')
    """
    return "U123BOT"


# ============================================================================
# Sample Markdown Content Fixtures
# ============================================================================


@pytest.fixture
def markdown_with_link():
    """Markdown text containing a standard link.

    Returns:
        str: Markdown with [text](url) format
    """
    return "Check out [this documentation](https://example.com/docs) for more info."


@pytest.fixture
def markdown_with_image():
    """Markdown text containing a single image.

    Returns:
        str: Markdown with ![alt](url) format
    """
    return "Here's a chart: ![Sales Chart](https://example.com/chart.png)"


@pytest.fixture
def markdown_with_multiple_images():
    """Markdown text containing multiple images.

    Useful for testing image extraction and max_images limiting.

    Returns:
        str: Markdown with multiple ![alt](url) patterns
    """
    return """
    First image: ![Image 1](https://example.com/img1.png)
    Second image: ![Image 2](https://example.com/img2.jpg)
    Third image: ![Image 3](https://example.com/img3.gif)
    """


@pytest.fixture
def markdown_with_code_block():
    """Markdown text containing a code block.

    Returns:
        str: Markdown with ```language code block
    """
    return """
    Here's some code:
    ```python
    def hello():
        print("world")
    ```
    """


# ============================================================================
# Environment Variable Fixtures for BotConfig
# ============================================================================


@pytest.fixture
def valid_env_vars(monkeypatch):
    """Set up valid environment variables for BotConfig testing.

    This fixture uses monkeypatch to set required env vars for a valid BotConfig.
    The env vars are automatically cleaned up after the test.

    Args:
        monkeypatch: pytest's monkeypatch fixture (auto-injected)

    Yields:
        dict: Environment variables that were set
    """
    env_vars = {
        "SLACK_BOT_TOKEN": "xoxb-test-token-123",
        "SLACK_SIGNING_SECRET": "test-signing-secret-456",
        "ASSISTANT_ID": "test-assistant",
        "LANGGRAPH_URL": "http://localhost:8123"
    }

    # Set all env vars
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    yield env_vars

    # Auto cleanup by monkeypatch fixture
