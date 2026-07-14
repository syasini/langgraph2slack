"""Unit tests for langgraph2slack.handlers.message module.

Tests MessageHandler functionality including:
- Message content extraction from various LangGraph response formats
- LangGraph invocation with metadata
- Full message processing pipeline
- Error handling
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from langgraph2slack.handlers.message import MessageHandler
from langgraph2slack.transformers import TransformerChain
from langgraph2slack.config import MessageContext


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_langgraph_client():
    """Create a mock LangGraph client for testing.

    This fixture provides a fake LangGraph client that doesn't make real API calls.
    Tests can configure the return values as needed.
    """
    mock = MagicMock()

    # Set up default successful responses
    mock.runs.create = AsyncMock(return_value={
        "run_id": "test-run-123"
    })

    mock.runs.join = AsyncMock(return_value={
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"}
        ]
    })

    return mock


@pytest.fixture
def basic_handler(mock_langgraph_client):
    """Create a basic MessageHandler with default settings.

    Uses empty transformer chains and mock LangGraph client.
    Good for testing core functionality without transformers.
    """
    return MessageHandler(
        langgraph_client=mock_langgraph_client,
        assistant_id="test-assistant",
        input_transformers=TransformerChain(),
        output_transformers=TransformerChain(),
    )


@pytest.fixture
def handler_with_transformers(mock_langgraph_client):
    """Create a MessageHandler with input and output transformers.

    Useful for testing that transformers are applied correctly.
    """
    input_chain = TransformerChain()
    output_chain = TransformerChain()

    # Add a simple transformer to each chain for testing
    @input_chain.add
    async def add_input_prefix(message: str) -> str:
        return f"[INPUT] {message}"

    @output_chain.add
    async def add_output_suffix(message: str) -> str:
        return f"{message} [OUTPUT]"

    return MessageHandler(
        langgraph_client=mock_langgraph_client,
        assistant_id="test-assistant",
        input_transformers=input_chain,
        output_transformers=output_chain,
    )


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
# Tests for _extract_message_content()
# ============================================================================


class TestExtractMessageContent:
    """Tests for extracting message content from LangGraph responses.

    This is a PURE FUNCTION - no mocking needed, just test inputs and outputs!
    """

    # Happy path tests - String content
    # ------------------------------------------------------------------------

    def test_extract_string_content(self, basic_handler):
        """Assistant message with simple string content should be extracted."""
        response = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "It's sunny today!"}
            ]
        }

        result = basic_handler._extract_message_content(response)

        assert result == "It's sunny today!"

    def test_extract_from_multiple_messages(self, basic_handler):
        """Should extract the LAST message (assistant's final response)."""
        response = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing great!"}  # ← Should get this
            ]
        }

        result = basic_handler._extract_message_content(response)

        assert result == "I'm doing great!"

    def test_extract_multiline_content(self, basic_handler):
        """Multiline content should be preserved."""
        response = {
            "messages": [{
                "role": "assistant",
                "content": "Here's a list:\n1. First item\n2. Second item\n3. Third item"
            }]
        }

        result = basic_handler._extract_message_content(response)

        assert "Here's a list:" in result
        assert "1. First item" in result
        assert "\n" in result  # Newlines preserved

    def test_extract_content_with_special_characters(self, basic_handler):
        """Content with special characters should be preserved."""
        special_content = "Hello! <@U123> Check this: https://example.com & more..."
        response = {
            "messages": [{
                "role": "assistant",
                "content": special_content
            }]
        }

        result = basic_handler._extract_message_content(response)

        assert result == special_content

    # Happy path tests - List content (blocks)
    # ------------------------------------------------------------------------

    def test_extract_list_content_with_text_blocks(self, basic_handler):
        """Content as list of blocks should extract and concatenate text blocks."""
        response = {
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Part 1: "},
                    {"type": "text", "text": "Part 2."},
                ]
            }]
        }

        result = basic_handler._extract_message_content(response)

        assert result == "Part 1: Part 2."

    def test_extract_list_content_mixed_blocks(self, basic_handler):
        """List content with mixed block types should only extract text blocks."""
        response = {
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here's an image: "},
                    {"type": "image", "url": "https://example.com/img.png"},  # Skip this
                    {"type": "text", "text": "What do you think?"},
                ]
            }]
        }

        result = basic_handler._extract_message_content(response)

        # Only text blocks should be extracted
        assert result == "Here's an image: What do you think?"
        assert "https://example.com/img.png" not in result

    def test_extract_list_content_no_text_blocks(self, basic_handler):
        """List content with no text blocks should return empty string."""
        response = {
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "image", "url": "https://example.com/img.png"},
                    {"type": "video", "url": "https://example.com/vid.mp4"},
                ]
            }]
        }

        result = basic_handler._extract_message_content(response)

        assert result == ""

    def test_extract_empty_list_content(self, basic_handler):
        """Empty list content should return empty string."""
        response = {
            "messages": [{
                "role": "assistant",
                "content": []
            }]
        }

        result = basic_handler._extract_message_content(response)

        assert result == ""

    # Critical negative tests - Error handling
    # ------------------------------------------------------------------------

    def test_extract_no_messages_returns_error_message(self, basic_handler):
        """Response with no messages should return friendly error message."""
        response = {"messages": []}

        result = basic_handler._extract_message_content(response)

        assert result == "I apologize, but I couldn't generate a response."

    def test_extract_missing_messages_key(self, basic_handler):
        """Response missing 'messages' key should return friendly error."""
        response = {}  # No 'messages' key

        result = basic_handler._extract_message_content(response)

        assert result == "I apologize, but I couldn't generate a response."

    def test_extract_missing_content_key(self, basic_handler):
        """Message missing 'content' key should return empty string."""
        response = {
            "messages": [{
                "role": "assistant"
                # No 'content' key
            }]
        }

        result = basic_handler._extract_message_content(response)

        assert result == ""

    def test_extract_none_content(self, basic_handler):
        """Content with None value should convert to string 'None'.

        The code treats None as unexpected content type and converts to string.
        """
        response = {
            "messages": [{
                "role": "assistant",
                "content": None
            }]
        }

        result = basic_handler._extract_message_content(response)

        # None gets converted to string "None" (unexpected type handling)
        assert result == "None"

    def test_extract_unexpected_content_type(self, basic_handler):
        """Content with unexpected type should convert to string."""
        response = {
            "messages": [{
                "role": "assistant",
                "content": {"unexpected": "dict"}  # Not string or list
            }]
        }

        result = basic_handler._extract_message_content(response)

        # Should convert to string representation
        assert "unexpected" in result or "dict" in result

    def test_extract_malformed_list_block(self, basic_handler):
        """List content with malformed blocks should be handled gracefully."""
        response = {
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Good block"},
                    {"type": "text"},  # Missing 'text' key
                    "not a dict",  # Not even a dict
                    {"type": "text", "text": "Another good block"},
                ]
            }]
        }

        result = basic_handler._extract_message_content(response)

        # Should extract valid blocks and skip invalid ones
        assert "Good block" in result
        assert "Another good block" in result


# ============================================================================
# Tests for _invoke_langgraph()
# ============================================================================


class TestInvokeLangGraph:
    """Tests for invoking LangGraph with mocked client.

    These tests use AsyncMock to simulate LangGraph API calls without
    actually connecting to the service.
    """

    # Happy path tests
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_invoke_successful(self, basic_handler, mock_langgraph_client, sample_context):
        """Successful LangGraph invocation should return response and run_id."""
        # The mock is already configured in the fixture

        result, run_id = await basic_handler._invoke_langgraph(
            message="Hello",
            thread_id="thread-abc-123",
            context=sample_context
        )

        # Verify run_id was extracted
        assert run_id == "test-run-123"

        # Verify response contains messages
        assert "messages" in result
        assert len(result["messages"]) == 2

        # Verify the client was called with correct parameters
        mock_langgraph_client.runs.create.assert_called_once_with(
            thread_id="thread-abc-123",
            assistant_id="test-assistant",
            input={"messages": [{"role": "user", "content": "Hello"}]},
            if_not_exists="create",
            multitask_strategy="interrupt",
            metadata={},
            config=None,
        )

        # Verify join was called with run_id
        mock_langgraph_client.runs.join.assert_called_once_with(
            "thread-abc-123",
            "test-run-123"
        )

    @pytest.mark.asyncio
    async def test_invoke_with_metadata_builder(self, mock_langgraph_client, sample_context):
        """Handler with metadata_builder should pass metadata to LangGraph."""
        # Create a custom metadata builder
        async def custom_metadata_builder(context: MessageContext):
            return {
                "user_id": context.user_id,
                "channel_id": context.channel_id,
                "custom_field": "test_value"
            }

        handler = MessageHandler(
            langgraph_client=mock_langgraph_client,
            assistant_id="test-assistant",
            input_transformers=TransformerChain(),
            output_transformers=TransformerChain(),
            metadata_builder=custom_metadata_builder,
        )

        await handler._invoke_langgraph(
            message="Test",
            thread_id="thread-123",
            context=sample_context
        )

        # Verify metadata was passed correctly
        mock_langgraph_client.runs.create.assert_called_once()
        call_kwargs = mock_langgraph_client.runs.create.call_args.kwargs

        assert call_kwargs["metadata"]["user_id"] == "U123USER"
        assert call_kwargs["metadata"]["channel_id"] == "C456CHANNEL"
        assert call_kwargs["metadata"]["custom_field"] == "test_value"

    @pytest.mark.asyncio
    async def test_invoke_passes_multitask_strategy(self, mock_langgraph_client, sample_context):
        """Non-streaming handler should forward multitask_strategy to runs.create."""
        handler = MessageHandler(
            langgraph_client=mock_langgraph_client,
            assistant_id="test-assistant",
            input_transformers=TransformerChain(),
            output_transformers=TransformerChain(),
            multitask_strategy="enqueue",
        )

        await handler._invoke_langgraph(
            message="Test",
            thread_id="thread-123",
            context=sample_context,
        )

        call_kwargs = mock_langgraph_client.runs.create.call_args.kwargs
        assert call_kwargs["multitask_strategy"] == "enqueue"

    @pytest.mark.asyncio
    async def test_invoke_default_multitask_strategy_is_interrupt(
        self, basic_handler, mock_langgraph_client, sample_context
    ):
        """Non-streaming handler defaults to interrupt strategy."""
        await basic_handler._invoke_langgraph(
            message="Test",
            thread_id="thread-123",
            context=sample_context,
        )

        call_kwargs = mock_langgraph_client.runs.create.call_args.kwargs
        assert call_kwargs["multitask_strategy"] == "interrupt"

    @pytest.mark.asyncio
    async def test_invoke_without_metadata_builder(self, basic_handler, mock_langgraph_client, sample_context):
        """Handler without metadata_builder should pass empty metadata dict."""
        await basic_handler._invoke_langgraph(
            message="Test",
            thread_id="thread-123",
            context=sample_context
        )

        # Verify empty metadata was passed
        call_kwargs = mock_langgraph_client.runs.create.call_args.kwargs
        assert call_kwargs["metadata"] == {}

    @pytest.mark.asyncio
    async def test_invoke_with_different_thread_ids(self, basic_handler, mock_langgraph_client, sample_context):
        """Different thread_ids should be passed correctly to LangGraph."""
        thread_ids = ["thread-1", "thread-2", "thread-3"]

        for thread_id in thread_ids:
            # Reset the mock to clear previous calls
            mock_langgraph_client.runs.create.reset_mock()

            await basic_handler._invoke_langgraph(
                message="Test",
                thread_id=thread_id,
                context=sample_context
            )

            # Verify correct thread_id was used
            call_kwargs = mock_langgraph_client.runs.create.call_args.kwargs
            assert call_kwargs["thread_id"] == thread_id

    # Error handling tests
    # ------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_invoke_create_fails(self, basic_handler, mock_langgraph_client, sample_context):
        """LangGraph create() failure should propagate exception."""
        # Make create() raise an error
        mock_langgraph_client.runs.create = AsyncMock(
            side_effect=Exception("LangGraph API Error: Rate limit exceeded")
        )

        # Should raise the exception
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await basic_handler._invoke_langgraph(
                message="Test",
                thread_id="thread-123",
                context=sample_context
            )

    @pytest.mark.asyncio
    async def test_invoke_join_fails(self, basic_handler, mock_langgraph_client, sample_context):
        """LangGraph join() failure should propagate exception."""
        # create() succeeds but join() fails
        mock_langgraph_client.runs.join = AsyncMock(
            side_effect=Exception("LangGraph API Error: Run timeout")
        )

        with pytest.raises(Exception, match="Run timeout"):
            await basic_handler._invoke_langgraph(
                message="Test",
                thread_id="thread-123",
                context=sample_context
            )

    @pytest.mark.asyncio
    async def test_invoke_network_error(self, basic_handler, mock_langgraph_client, sample_context):
        """Network errors should propagate."""
        mock_langgraph_client.runs.create = AsyncMock(
            side_effect=ConnectionError("Network unreachable")
        )

        with pytest.raises(ConnectionError, match="Network unreachable"):
            await basic_handler._invoke_langgraph(
                message="Test",
                thread_id="thread-123",
                context=sample_context
            )


# ============================================================================
# Tests for process_message() - Full Pipeline
# ============================================================================


class TestProcessMessage:
    """Tests for the complete message processing pipeline.

    This tests the full flow:
    1. Apply input transformers
    2. Create thread ID
    3. Invoke LangGraph
    4. Extract content
    5. Apply output transformers
    6. Format for Slack
    7. Create blocks
    """

    @pytest.mark.asyncio
    async def test_process_message_basic(self, basic_handler, sample_context):
        """Basic message processing should work end-to-end."""
        result = await basic_handler.process_message(
            message="What's the weather?",
            context=sample_context
        )

        formatted_text, blocks, thread_id, run_id, has_custom_blocks = result

        # Verify response text
        assert formatted_text == "Hi there! How can I help?"

        # Verify run_id
        assert run_id == "test-run-123"

        # Verify thread_id exists (deterministic UUID)
        assert thread_id is not None
        assert len(thread_id) == 36  # UUID format

        # Verify blocks were created (should have feedback blocks by default)
        assert isinstance(blocks, list)

    @pytest.mark.asyncio
    async def test_process_message_with_transformers(self, handler_with_transformers, sample_context):
        """Message processing with transformers should apply them correctly."""
        result = await handler_with_transformers.process_message(
            message="Hello",
            context=sample_context
        )

        formatted_text, blocks, thread_id, run_id, has_custom_blocks = result

        # Output transformer adds " [OUTPUT]" suffix
        assert formatted_text.endswith(" [OUTPUT]")
        assert "Hi there! How can I help?" in formatted_text

    @pytest.mark.asyncio
    async def test_process_message_thread_id_deterministic(self, basic_handler):
        """Same channel/thread should produce same thread_id."""
        context1 = MessageContext({
            "user": "U123",
            "channel": "C456",
            "channel_type": "channel",
            "ts": "1234567.890"
        })

        context2 = MessageContext({
            "user": "U999",  # Different user
            "channel": "C456",  # Same channel
            "channel_type": "channel",
            "ts": "1234567.890"  # Same timestamp
        })

        _, _, thread_id_1, _, _ = await basic_handler.process_message("Test", context1)
        _, _, thread_id_2, _, _ = await basic_handler.process_message("Test", context2)

        # Same channel + timestamp = same thread
        assert thread_id_1 == thread_id_2

    @pytest.mark.asyncio
    async def test_process_message_in_thread_uses_thread_ts(self, basic_handler):
        """Message in a thread should use thread_ts for thread_id generation."""
        # Message in a thread
        thread_context = MessageContext({
            "user": "U123",
            "channel": "C456",
            "channel_type": "channel",
            "ts": "9999999.999",  # Current message timestamp
            "thread_ts": "1234567.890"  # Parent thread timestamp
        })

        # Standalone message
        standalone_context = MessageContext({
            "user": "U123",
            "channel": "C456",
            "channel_type": "channel",
            "ts": "1234567.890"  # Same as thread_ts above
        })

        _, _, thread_id_1, _, _ = await basic_handler.process_message("Test", thread_context)
        _, _, thread_id_2, _, _ = await basic_handler.process_message("Test", standalone_context)

        # Should use thread_ts, so thread_ids should match
        assert thread_id_1 == thread_id_2

    @pytest.mark.asyncio
    async def test_process_message_with_markdown_links(self, basic_handler, mock_langgraph_client, sample_context):
        """Message with markdown links should be converted to Slack format."""
        # Configure mock to return markdown content
        mock_langgraph_client.runs.join = AsyncMock(return_value={
            "messages": [{
                "role": "assistant",
                "content": "Check out [this link](https://example.com)"
            }]
        })

        formatted_text, _, _, _, _ = await basic_handler.process_message("Test", sample_context)

        # Markdown should be converted to Slack format
        assert "<https://example.com|this link>" in formatted_text

    @pytest.mark.asyncio
    async def test_process_message_with_images(self, basic_handler, mock_langgraph_client, sample_context):
        """Message with markdown images should create image blocks."""
        # Configure mock to return content with images
        mock_langgraph_client.runs.join = AsyncMock(return_value={
            "messages": [{
                "role": "assistant",
                "content": "Here's a chart: ![Sales](https://example.com/chart.png)"
            }]
        })

        _, blocks, _, _, _ = await basic_handler.process_message("Test", sample_context)

        # Should have image blocks
        image_blocks = [b for b in blocks if b.get("type") == "image"]
        assert len(image_blocks) > 0
        assert image_blocks[0]["image_url"] == "https://example.com/chart.png"

    @pytest.mark.asyncio
    async def test_process_message_dm_context(self, basic_handler):
        """Processing DM should work correctly."""
        dm_context = MessageContext({
            "user": "U123",
            "channel": "D789DM",
            "channel_type": "im",
            "ts": "1234567.890"
        })

        formatted_text, _, thread_id, _, _ = await basic_handler.process_message("Hello", dm_context)

        assert formatted_text is not None
        assert thread_id is not None


# ============================================================================
# Edge Cases and Integration
# ============================================================================


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_empty_message(self, basic_handler, sample_context):
        """Empty message should be processed (LangGraph may still respond)."""
        result = await basic_handler.process_message("", sample_context)

        formatted_text, blocks, thread_id, run_id, has_custom_blocks = result

        # Should still work (LangGraph decides how to handle empty input)
        assert formatted_text is not None
        assert thread_id is not None

    @pytest.mark.asyncio
    async def test_very_long_message(self, basic_handler, sample_context):
        """Very long messages should be processed."""
        long_message = "Hello! " * 1000  # Very long message

        result = await basic_handler.process_message(long_message, sample_context)

        formatted_text, _, _, _, _ = result
        assert formatted_text is not None

    @pytest.mark.asyncio
    async def test_message_with_special_characters(self, basic_handler, sample_context):
        """Messages with special characters should be handled."""
        special_message = "Test <@U123> & \"quotes\" 'apostrophes' ñ 你好 🌱"

        result = await basic_handler.process_message(special_message, sample_context)

        formatted_text, _, _, _, _ = result
        assert formatted_text is not None


# ============================================================================
# Block-Returning Transformers
# ============================================================================


class TestCustomBlockTransformers:
    """Tests for output transformers that return Slack block lists."""

    @pytest.mark.asyncio
    async def test_transformer_returning_blocks_sets_has_custom_blocks(
        self, mock_langgraph_client, sample_context
    ):
        """When a transformer returns list[dict], has_custom_blocks should be True."""
        output_chain = TransformerChain()

        @output_chain.add
        async def render_as_blocks(text: str):
            return [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]

        handler = MessageHandler(
            langgraph_client=mock_langgraph_client,
            assistant_id="test",
            input_transformers=TransformerChain(),
            output_transformers=output_chain,
        )

        _, blocks, _, _, has_custom_blocks = await handler.process_message("Hi", sample_context)

        assert has_custom_blocks is True
        # Custom blocks should be in the block list (plus any feedback blocks)
        section_blocks = [b for b in blocks if b.get("type") == "section"]
        assert len(section_blocks) >= 1

    @pytest.mark.asyncio
    async def test_transformer_returning_string_sets_has_custom_blocks_false(
        self, basic_handler, sample_context
    ):
        """When no transformer or a string-returning transformer is used, has_custom_blocks is False."""
        _, _, _, _, has_custom_blocks = await basic_handler.process_message("Hi", sample_context)
        assert has_custom_blocks is False

    @pytest.mark.asyncio
    async def test_chain_short_circuits_on_blocks(self, mock_langgraph_client, sample_context):
        """When first transformer returns blocks, second transformer should not be called."""
        call_log = []
        output_chain = TransformerChain()

        @output_chain.add
        async def first_returns_blocks(text: str):
            call_log.append("first")
            return [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]

        @output_chain.add
        async def second_should_not_run(text):
            call_log.append("second")
            return text

        handler = MessageHandler(
            langgraph_client=mock_langgraph_client,
            assistant_id="test",
            input_transformers=TransformerChain(),
            output_transformers=output_chain,
        )

        await handler.process_message("Hi", sample_context)

        assert "first" in call_log
        assert "second" not in call_log  # chain short-circuited

    @pytest.mark.asyncio
    async def test_no_transformer_returns_safe_text(self, basic_handler, sample_context):
        """With no output transformers, response should be a string with has_custom_blocks=False."""
        formatted_text, blocks, _, _, has_custom_blocks = await basic_handler.process_message(
            "Hi", sample_context
        )

        assert isinstance(formatted_text, str)
        assert has_custom_blocks is False
        assert isinstance(blocks, list)
