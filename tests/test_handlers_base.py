"""Unit tests for langgraph2slack.handlers.base module.

Tests BaseHandler functionality including:
- Thread ID generation (deterministic UUID5)
- Thread timestamp determination
- Block creation (images + feedback)
"""

import pytest
from langgraph2slack.handlers.base import BaseHandler
from langgraph2slack.transformers import TransformerChain
from langgraph2slack.config import MessageContext


# ============================================================================
# Fixtures for BaseHandler
# ============================================================================


@pytest.fixture
def base_handler():
    """Create a BaseHandler instance for testing.

    Uses default settings for all parameters.
    """
    return BaseHandler(
        assistant_id="test-assistant",
        input_transformers=TransformerChain(),
        output_transformers=TransformerChain(),
        show_feedback_buttons=True,
        show_thread_id=True,
        extract_images=True,
        max_image_blocks=5,
    )


@pytest.fixture
def handler_no_feedback():
    """BaseHandler with feedback disabled."""
    return BaseHandler(
        assistant_id="test-assistant",
        input_transformers=TransformerChain(),
        output_transformers=TransformerChain(),
        show_feedback_buttons=False,
        show_thread_id=False,
        extract_images=True,
        max_image_blocks=5,
    )


@pytest.fixture
def handler_no_images():
    """BaseHandler with image extraction disabled."""
    return BaseHandler(
        assistant_id="test-assistant",
        input_transformers=TransformerChain(),
        output_transformers=TransformerChain(),
        show_feedback_buttons=True,
        show_thread_id=True,
        extract_images=False,  # Disabled
        max_image_blocks=5,
    )


# ============================================================================
# Tests for _create_thread_id()
# ============================================================================


class TestCreateThreadId:
    """Tests for deterministic thread ID generation using UUID5."""

    # Happy path tests - Determinism
    # ------------------------------------------------------------------------

    def test_same_inputs_produce_same_thread_id(self, base_handler):
        """Same channel and timestamp should always produce same UUID.

        This is CRITICAL for conversation continuity - same Slack thread
        must always map to same LangGraph thread.
        """
        thread_id_1 = base_handler._create_thread_id("C123ABC", "1234567890.123456")
        thread_id_2 = base_handler._create_thread_id("C123ABC", "1234567890.123456")

        assert thread_id_1 == thread_id_2

    def test_multiple_calls_deterministic(self, base_handler):
        """Multiple calls with same inputs should be deterministic."""
        results = [
            base_handler._create_thread_id("C123", "12345.678")
            for _ in range(10)
        ]

        # All results should be identical
        assert len(set(results)) == 1

    def test_thread_id_is_valid_uuid_format(self, base_handler):
        """Generated thread ID should be valid UUID string."""
        thread_id = base_handler._create_thread_id("C123ABC", "1234567890.123456")

        # Should be a string
        assert isinstance(thread_id, str)

        # UUID format: 8-4-4-4-12 characters with hyphens
        # Example: a1b2c3d4-e5f6-5789-a1b2-c3d4e5f67890
        assert len(thread_id) == 36
        assert thread_id.count("-") == 4

        # Should be convertible to UUID object
        import uuid
        uuid_obj = uuid.UUID(thread_id)
        assert str(uuid_obj) == thread_id

    # Happy path tests - Uniqueness
    # ------------------------------------------------------------------------

    def test_different_channels_produce_different_ids(self, base_handler):
        """Different channels should produce different thread IDs."""
        thread_id_1 = base_handler._create_thread_id("C111", "1234567890.123456")
        thread_id_2 = base_handler._create_thread_id("C222", "1234567890.123456")

        assert thread_id_1 != thread_id_2

    def test_different_timestamps_produce_different_ids(self, base_handler):
        """Different timestamps should produce different thread IDs."""
        thread_id_1 = base_handler._create_thread_id("C123ABC", "1111111111.111111")
        thread_id_2 = base_handler._create_thread_id("C123ABC", "2222222222.222222")

        assert thread_id_1 != thread_id_2

    def test_similar_inputs_produce_different_ids(self, base_handler):
        """Similar but different inputs should produce different IDs.

        Guards against substring matching bugs.
        """
        thread_id_1 = base_handler._create_thread_id("C123", "12345.678")
        thread_id_2 = base_handler._create_thread_id("C123", "12345.679")  # +1
        thread_id_3 = base_handler._create_thread_id("C124", "12345.678")  # C124 vs C123

        assert thread_id_1 != thread_id_2
        assert thread_id_1 != thread_id_3
        assert thread_id_2 != thread_id_3

    # Critical negative tests - Edge cases
    # ------------------------------------------------------------------------

    def test_empty_channel_id(self, base_handler):
        """Empty channel ID should still generate valid UUID (edge case)."""
        thread_id = base_handler._create_thread_id("", "12345.678")

        assert isinstance(thread_id, str)
        assert len(thread_id) == 36  # Valid UUID format

        # Should be different from non-empty channel
        thread_id_2 = base_handler._create_thread_id("C123", "12345.678")
        assert thread_id != thread_id_2

    def test_empty_timestamp(self, base_handler):
        """Empty timestamp should still generate valid UUID (edge case)."""
        thread_id = base_handler._create_thread_id("C123", "")

        assert isinstance(thread_id, str)
        assert len(thread_id) == 36

        # Should be different from non-empty timestamp
        thread_id_2 = base_handler._create_thread_id("C123", "12345.678")
        assert thread_id != thread_id_2

    def test_both_empty(self, base_handler):
        """Both empty should still work (extreme edge case)."""
        thread_id = base_handler._create_thread_id("", "")

        assert isinstance(thread_id, str)
        assert len(thread_id) == 36

    def test_special_characters_in_inputs(self, base_handler):
        """Special characters should be handled correctly."""
        thread_id = base_handler._create_thread_id("C123!@#$%", "1234.567:890")

        assert isinstance(thread_id, str)
        assert len(thread_id) == 36

    def test_unicode_characters(self, base_handler):
        """Unicode characters should work in UUID5 generation."""
        thread_id = base_handler._create_thread_id("C123🌱", "12345.678")

        assert isinstance(thread_id, str)
        assert len(thread_id) == 36

    def test_very_long_inputs(self, base_handler):
        """Very long inputs should still generate valid UUID."""
        long_channel = "C" + "x" * 10000
        long_timestamp = "1" * 10000

        thread_id = base_handler._create_thread_id(long_channel, long_timestamp)

        # UUID5 should handle arbitrarily long inputs
        assert isinstance(thread_id, str)
        assert len(thread_id) == 36


# ============================================================================
# Tests for _determine_thread_timestamp()
# ============================================================================


class TestDetermineThreadTimestamp:
    """Tests for determining thread timestamp from MessageContext."""

    def test_uses_thread_ts_when_present(self, base_handler, sample_thread_event):
        """Should use thread_ts when message is in a thread."""
        context = MessageContext(sample_thread_event)
        timestamp = base_handler._determine_thread_timestamp(context)

        assert timestamp == "1234567.890"  # thread_ts, not message_ts

    def test_uses_message_ts_when_no_thread(self, base_handler, sample_dm_event):
        """Should use message_ts when not in a thread."""
        context = MessageContext(sample_dm_event)
        timestamp = base_handler._determine_thread_timestamp(context)

        assert timestamp == "1234567.890"  # message_ts

    def test_channel_message_not_in_thread(self, base_handler, sample_channel_event):
        """Channel message not in thread should use message_ts."""
        context = MessageContext(sample_channel_event)
        timestamp = base_handler._determine_thread_timestamp(context)

        assert timestamp == "1234567.890"  # message_ts


# ============================================================================
# Tests for _create_blocks()
# ============================================================================


class TestCreateBlocks:
    """Tests for creating Slack blocks (images + feedback)."""

    # Happy path tests
    # ------------------------------------------------------------------------

    def test_create_blocks_with_image(self, base_handler):
        """Response with image should create image blocks + feedback."""
        response = "Here's a chart: ![Sales](https://example.com/chart.png)"
        thread_id = "abc-123"

        blocks = base_handler._create_blocks(response, thread_id)

        # Should have: 1 image block + 2 feedback blocks (context + buttons)
        assert len(blocks) == 3

        # First block: image
        assert blocks[0]["type"] == "image"
        assert blocks[0]["image_url"] == "https://example.com/chart.png"

        # Last two blocks: feedback (context with thread_id + buttons)
        assert blocks[1]["type"] == "context"
        assert blocks[2]["type"] == "context_actions"

    def test_create_blocks_with_multiple_images(self, base_handler):
        """Multiple images should all be extracted."""
        response = """
        ![Image 1](https://example.com/1.png)
        ![Image 2](https://example.com/2.jpg)
        ![Image 3](https://example.com/3.gif)
        """
        thread_id = "abc-123"

        blocks = base_handler._create_blocks(response, thread_id)

        # 3 images + 2 feedback blocks
        assert len(blocks) == 5

        # Verify all images
        assert blocks[0]["image_url"] == "https://example.com/1.png"
        assert blocks[1]["image_url"] == "https://example.com/2.jpg"
        assert blocks[2]["image_url"] == "https://example.com/3.gif"

    def test_create_blocks_respects_max_image_blocks(self):
        """max_image_blocks setting should limit number of images."""
        handler = BaseHandler(
            assistant_id="test",
            input_transformers=TransformerChain(),
            output_transformers=TransformerChain(),
            max_image_blocks=2,  # Limit to 2 images
        )

        # Response with 5 images
        response = " ".join([f"![img{i}](url{i})" for i in range(5)])
        thread_id = "abc-123"

        blocks = handler._create_blocks(response, thread_id)

        # Should have: 2 images (limited) + 2 feedback blocks
        assert len(blocks) == 4

        # Only first 2 images
        assert blocks[0]["image_url"] == "url0"
        assert blocks[1]["image_url"] == "url1"

    def test_create_blocks_no_images_only_feedback(self, base_handler):
        """Response without images should only create feedback blocks."""
        response = "Just plain text response"
        thread_id = "abc-123"

        blocks = base_handler._create_blocks(response, thread_id)

        # Only feedback blocks (context + buttons)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "context"
        assert blocks[1]["type"] == "context_actions"

    def test_create_blocks_image_extraction_disabled(self, handler_no_images):
        """When extract_images=False, should not create image blocks."""
        response = "Here's an image: ![Chart](https://example.com/chart.png)"
        thread_id = "abc-123"

        blocks = handler_no_images._create_blocks(response, thread_id)

        # No image blocks, only feedback
        assert len(blocks) == 2
        assert all(block["type"] != "image" for block in blocks)

    def test_create_blocks_feedback_disabled(self, handler_no_feedback):
        """When feedback disabled, should only create image blocks."""
        response = "![Chart](https://example.com/chart.png)"
        thread_id = "abc-123"

        blocks = handler_no_feedback._create_blocks(response, thread_id)

        # Only image block, no feedback
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"

    # Critical negative tests
    # ------------------------------------------------------------------------

    def test_create_blocks_empty_response(self, base_handler):
        """Empty response should still create feedback blocks."""
        response = ""
        thread_id = "abc-123"

        blocks = base_handler._create_blocks(response, thread_id)

        # Only feedback blocks (no images in empty string)
        assert len(blocks) == 2

    def test_create_blocks_with_malformed_image_markdown(self, base_handler):
        """Malformed image markdown should be ignored gracefully."""
        response = "![missing closing paren(https://example.com/img.png"
        thread_id = "abc-123"

        blocks = base_handler._create_blocks(response, thread_id)

        # Malformed image should not match, only feedback blocks
        assert len(blocks) == 2
        assert all(block["type"] != "image" for block in blocks)

    def test_create_blocks_max_images_zero(self):
        """max_image_blocks=0 should extract no images."""
        handler = BaseHandler(
            assistant_id="test",
            input_transformers=TransformerChain(),
            output_transformers=TransformerChain(),
            max_image_blocks=0,  # No images
        )

        response = "![Image](https://example.com/img.png)"
        thread_id = "abc-123"

        blocks = handler._create_blocks(response, thread_id)

        # No images, only feedback
        assert len(blocks) == 2
        assert all(block["type"] != "image" for block in blocks)

    def test_create_blocks_all_features_disabled(self):
        """Both images and feedback disabled should return empty list."""
        handler = BaseHandler(
            assistant_id="test",
            input_transformers=TransformerChain(),
            output_transformers=TransformerChain(),
            show_feedback_buttons=False,
            show_thread_id=False,
            extract_images=False,
        )

        response = "![Image](url) Some text"
        thread_id = "abc-123"

        blocks = handler._create_blocks(response, thread_id)

        # No blocks at all
        assert blocks == []
