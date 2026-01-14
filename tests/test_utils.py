"""Unit tests for langgraph2slack.utils module.

Tests utility functions for Slack integration including:
- Markdown conversion (clean_markdown)
- Image extraction (extract_markdown_images)
- Event detection (is_bot_mention, is_dm)
- Block creation (create_feedback_block, create_feedback_modal, extract_feedback_text)
"""

import pytest
from langgraph2slack.utils import (
    clean_markdown,
    extract_markdown_images,
    is_bot_mention,
    is_dm,
    create_feedback_block,
    create_feedback_modal,
    extract_feedback_text,
)


# ============================================================================
# Tests for clean_markdown()
# ============================================================================


class TestCleanMarkdown:
    """Tests for converting standard markdown to Slack mrkdwn format."""

    # Happy path tests
    # ------------------------------------------------------------------------

    def test_convert_simple_link(self):
        """Standard markdown link should convert to Slack format."""
        input_md = "Check [this link](https://example.com)"
        result = clean_markdown(input_md)
        assert result == "Check <https://example.com|this link>"

    def test_convert_multiple_links(self):
        """Multiple links should all be converted."""
        input_md = "[First](https://example.com) and [Second](https://test.com)"
        result = clean_markdown(input_md)
        assert result == "<https://example.com|First> and <https://test.com|Second>"

    def test_convert_simple_image(self):
        """Standard markdown image should convert to Slack format."""
        input_md = "![Plant Photo](https://example.com/plant.png)"
        result = clean_markdown(input_md)
        assert result == "!<https://example.com/plant.png|Plant Photo>"

    def test_convert_multiple_images(self):
        """Multiple images should all be converted."""
        input_md = "![First](url1.png) and ![Second](url2.jpg)"
        result = clean_markdown(input_md)
        assert result == "!<url1.png|First> and !<url2.jpg|Second>"

    def test_convert_code_block_with_language(self):
        """Code blocks with language identifiers should have them removed."""
        input_md = "```python\nprint('hello')\n```"
        result = clean_markdown(input_md)
        # Slack doesn't use language identifiers
        assert result == "```\nprint('hello')\n```"

    @pytest.mark.parametrize("language", ["python", "javascript", "bash", "json"])
    def test_convert_code_block_various_languages(self, language):
        """Code blocks with various languages should all be cleaned."""
        input_md = f"```{language}\ncode here\n```"
        result = clean_markdown(input_md)
        assert result == "```\ncode here\n```"

    def test_convert_mixed_content(self):
        """Markdown with links, images, and code should all convert correctly."""
        input_md = """
Check [docs](https://example.com/docs).

![Chart](https://example.com/chart.png)

```python
print("test")
```
"""
        result = clean_markdown(input_md)

        # All conversions should happen
        assert "<https://example.com/docs|docs>" in result
        assert "!<https://example.com/chart.png|Chart>" in result
        assert "```\n" in result
        assert "```python" not in result

    # Critical negative tests
    # ------------------------------------------------------------------------

    def test_url_with_parentheses(self):
        """URLs containing parentheses should be handled correctly.

        This is a CRITICAL test - many Wikipedia URLs contain parentheses.
        Example: https://en.wikipedia.org/wiki/File:Monstera_(plant).jpg
        """
        input_md = "![Monstera](https://en.wikipedia.org/wiki/File:Monstera_(plant).jpg)"
        result = clean_markdown(input_md)
        # Should capture full URL including parentheses
        assert "!<https://en.wikipedia.org/wiki/File:Monstera_(plant).jpg|Monstera>" in result

    def test_url_with_multiple_parentheses(self):
        """URLs with multiple parentheses should work correctly."""
        input_md = "![Test](https://example.com/file(1)(2).png)"
        result = clean_markdown(input_md)
        assert "!<https://example.com/file(1)(2).png|Test>" in result

    def test_link_with_parentheses_in_url(self):
        """Regular links (not images) with parentheses should also work."""
        input_md = "[Article](https://en.wikipedia.org/wiki/Python_(programming_language))"
        result = clean_markdown(input_md)
        assert "<https://en.wikipedia.org/wiki/Python_(programming_language)|Article>" in result

    def test_empty_string(self):
        """Empty string should return empty string (graceful handling)."""
        result = clean_markdown("")
        assert result == ""

    def test_no_markdown(self):
        """Plain text without markdown should pass through unchanged."""
        input_text = "Just plain text with no markdown"
        result = clean_markdown(input_text)
        assert result == input_text

    def test_empty_link_text(self):
        """Link with empty text should still convert (edge case)."""
        input_md = "[](https://example.com)"
        result = clean_markdown(input_md)
        assert result == "<https://example.com|>"

    def test_empty_image_alt_text(self):
        """Image with empty alt text should still convert."""
        input_md = "![](https://example.com/image.png)"
        result = clean_markdown(input_md)
        assert result == "!<https://example.com/image.png|>"

    def test_consecutive_images_no_spaces(self):
        """Multiple images next to each other should all convert."""
        input_md = "![img1](url1)![img2](url2)![img3](url3)"
        result = clean_markdown(input_md)
        assert result == "!<url1|img1>!<url2|img2>!<url3|img3>"

    # Tests for for_blocks parameter (bold/italic/bullet conversion)
    # ------------------------------------------------------------------------

    def test_bold_not_converted_for_streaming(self):
        """Bold should NOT be converted when for_blocks=False (streaming)."""
        input_md = "This is **bold text** here"
        result = clean_markdown(input_md, for_blocks=False)
        # Should keep standard markdown format
        assert result == "This is **bold text** here"

    def test_bold_converted_for_blocks(self):
        """Bold should be converted to Slack format when for_blocks=True."""
        input_md = "This is **bold text** here"
        result = clean_markdown(input_md, for_blocks=True)
        # Should convert to Slack mrkdwn (single asterisk)
        assert result == "This is *bold text* here"

    def test_italic_not_converted_for_streaming(self):
        """Italic should NOT be converted when for_blocks=False (streaming)."""
        input_md = "This is *italic text* here"
        result = clean_markdown(input_md, for_blocks=False)
        # Should keep standard markdown format
        assert result == "This is *italic text* here"

    def test_italic_converted_for_blocks(self):
        """Italic should be converted to Slack format when for_blocks=True."""
        input_md = "This is *italic text* here"
        result = clean_markdown(input_md, for_blocks=True)
        # Should convert to Slack mrkdwn (underscore)
        assert result == "This is _italic text_ here"

    def test_mixed_bold_and_italic_for_blocks(self):
        """Mixed bold and italic should both convert correctly."""
        input_md = "This is **bold** and this is *italic* text"
        result = clean_markdown(input_md, for_blocks=True)
        # Bold -> single asterisk, italic -> underscore
        assert result == "This is *bold* and this is _italic_ text"

    def test_bullets_not_converted_for_streaming(self):
        """Bullets should NOT be converted when for_blocks=False (streaming)."""
        input_md = "- First item\n- Second item\n- Third item"
        result = clean_markdown(input_md, for_blocks=False)
        # Should keep standard markdown format
        assert result == "- First item\n- Second item\n- Third item"

    def test_bullets_converted_for_blocks(self):
        """Bullets should be converted to • when for_blocks=True."""
        input_md = "- First item\n- Second item\n- Third item"
        result = clean_markdown(input_md, for_blocks=True)
        # Should convert to bullet points
        assert result == "• First item\n• Second item\n• Third item"

    def test_asterisk_bullets_converted_for_blocks(self):
        """Asterisk bullets should also convert to • for blocks."""
        input_md = "* First item\n* Second item"
        result = clean_markdown(input_md, for_blocks=True)
        assert result == "• First item\n• Second item"

    def test_indented_bullets_converted_for_blocks(self):
        """Indented bullets should preserve indentation."""
        input_md = "  - Indented item\n    - More indented"
        result = clean_markdown(input_md, for_blocks=True)
        assert result == "  • Indented item\n    • More indented"

    def test_complex_markdown_for_blocks(self):
        """Complex markdown with bold, italic, bullets, and links for blocks."""
        input_md = """**Bold text** and *italic text*
- First bullet
- Second bullet with **bold**
[Link](https://example.com)"""
        result = clean_markdown(input_md, for_blocks=True)

        # Check all conversions
        assert "*Bold text*" in result  # Bold converted
        assert "_italic text_" in result  # Italic converted
        assert "• First bullet" in result  # Bullet converted
        assert "• Second bullet with *bold*" in result  # Bullet + bold
        assert "<https://example.com|Link>" in result  # Link converted

    def test_default_parameter_is_streaming(self):
        """Default behavior (no parameter) should be streaming mode."""
        input_md = "**bold** and *italic* and - bullet"
        result = clean_markdown(input_md)  # No parameter
        # Should NOT convert bold/italic/bullets (same as for_blocks=False)
        assert result == "**bold** and *italic* and - bullet"


# ============================================================================
# Tests for extract_markdown_images()
# ============================================================================


class TestExtractMarkdownImages:
    """Tests for extracting markdown images and creating Slack image blocks."""

    # Happy path tests
    # ------------------------------------------------------------------------

    def test_extract_single_image(self):
        """Single image should be extracted correctly."""
        text = "Here's a chart: ![Sales Chart](https://example.com/chart.png)"
        blocks = extract_markdown_images(text)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["image_url"] == "https://example.com/chart.png"
        assert blocks[0]["alt_text"] == "Sales Chart"

    def test_extract_multiple_images(self):
        """Multiple images should all be extracted."""
        text = """
        ![First](https://example.com/1.png)
        ![Second](https://example.com/2.jpg)
        ![Third](https://example.com/3.gif)
        """
        blocks = extract_markdown_images(text)

        assert len(blocks) == 3
        assert blocks[0]["image_url"] == "https://example.com/1.png"
        assert blocks[1]["image_url"] == "https://example.com/2.jpg"
        assert blocks[2]["image_url"] == "https://example.com/3.gif"

    def test_extract_with_max_images_limit(self):
        """max_images parameter should limit number of extracted images."""
        text = "![1](url1) ![2](url2) ![3](url3) ![4](url4) ![5](url5)"
        blocks = extract_markdown_images(text, max_images=3)

        assert len(blocks) == 3
        # Should get first 3 images
        assert blocks[0]["image_url"] == "url1"
        assert blocks[1]["image_url"] == "url2"
        assert blocks[2]["image_url"] == "url3"

    def test_extract_empty_alt_text_uses_default(self):
        """Empty alt text should default to 'Image'."""
        text = "![](https://example.com/photo.png)"
        blocks = extract_markdown_images(text)

        assert len(blocks) == 1
        assert blocks[0]["alt_text"] == "Image"  # Default value

    @pytest.mark.parametrize("alt_text,url", [
        ("Photo", "https://example.com/photo.png"),
        ("Chart", "https://example.com/chart.jpg"),
        ("Diagram", "https://example.com/diagram.svg"),
    ])
    def test_extract_various_image_formats(self, alt_text, url):
        """Various image formats should all be extracted."""
        text = f"![{alt_text}]({url})"
        blocks = extract_markdown_images(text)

        assert len(blocks) == 1
        assert blocks[0]["image_url"] == url
        assert blocks[0]["alt_text"] == alt_text

    # Critical negative tests
    # ------------------------------------------------------------------------

    def test_extract_wikipedia_url_with_parentheses(self):
        """Wikipedia URLs with parentheses should be extracted correctly.

        This is a CRITICAL real-world test case. Many Wikipedia image URLs
        contain parentheses and must be handled correctly.
        """
        text = "![Monstera](https://en.wikipedia.org/wiki/File:Monstera_deliciosa_(plant).jpg)"
        blocks = extract_markdown_images(text)

        assert len(blocks) == 1
        assert blocks[0]["image_url"] == "https://en.wikipedia.org/wiki/File:Monstera_deliciosa_(plant).jpg"
        assert blocks[0]["alt_text"] == "Monstera"

    def test_extract_no_images(self):
        """Text without images should return empty list (graceful handling)."""
        text = "Just regular text with no images"
        blocks = extract_markdown_images(text)

        assert blocks == []

    def test_extract_empty_string(self):
        """Empty string should return empty list (graceful handling)."""
        blocks = extract_markdown_images("")
        assert blocks == []

    def test_extract_max_images_zero(self):
        """max_images=0 should return empty list (boundary condition)."""
        text = "![img1](url1) ![img2](url2)"
        blocks = extract_markdown_images(text, max_images=0)

        assert blocks == []

    def test_extract_max_images_equals_count(self):
        """max_images exactly equal to image count should return all (boundary)."""
        text = "![1](url1) ![2](url2) ![3](url3)"
        blocks = extract_markdown_images(text, max_images=3)

        assert len(blocks) == 3  # All images returned, no truncation

    def test_extract_max_images_greater_than_count(self):
        """max_images greater than image count should return all images."""
        text = "![1](url1) ![2](url2)"
        blocks = extract_markdown_images(text, max_images=10)

        assert len(blocks) == 2  # Only 2 images exist

    def test_extract_with_special_chars_in_alt_text(self):
        """Special characters in alt text should be preserved."""
        text = "![<Plant & Flower>](https://example.com/img.png)"
        blocks = extract_markdown_images(text)

        assert len(blocks) == 1
        # Alt text should be preserved as-is (Slack handles escaping)
        assert blocks[0]["alt_text"] == "<Plant & Flower>"

    def test_extract_url_with_query_params(self):
        """URLs with query parameters should be extracted fully."""
        text = "![Chart](https://example.com/img.png?size=large&format=png)"
        blocks = extract_markdown_images(text)

        assert len(blocks) == 1
        assert blocks[0]["image_url"] == "https://example.com/img.png?size=large&format=png"

    def test_extract_url_with_fragment(self):
        """URLs with fragments should be extracted fully."""
        text = "![Section](https://example.com/doc.html#section1)"
        blocks = extract_markdown_images(text)

        assert len(blocks) == 1
        assert blocks[0]["image_url"] == "https://example.com/doc.html#section1"


# ============================================================================
# Tests for is_bot_mention()
# ============================================================================


class TestIsBotMention:
    """Tests for detecting bot mentions in Slack messages."""

    # Happy path tests
    # ------------------------------------------------------------------------

    def test_bot_mentioned(self, bot_user_id):
        """Text with bot mention should return True."""
        text = f"<@{bot_user_id}> hello there!"
        assert is_bot_mention(text, bot_user_id) is True

    def test_bot_mentioned_middle_of_text(self, bot_user_id):
        """Bot mention in middle of text should be detected."""
        text = f"Hey <@{bot_user_id}> can you help?"
        assert is_bot_mention(text, bot_user_id) is True

    def test_bot_mentioned_at_end(self, bot_user_id):
        """Bot mention at end of text should be detected."""
        text = f"Thanks <@{bot_user_id}>"
        assert is_bot_mention(text, bot_user_id) is True

    def test_bot_not_mentioned(self, bot_user_id):
        """Text without bot mention should return False."""
        text = "Just a regular message"
        assert is_bot_mention(text, bot_user_id) is False

    def test_different_user_mentioned(self, bot_user_id):
        """Mention of different user should return False."""
        text = "<@U999DIFFERENT> hello"
        assert is_bot_mention(text, bot_user_id) is False

    # Critical negative tests
    # ------------------------------------------------------------------------

    def test_empty_text(self, bot_user_id):
        """Empty text should return False (graceful handling)."""
        assert is_bot_mention("", bot_user_id) is False

    def test_partial_mention_no_closing(self, bot_user_id):
        """Malformed mention without closing bracket should return False."""
        text = f"<@{bot_user_id} hello"  # Missing '>'
        assert is_bot_mention(text, bot_user_id) is False

    def test_similar_user_id(self, bot_user_id):
        """Similar but different user ID should return False.

        Guards against substring matching bugs.
        """
        # If bot_user_id is U123BOT, then U123BOTX should NOT match
        text = f"<@{bot_user_id}X>"
        assert is_bot_mention(text, bot_user_id) is False

    def test_multiple_mentions_includes_bot(self, bot_user_id):
        """Multiple mentions including bot should return True."""
        text = f"<@U111> and <@{bot_user_id}> and <@U222>"
        assert is_bot_mention(text, bot_user_id) is True


# ============================================================================
# Tests for is_dm()
# ============================================================================


class TestIsDM:
    """Tests for detecting direct messages (DMs)."""

    # Happy path tests
    # ------------------------------------------------------------------------

    def test_is_dm_true(self, sample_dm_event):
        """Event with channel_type='im' should return True."""
        assert is_dm(sample_dm_event) is True

    def test_is_dm_false_channel(self, sample_channel_event):
        """Event with channel_type='channel' should return False."""
        assert is_dm(sample_channel_event) is False

    def test_is_dm_false_group(self, sample_group_event):
        """Event with channel_type='group' should return False."""
        assert is_dm(sample_group_event) is False

    @pytest.mark.parametrize("channel_type,expected", [
        ("im", True),           # DM
        ("channel", False),     # Public channel
        ("group", False),       # Private group
        ("mpim", False),        # Multi-person IM
    ])
    def test_is_dm_various_channel_types(self, channel_type, expected):
        """Various channel types should be correctly identified."""
        event = {"channel_type": channel_type}
        assert is_dm(event) is expected

    # Critical negative tests
    # ------------------------------------------------------------------------

    def test_is_dm_missing_channel_type(self):
        """Event without channel_type should return False (graceful handling)."""
        event = {"user": "U123", "channel": "C456"}  # No channel_type
        assert is_dm(event) is False

    def test_is_dm_empty_event(self):
        """Empty event dict should return False (graceful handling)."""
        event = {}
        assert is_dm(event) is False

    def test_is_dm_channel_type_none(self):
        """Event with channel_type=None should return False."""
        event = {"channel_type": None}
        assert is_dm(event) is False


# ============================================================================
# Tests for create_feedback_block()
# ============================================================================


class TestCreateFeedbackBlock:
    """Tests for creating Slack feedback button blocks."""

    # Happy path tests
    # ------------------------------------------------------------------------

    def test_create_with_both_enabled(self):
        """Both feedback buttons and thread_id enabled should create 2 blocks."""
        blocks = create_feedback_block(
            thread_id="abc-123",
            show_feedback_buttons=True,
            show_thread_id=True
        )

        assert len(blocks) == 2
        # First block: context with thread_id
        assert blocks[0]["type"] == "context"
        assert "abc-123" in blocks[0]["elements"][0]["text"]
        # Second block: feedback buttons
        assert blocks[1]["type"] == "context_actions"

    def test_create_with_only_buttons(self):
        """Only feedback buttons enabled should create 1 block."""
        blocks = create_feedback_block(
            thread_id="abc-123",
            show_feedback_buttons=True,
            show_thread_id=False  # Disabled
        )

        assert len(blocks) == 1
        assert blocks[0]["type"] == "context_actions"

    def test_create_with_only_thread_id(self):
        """Only thread_id enabled should create 1 block."""
        blocks = create_feedback_block(
            thread_id="abc-123",
            show_feedback_buttons=False,  # Disabled
            show_thread_id=True
        )

        assert len(blocks) == 1
        assert blocks[0]["type"] == "context"

    def test_create_with_both_disabled(self):
        """Both disabled should return empty list."""
        blocks = create_feedback_block(
            thread_id="abc-123",
            show_feedback_buttons=False,
            show_thread_id=False
        )

        assert blocks == []

    def test_feedback_button_structure(self):
        """Feedback buttons should have correct structure."""
        blocks = create_feedback_block(
            show_feedback_buttons=True,
            show_thread_id=False
        )

        assert len(blocks) == 1
        button_block = blocks[0]

        # Verify structure
        assert button_block["type"] == "context_actions"
        assert "elements" in button_block
        assert len(button_block["elements"]) == 1

        feedback_element = button_block["elements"][0]
        assert feedback_element["type"] == "feedback_buttons"
        assert feedback_element["action_id"] == "feedback"
        assert "positive_button" in feedback_element
        assert "negative_button" in feedback_element

    # Critical negative tests
    # ------------------------------------------------------------------------

    def test_create_without_thread_id_but_show_enabled(self):
        """show_thread_id=True but no thread_id provided should not add context block."""
        blocks = create_feedback_block(
            thread_id=None,  # No thread_id
            show_feedback_buttons=False,
            show_thread_id=True
        )

        # Should not add context block without thread_id
        assert blocks == []

    def test_create_with_empty_thread_id(self):
        """Empty string thread_id should not add context block."""
        blocks = create_feedback_block(
            thread_id="",  # Empty string
            show_feedback_buttons=False,
            show_thread_id=True
        )

        assert blocks == []


# ============================================================================
# Tests for create_feedback_modal()
# ============================================================================


class TestCreateFeedbackModal:
    """Tests for creating feedback modal view."""

    def test_modal_structure(self):
        """Modal should have correct structure for Slack views.open API."""
        message_context = '{"channel_id": "C123", "message_ts": "1234.567", "run_id": "run-123"}'
        modal = create_feedback_modal(message_context)

        # Verify top-level structure
        assert modal["type"] == "modal"
        assert modal["callback_id"] == "feedback_modal"
        assert modal["private_metadata"] == message_context

        # Verify title, submit, close buttons
        assert modal["title"]["type"] == "plain_text"
        assert modal["title"]["text"] == "Feedback"
        assert modal["submit"]["type"] == "plain_text"
        assert modal["close"]["type"] == "plain_text"

    def test_modal_input_block(self):
        """Modal should contain input block for feedback text."""
        modal = create_feedback_modal("{}")

        assert len(modal["blocks"]) == 1
        input_block = modal["blocks"][0]

        assert input_block["type"] == "input"
        assert input_block["block_id"] == "feedback_text"
        assert input_block["optional"] is True  # User can submit without text

        # Verify input element
        element = input_block["element"]
        assert element["type"] == "plain_text_input"
        assert element["action_id"] == "feedback_input"
        assert element["multiline"] is True

    def test_modal_preserves_metadata(self):
        """Modal should preserve message_context in private_metadata."""
        context = '{"run_id": "abc-123", "channel_id": "C456"}'
        modal = create_feedback_modal(context)

        assert modal["private_metadata"] == context


# ============================================================================
# Tests for extract_feedback_text()
# ============================================================================


class TestExtractFeedbackText:
    """Tests for extracting feedback text from modal submission."""

    def test_extract_with_text(self):
        """Should extract feedback text from view state."""
        view_state = {
            "feedback_text": {
                "feedback_input": {
                    "type": "plain_text_input",
                    "value": "The response was inaccurate"
                }
            }
        }

        text = extract_feedback_text(view_state)
        assert text == "The response was inaccurate"

    def test_extract_empty_text(self):
        """Empty feedback text should return empty string."""
        view_state = {
            "feedback_text": {
                "feedback_input": {
                    "type": "plain_text_input",
                    "value": ""
                }
            }
        }

        text = extract_feedback_text(view_state)
        assert text == ""

    # Critical negative tests
    # ------------------------------------------------------------------------

    def test_extract_missing_value(self):
        """Missing 'value' key should return empty string (graceful handling)."""
        view_state = {
            "feedback_text": {
                "feedback_input": {
                    "type": "plain_text_input"
                    # No 'value' key
                }
            }
        }

        text = extract_feedback_text(view_state)
        assert text == ""

    def test_extract_none_value(self):
        """None value should return empty string."""
        view_state = {
            "feedback_text": {
                "feedback_input": {
                    "type": "plain_text_input",
                    "value": None
                }
            }
        }

        text = extract_feedback_text(view_state)
        assert text == ""

    def test_extract_missing_block(self):
        """Missing feedback_text block should return empty string."""
        view_state = {}  # Empty state

        text = extract_feedback_text(view_state)
        assert text == ""

    def test_extract_malformed_state(self):
        """Malformed state structure should return empty string (graceful handling)."""
        view_state = {
            "feedback_text": "not a dict"  # Wrong structure
        }

        text = extract_feedback_text(view_state)
        assert text == ""
