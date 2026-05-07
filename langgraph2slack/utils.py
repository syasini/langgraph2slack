"""Utility functions for Slack integration.

Helper functions for message formatting, markdown conversion, and event detection.
"""

import re
from typing import Dict, List, Optional

# Action ID for the "View Full Details" button in plan block messages
TOOL_CALL_DETAILS_ACTION_ID = "tool_call_details"


def is_bot_mention(text: str, bot_user_id: str) -> bool:
    """Check if bot is mentioned in a message.

    Slack mentions look like: <@U123ABC456>

    Args:
        text: Message text from Slack
        bot_user_id: Bot's Slack user ID (e.g., 'U123ABC456')

    Returns:
        True if bot is mentioned in the text

    Example:
        >>> is_bot_mention("<@U123> hello!", "U123")
        True
        >>> is_bot_mention("hello world", "U123")
        False
    """
    # Pattern matches: <@USER_ID>
    pattern = rf"<@{bot_user_id}>"
    return bool(re.search(pattern, text))


def is_dm(event: dict) -> bool:
    """Check if message is a direct message.

    Args:
        event: Slack event dict

    Returns:
        True if this is a direct message (DM)

    Example:
        >>> is_dm({"channel_type": "im"})
        True
        >>> is_dm({"channel_type": "channel"})
        False
    """
    return event.get("channel_type") == "im"


def clean_markdown(text: str, for_blocks: bool = False) -> str:
    """Convert standard markdown to Slack mrkdwn format.

    Slack uses TWO different markdown parsers:
    1. Streaming (markdown_text): Uses standard markdown (**bold**, *italic*, - bullets)
    2. Blocks (mrkdwn): Uses Slack format (*bold*, _italic_, • bullets)

    This function converts links/images for both, but only converts bold/italic/bullets
    when for_blocks=True to match the block parser.

    Args:
        text: Standard markdown text
        for_blocks: If True, convert bold/italic/bullets to Slack mrkdwn format.
                   If False (default), only convert links/images (for streaming).

    Returns:
        Slack mrkdwn formatted text

    Example:
        >>> clean_markdown("Check [this link](https://example.com)")
        'Check <https://example.com|this link>'
        >>> clean_markdown("**bold** and *italic*", for_blocks=True)
        '*bold* and _italic_'
    """
    # URL pattern that handles parentheses in URLs
    # Matches: non-paren chars OR balanced single-level parens like (text)
    # This handles URLs like: https://wiki.org/File:Name_(detail).jpg
    url_pattern = r"(?:[^()]|\([^()]*\))+"

    # Convert markdown links: [text](url) -> <url|text>
    # Changed [^\]]+ to [^\]]* to allow empty link text
    # Use balanced parentheses pattern for URLs with parens
    text = re.sub(rf"\[([^\]]*)\]\(({url_pattern})\)", r"<\2|\1>", text)

    # Convert markdown images: ![alt](url) -> !<url|alt>
    # Note: Slack doesn't render images inline in text, but this preserves the format
    # Use balanced parentheses pattern to handle URLs with parens
    text = re.sub(rf"!\[([^\]]*)\]\(({url_pattern})\)", r"!<\2|\1>", text)

    # Clean up code blocks: remove language identifier after ```
    # Slack doesn't use language identifiers in the same way
    text = re.sub(r"^```[^\n]*\n", "```\n", text, flags=re.MULTILINE)

    # Only convert bold/italic/bullets for blocks (mrkdwn format)
    # For streaming (markdown_text), keep standard markdown
    if for_blocks:
        # Strategy: Use placeholders to avoid conflicts between bold/italic/bullet conversions

        # Step 1: Convert bullets FIRST (before italic, since * at line start could be mistaken for italic)
        # Convert bullet points: - or * at start of line -> •
        text = re.sub(r"^(\s*)[-*]\s+", r"\1• ", text, flags=re.MULTILINE)

        # Step 2: Convert bold **text** to placeholder to avoid italic regex matching it
        # Use a placeholder that won't appear in normal text
        BOLD_START = "<<<BOLD_START>>>"
        BOLD_END = "<<<BOLD_END>>>"
        text = re.sub(r"\*\*([^*]+)\*\*", rf"{BOLD_START}\1{BOLD_END}", text)

        # Step 3: Now convert italic *text* -> _text_ (won't match placeholders)
        text = re.sub(r"\*([^*]+)\*", r"_\1_", text)

        # Step 4: Replace bold placeholders with Slack format
        text = text.replace(BOLD_START, "*").replace(BOLD_END, "*")

    return text


def create_markdown_text_block(text: str) -> Dict:
    """Create a Slack ``markdown`` block from standard Markdown text.

    Slack's ``markdown`` block accepts standard Markdown and converts it to
    native Block Kit structures (headers, tables, rich text, dividers, etc.).
    """
    return {"type": "markdown", "text": text if text.strip() else " "}


def create_mrkdwn_text_block(text: str) -> Dict:
    """Create a legacy ``section`` block with Slack ``mrkdwn`` text."""
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": text if text.strip() else " "},
    }


def extract_markdown_images(text: str, max_images: int = None) -> List[Dict]:
    """Extract markdown images and return Slack image blocks.

    Finds all markdown images in format: ![alt](url)
    Converts them to Slack image block format.

    Args:
        text: Text containing markdown images
        max_images: Maximum number of image blocks to return (None for unlimited)

    Returns:
        List of Slack image block dicts (limited to max_images if specified)

    Example:
        >>> extract_markdown_images("Here's a chart: ![Sales Chart](https://example.com/chart.png)")
        [{'type': 'image', 'image_url': 'https://example.com/chart.png', 'alt_text': 'Sales Chart'}]
    """
    import logging

    logger = logging.getLogger(__name__)

    # Pattern matches: ![alt text](url)
    # Group 1: alt text (can be empty)
    # Group 2: url - uses balanced parentheses pattern to handle URLs with parens
    # URL pattern: non-paren chars OR balanced single-level parens like (text)
    # This handles URLs like: https://wiki.org/File:Name_(detail).jpg
    url_pattern = r"(?:[^()]|\([^()]*\))+"
    pattern = rf"!\[([^\]]*)\]\(({url_pattern})\)"

    # Find all image markdown patterns using findall (simpler than manual iteration)
    # findall returns list of tuples: [(alt1, url1), (alt2, url2), ...]
    matches = re.findall(pattern, text)

    logger.info(f"Searching for markdown images in text (length={len(text)})")
    logger.info(f"Full text to search: {text}")
    logger.info(f"Found {len(matches)} markdown image patterns")

    image_blocks = []
    for alt_text, url in matches:
        block = {
            "type": "image",
            "image_url": url.strip(),
            "alt_text": alt_text or "Image",  # Default alt text if empty
        }
        logger.info(f"Creating image block: alt_text='{alt_text}', url='{url.strip()}'")
        image_blocks.append(block)

    # Limit to max_images if specified
    if max_images is not None and len(image_blocks) > max_images:
        logger.warning(
            f"Limited images from {len(image_blocks)} to {max_images} (max_images setting)"
        )
        return image_blocks[:max_images]

    return image_blocks


def remove_markdown_images(text: str) -> str:
    """Remove markdown image syntax from text.

    Strips all ``![alt](url)`` occurrences using the same balanced-parentheses
    URL pattern as :func:`extract_markdown_images`, so URLs containing
    parentheses (e.g. Wikipedia links) are handled correctly.

    Args:
        text: Text possibly containing markdown image syntax.

    Returns:
        Text with all markdown image references removed.

    Example:
        >>> remove_markdown_images("See ![chart](https://example.com/chart.png) above.")
        'See  above.'
        >>> remove_markdown_images("Plant: ![name](https://wiki.org/Monstera_(plant))")
        'Plant: '
    """
    url_pattern = r"(?:[^()]|\([^()]*\))+"
    return re.sub(rf"!\[([^\]]*)\]\(({url_pattern})\)", "", text)


def create_feedback_block(
    thread_id: str = None,
    show_feedback_buttons: bool = True,
    show_thread_id: bool = True,
) -> List[Dict]:
    """Create Slack feedback button blocks with optional thread ID.

    Creates a context_actions block with feedback_buttons widget.
    Optionally includes a context block with thread_id for reporting.

    Args:
        thread_id: Optional LangGraph thread ID to display for reporting
        show_feedback_buttons: Whether to show feedback buttons (default: True)
        show_thread_id: Whether to show thread_id in footer (default: True)

    Returns:
        List of Slack block dicts with feedback buttons (and optional context)

    Example usage in Slack API:
        blocks = create_feedback_block(thread_id="abc-123")
        client.chat_postMessage(channel=channel, text=text, blocks=blocks)
    """
    blocks = []

    # Add context block with thread_id if provided and enabled
    if thread_id and show_thread_id:
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"_Thread ID: `{thread_id}`_"}],
            }
        )

    # Add feedback buttons if enabled
    if show_feedback_buttons:
        blocks.append(
            {
                "type": "context_actions",
                "elements": [
                    {
                        "type": "feedback_buttons",
                        "action_id": "feedback",  # Used to identify button clicks
                        "positive_button": {
                            "text": {"type": "plain_text", "text": "Good Response"},
                            "value": "positive",
                        },
                        "negative_button": {
                            "text": {"type": "plain_text", "text": "Bad Response"},
                            "value": "negative",
                        },
                    }
                ],
            }
        )

    return blocks


def create_feedback_modal(message_context: str) -> Dict:
    """Create modal view for collecting negative feedback text.

    Args:
        message_context: JSON string with channel_id, message_ts, and run_id
                        to pass through modal submission

    Returns:
        Modal view dict ready for views.open API

    Example:
        view = create_feedback_modal(message_context='{"channel_id": "C123", ...}')
        client.views_open(trigger_id=trigger_id, view=view)
    """
    return {
        "type": "modal",
        "callback_id": "feedback_modal",
        "private_metadata": message_context,  # Pass context through to submission
        "title": {"type": "plain_text", "text": "Feedback"},
        "submit": {"type": "plain_text", "text": "Submit"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "block_id": "feedback_text",
                "optional": True,  # User can submit without text
                "label": {"type": "plain_text", "text": "What went wrong?"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "feedback_input",
                    "multiline": True,
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Optional: Tell us how we can improve...",
                    },
                },
            }
        ],
    }


def _make_rich_text(text: str) -> Dict:
    """Wrap plain string as a Slack rich_text block element."""
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": [{"type": "text", "text": text}],
            }
        ],
    }


def _make_rich_text_code(text: str) -> Dict:
    """Wrap plain string as a Slack rich_text preformatted (code block) element."""
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_preformatted",
                "elements": [{"type": "text", "text": text}],
            }
        ],
    }


def create_task_card(call, show_details: bool, run_id: Optional[str] = None) -> Dict:
    """Build a Slack task_card block from an ActiveToolCall.

    Args:
        call: ActiveToolCall instance with name, status, args_json(), and result
        show_details: Whether to include truncated input/output preview in the card
        run_id: Optional LangGraph run ID for LangSmith URL source link

    Returns:
        Slack task_card block dict
    """
    task: Dict = {
        "type": "task_card",
        "task_id": call.call_id,
        "title": call.name,
        "status": call.status,
    }

    if show_details:
        args = call.args_json()
        if args.strip():
            truncated = args[:300] + "…" if len(args) > 300 else args
            task["details"] = _make_rich_text_code(truncated)

        if call.result:
            result_str = str(call.result)
            truncated = result_str[:500] + "…" if len(result_str) > 500 else result_str
            task["output"] = _make_rich_text(truncated)

    if run_id:
        task["sources"] = [
            {
                "type": "url",
                "url": f"https://smith.langchain.com/runs/{run_id}",
                "text": "LangSmith trace",
            }
        ]

    return task


def create_plan_block(
    calls: List,
    show_details: bool,
    run_id: Optional[str] = None,
    title: str = "Working on it...",
) -> Dict:
    """Build a Slack plan block containing multiple task_card blocks.

    Args:
        calls: List of ActiveToolCall instances
        show_details: Whether to include truncated input/output in each task card
        run_id: Optional LangGraph run ID for LangSmith source links
        title: Plan block title (default: "Working on it...")

    Returns:
        Slack plan block dict
    """
    tasks = [create_task_card(c, show_details, run_id) for c in calls]
    return {
        "type": "plan",
        "title": title,
        "tasks": tasks,
    }


def create_tool_details_button(plan_ts: str) -> Dict:
    """Return a Slack actions block with a 'View Full Details' button.

    When clicked, the button triggers the tool_call_details action handler,
    which opens a modal with untruncated tool call inputs and outputs.

    Args:
        plan_ts: Slack message timestamp of the plan block message.
                 Used as the key to look up stored tool call data.

    Returns:
        Slack actions block dict with a single button element
    """
    return {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "action_id": TOOL_CALL_DETAILS_ACTION_ID,
                "text": {"type": "plain_text", "text": "View Full Details"},
                "value": plan_ts,
            }
        ],
    }


def extract_feedback_text(view_state: Dict) -> str:
    """Extract feedback text from modal submission view state.

    Args:
        view_state: The view["state"]["values"] dict from view_submission payload

    Returns:
        Feedback text string (empty string if not provided)

    Example:
        text = extract_feedback_text(body["view"]["state"]["values"])
    """
    try:
        # Navigate the nested structure: values -> block_id -> action_id -> value
        feedback_block = view_state.get("feedback_text", {})
        feedback_input = feedback_block.get("feedback_input", {})
        text = feedback_input.get("value", "")
        return text or ""
    except Exception:
        # Return empty string if extraction fails
        return ""
