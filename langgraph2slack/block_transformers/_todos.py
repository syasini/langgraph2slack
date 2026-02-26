"""Markdown todo list → Slack rich_text list block transformer.

Renders todo items using Slack's native rich_text_list block (bullet style).
Completed items are displayed with strikethrough.

Supported input formats:
    - [ ] unchecked item     (standard markdown)
    - [x] checked item       (standard markdown)
    ☐ unchecked item         (LLM direct output)
    ☑ checked item           (LLM direct output)
"""

import re
from ..utils import clean_markdown

# Matches a run of one or more consecutive todo lines.
# Handles both standard markdown (- [ ]) and direct unicode (☐/☑) formats.
_TODO_BLOCK_RE = re.compile(
    r"(?:^[ \t]*(?:[-*][ \t]\[[ xX]\]|[☐☑])[ \t]*[^\n]*\n?)+",
    re.MULTILINE,
)

# Extracts checked state and text from a single todo line
_TODO_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]\[(?P<md_state>[ xX])\]|(?P<unicode_unchecked>☐)|(?P<unicode_checked>☑))[ \t]*(?P<text>.*)$"
)


def _is_checked(line_match: re.Match) -> bool:
    """Return True if the todo line is in a checked/completed state."""
    if line_match.group("md_state") is not None:
        return line_match.group("md_state").strip().lower() == "x"
    return line_match.group("unicode_checked") is not None


def _parse_todo_items(match_text: str) -> list[dict]:
    """Parse a block of todo lines into rich_text_section elements.

    Args:
        match_text: A multi-line string of consecutive todo items.

    Returns:
        List of rich_text_section dicts, one per item. Checked items have
        strikethrough style applied.
    """
    sections = []
    for line in match_text.splitlines():
        m = _TODO_LINE_RE.match(line)
        if not m:
            continue
        text = m.group("text").strip()
        if not text:
            continue

        checked = _is_checked(m)
        text_element = {"type": "text", "text": text}
        if checked:
            text_element["style"] = {"strike": True}

        sections.append({
            "type": "rich_text_section",
            "elements": [text_element],
        })

    return sections


def _build_slack_todo_block(sections: list[dict]) -> dict:
    """Build a Slack rich_text block with a bullet list of todo items.

    Args:
        sections: List of rich_text_section dicts (one per item).

    Returns:
        A Slack rich_text block dict with a rich_text_list element.
    """
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_list",
                "style": "bullet",
                "elements": sections,
            }
        ],
    }


async def render_todo_lists(response: str) -> "str | list[dict]":
    """Convert markdown todo lists in a response to Slack rich_text list blocks.

    Handles both ``- [ ] / - [x]`` markdown syntax and ``☐ / ☑`` unicode
    characters that LLMs sometimes output directly.

    Completed items are rendered with strikethrough. Text surrounding the
    todo blocks is preserved as mrkdwn section blocks.
    Returns the original string unchanged if no todo items are found.

    Args:
        response: LangGraph response text.

    Returns:
        Original string if no todo lists found, otherwise a list of Slack block dicts.
    """
    matches = list(_TODO_BLOCK_RE.finditer(response))
    if not matches:
        return response

    blocks = []
    last_end = 0

    for match in matches:
        # Text before this todo block
        before = response[last_end : match.start()].strip()
        if before:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": clean_markdown(before, for_blocks=True)},
            })

        # The todo block itself
        sections = _parse_todo_items(match.group())
        if sections:
            blocks.append(_build_slack_todo_block(sections))

        last_end = match.end()

    # Trailing text after the last todo block
    after = response[last_end:].strip()
    if after:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": clean_markdown(after, for_blocks=True)},
        })

    return blocks
