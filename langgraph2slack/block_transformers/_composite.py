"""Composite multi-pattern block renderer.

Scans the response text in a single pass for all supported patterns (tables,
todo lists, code blocks) and renders them in document order. Text between
patterns is preserved as mrkdwn section blocks.
"""

import logging
import re
from ..utils import clean_markdown
from ._tables import _parse_markdown_table, _build_slack_table_block
from ._todos import _TODO_BLOCK_RE, _parse_todo_items, _build_slack_todo_block
from ._code import _CODE_FENCE_RE, _build_slack_code_block

logger = logging.getLogger(__name__)

# Markdown table: header row, separator, one or more data rows
_TABLE_RE = re.compile(
    r"(\|[^\n]+\|\n?)"        # header row
    r"\|[-| :]+\|\n?"         # separator row
    r"((?:\|[^\n]+\|\n?)+)",  # data rows
)


async def render_blocks(response: str) -> "str | list[dict]":
    """Convert all supported markdown patterns to Slack blocks in document order.

    Handles tables, todo lists, and code fences in a single pass. Each pattern
    is rendered at its original position in the text; surrounding text becomes
    mrkdwn section blocks.

    Returns the original string unchanged if none of the supported patterns are
    found — making it safe to always register this transformer.

    Note:
        Slack enforces a hard limit of one ``table`` block per message
        (API error: ``only_one_table_allowed``). If the response contains
        multiple tables, only the first is rendered as a native table block;
        subsequent tables are left as markdown text.

    Args:
        response: LangGraph response text.

    Returns:
        Original string if no patterns found, otherwise a list of Slack block dicts.

    Example:
        from langgraph2slack.block_transformers import render_blocks

        @bot.transform_output
        async def my_renderer(message: str) -> str | list[dict]:
            return await render_blocks(message)

        # or register directly:
        bot.transform_output(render_blocks)
    """
    # Collect all pattern match positions: (start, end, type, match_object)
    candidates: list[tuple[int, int, str, re.Match]] = []

    for m in _TABLE_RE.finditer(response):
        candidates.append((m.start(), m.end(), "table", m))

    for m in _TODO_BLOCK_RE.finditer(response):
        candidates.append((m.start(), m.end(), "todo", m))

    for m in _CODE_FENCE_RE.finditer(response):
        candidates.append((m.start(), m.end(), "code", m))

    if not candidates:
        return response

    # Sort by position in the document
    candidates.sort(key=lambda c: c[0])

    # Remove overlapping matches (keep the one that starts earliest)
    non_overlapping: list[tuple[int, int, str, re.Match]] = []
    last_end = 0
    for start, end, kind, match in candidates:
        if start >= last_end:
            non_overlapping.append((start, end, kind, match))
            last_end = end

    if not non_overlapping:
        return response

    blocks: list[dict] = []
    cursor = 0
    table_rendered = False

    for start, end, kind, match in non_overlapping:
        # Text before this pattern
        before = response[cursor:start].strip()
        if before:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": clean_markdown(before, for_blocks=True)},
            })

        # Render the matched pattern
        if kind == "table":
            headers, rows, _, _ = _parse_markdown_table(match.group())
            if headers:
                if not table_rendered:
                    blocks.append(_build_slack_table_block(headers, rows))
                    table_rendered = True
                else:
                    # Slack only allows one table block per message; leave extra tables as mrkdwn
                    blocks.append({
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": match.group()},
                    })
        elif kind == "todo":
            sections = _parse_todo_items(match.group())
            if sections:
                blocks.append(_build_slack_todo_block(sections))
        elif kind == "code":
            code = match.group("code")
            blocks.append(_build_slack_code_block(code))

        cursor = end

    # Trailing text after the last pattern
    after = response[cursor:].strip()
    if after:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": clean_markdown(after, for_blocks=True)},
        })

    return blocks
