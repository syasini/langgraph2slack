"""Markdown table → Slack table block transformer."""

import re
from typing import Optional
from ..utils import clean_markdown


def _parse_markdown_table(text: str) -> tuple[Optional[list], Optional[list], str, str]:
    """Find and parse the first markdown table in text.

    Args:
        text: Text that may contain a markdown table.

    Returns:
        (headers, rows, before, after) where before/after are the text
        segments surrounding the table. Returns (None, None, text, "") if
        no table is found.
    """
    pattern = re.compile(
        r"(\|[^\n]+\|\n?)"        # header row
        r"\|[-| :]+\|\n?"         # separator row (---|---)
        r"((?:\|[^\n]+\|\n?)+)",  # data rows (one or more)
    )
    match = pattern.search(text)
    if not match:
        return None, None, text, ""

    header_line = match.group(1)
    data_lines = match.group(2)

    # Parse header cells
    headers = [c.strip() for c in header_line.strip().strip("|").split("|")]
    headers = [h for h in headers if h]  # drop empty edge cells

    # Parse data rows
    rows = []
    for line in data_lines.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Pad or trim to match header count
        while len(cells) < len(headers):
            cells.append("")
        rows.append(cells[: len(headers)])

    before = text[: match.start()].strip()
    after = text[match.end() :].strip()

    return headers, rows, before, after


def _build_slack_table_block(headers: list, rows: list) -> dict:
    """Convert parsed table data into a Slack table block.

    Args:
        headers: List of header cell strings.
        rows: List of rows, each a list of cell strings.

    Returns:
        A Slack table block dict.
    """
    column_settings = [{"is_wrapped": True} for _ in headers]

    slack_rows = []

    # Header row
    slack_rows.append([{"type": "raw_text", "text": h} for h in headers])

    # Data rows
    for row in rows:
        slack_rows.append([{"type": "raw_text", "text": str(cell)} for cell in row])

    return {
        "type": "table",
        "column_settings": column_settings,
        "rows": slack_rows,
    }


async def render_tables(response: str) -> "str | list[dict]":
    """Convert the first markdown table in a response to a Slack table block.

    Renders the first table as a native Slack table block. Any text before the
    table and all text after it (including any additional markdown tables) are
    preserved as a single mrkdwn section block each.

    Returns the original string unchanged if no table is found.

    Note:
        Slack enforces a hard limit of one ``table`` block per message
        (API error: ``only_one_table_allowed``). Additional tables in the
        response are left as markdown text rather than converted.

    Args:
        response: LangGraph response text.

    Returns:
        Original string if no table found, otherwise a list of Slack block dicts.
    """
    headers, rows, before, after = _parse_markdown_table(response)
    if headers is None:
        return response

    blocks = []

    if before:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": clean_markdown(before, for_blocks=True)},
        })

    blocks.append(_build_slack_table_block(headers, rows))

    if after:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": clean_markdown(after, for_blocks=True)},
        })

    return blocks
