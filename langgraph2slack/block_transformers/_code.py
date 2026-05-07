"""Markdown code fence → Slack rich_text preformatted block transformer.

Triple-backtick code fences are rendered as Slack rich_text blocks with a
rich_text_preformatted element, which displays in a monospace code style.
Language identifiers are stripped (Slack doesn't support syntax highlighting).
"""

import re

from ..utils import clean_markdown

# Matches a triple-backtick code fence, optionally with a language identifier.
# Captures: optional language tag, code body.
_CODE_FENCE_RE = re.compile(
    r"```(?P<lang>[^\n`]*)?\n(?P<code>.*?)```",
    re.DOTALL,
)


def _build_slack_code_block(code: str) -> dict:
    """Build a Slack rich_text block with a preformatted code element.

    Args:
        code: Raw code text (no backticks).

    Returns:
        A Slack rich_text block dict.
    """
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_preformatted",
                "elements": [{"type": "text", "text": code.strip()}],
            }
        ],
    }


async def render_code_blocks(response: str) -> "str | list[dict]":
    """Convert markdown code fences in a response to Slack rich_text blocks.

    Text outside code fences is preserved as mrkdwn section blocks.
    Language identifiers (e.g. ```python) are stripped.
    Returns the original string unchanged if no code fences are found.

    Args:
        response: LangGraph response text.

    Returns:
        Original string if no code fences found, otherwise a list of Slack block dicts.
    """
    matches = list(_CODE_FENCE_RE.finditer(response))
    if not matches:
        return response

    blocks = []
    last_end = 0

    for match in matches:
        # Text before this code block
        before = response[last_end : match.start()].strip()
        if before:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": clean_markdown(before, for_blocks=True)},
                }
            )

        # The code block itself
        code = match.group("code")
        blocks.append(_build_slack_code_block(code))

        last_end = match.end()

    # Trailing text after the last code block
    after = response[last_end:].strip()
    if after:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": clean_markdown(after, for_blocks=True)},
            }
        )

    return blocks
