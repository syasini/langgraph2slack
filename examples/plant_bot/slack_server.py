"""Slack bot server using langgraph2slack."""

from langgraph2slack import SlackBot
from langgraph2slack.block_transformers import render_todo_lists, render_code_blocks
from langgraph2slack.block_transformers._tables import _parse_markdown_table, _build_slack_table_block
from langgraph2slack.utils import clean_markdown

bot = SlackBot(
    streaming=True,
    reply_in_thread=True,
    show_feedback_buttons=True,
    show_thread_id=True,
    extract_images=True,
    include_metadata=True,
    enable_feedback_comments=True,
    reactions=[
        {"emoji": "white_check_mark", "target": "user", "when": "complete"},
        {"emoji": "hourglass", "target": "bot", "when": "processing", "persist": False},
        {"emoji": "eyes", "target": "user", "when": "processing", "persist": False},
    ],
    show_tool_calls=True,
    show_tool_call_details=True,
)


@bot.transform_input
async def talk_like_a_plant(message: str) -> str:
    return f"[answer in bullet points!] {message}"


@bot.transform_output
async def render_blocks(message: str) -> "str | list":
    """Render markdown tables, todo lists, and code blocks as Slack blocks.

    If the response contains multiple tables, each table (with its preceding
    text) is returned as a separate message payload — list[list[dict]] —
    which the bot sends as consecutive thread messages. Single-table or
    table-free responses return a flat list[dict] (one message).
    """
    # Split on markdown tables first
    messages: list[list[dict]] = []
    remaining = message

    while True:
        headers, rows, before, after = _parse_markdown_table(remaining)
        if headers is None:
            break
        msg_blocks: list[dict] = []
        if before.strip():
            msg_blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": clean_markdown(before.strip(), for_blocks=True)},
            })
        msg_blocks.append(_build_slack_table_block(headers, rows))
        messages.append(msg_blocks)
        remaining = after

    if not messages:
        # No tables — fall through to todo/code transformers as a single message
        result = await render_todo_lists(message)
        if isinstance(result, str):
            result = await render_code_blocks(result)
        return result

    # Trailing text after the last table (may contain todos/code — keep as mrkdwn for now)
    if remaining.strip():
        messages.append([{
            "type": "section",
            "text": {"type": "mrkdwn", "text": clean_markdown(remaining.strip(), for_blocks=True)},
        }])

    # Single table → flat list[dict] (normal one-message path)
    if len(messages) == 1:
        return messages[0]

    # Multiple tables → list[list[dict]] signals the bot to send separate messages
    return messages


# Export the app for langgraph.json
app = bot.app
