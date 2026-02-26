"""Slack bot server using langgraph2slack."""

from langgraph2slack import SlackBot
from langgraph2slack.block_transformers import render_tables, render_todo_lists, render_code_blocks

bot = SlackBot(
    streaming=False,
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
async def render_blocks(message: str) -> "str | list[dict]":
    """Render markdown tables, todo lists, and code blocks as Slack blocks."""
    result = await render_tables(message)
    if isinstance(result, str):
        result = await render_todo_lists(result)
    if isinstance(result, str):
        result = await render_code_blocks(result)
    return result


# Export the app for langgraph.json
app = bot.app
