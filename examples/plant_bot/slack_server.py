"""Slack bot server using langgraph2slack."""

from langgraph2slack import SlackBot

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


# Export the app for langgraph.json
app = bot.app
