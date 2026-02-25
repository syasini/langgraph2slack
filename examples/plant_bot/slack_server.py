"""Slack bot server using langgraph2slack.

Just a few lines of code to connect the LangGraph agent to Slack!
"""

import re
from langgraph2slack import SlackBot

bot = SlackBot(
    streaming=True,
    reply_in_thread=True,
    show_feedback_buttons=True,
    show_thread_id=True,
    extract_images=True,
    include_metadata=True,
    enable_feedback_comments= True,
    reactions=[
                    {"emoji": "white_check_mark", "target": "user", "when": "complete"},
                    {"emoji": "hourglass", "target": "bot", "when": "processing", "persist": False},
                    {"emoji": "eyes", "target": "user", "when": "processing", "persist":False}
                ],
    stream_buffer_time=0.1,
    show_tool_calls=True,
    show_tool_call_details=True,
    )


# --- Input Transformation ---
# You can modify the user's message before it's sent to the LangGraph agent
@bot.transform_input
async def talk_like_a_plant(message: str) -> str:
    """Transform user messages to sound like a plant."""
    return f"[answer in bullet points only!] {message}"


# --- Output Transformation ---
# You can modify the agent's response before it's sent back to Slack

@bot.transform_output
async def add_greeting(message: str, context) -> str:
    """Add a greeting with the user's name."""
    return f"Hello <@{context.user_id}>!\n\n{message}"


@bot.transform_output
async def test_error_handler(message: str) -> str:
    """Raise an error if message contains 'error' for testing error handling."""
    if "error" in message.lower():
        raise ValueError("This is a test error triggered by 'error' in your message!")
    return message


# Export the app for langgraph.json
app = bot.app
