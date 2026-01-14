"""Basic example of langgraph2slack usage.

This is the simplest possible setup - just create a bot and export the app.
All configuration comes from environment variables.
"""

from langgraph2slack import SlackBot

# Create bot with minimal configuration
# Reads SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, and ASSISTANT_ID from .env
bot = SlackBot()

# Export the FastAPI app for langgraph.json
app = bot.app
