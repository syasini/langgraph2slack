"""Example with input and output transformers.

Shows how to customize message processing with transformers.
"""

from langgraph2slack import SlackBot

# Create bot
bot = SlackBot()


# Add input transformer to add user context
@bot.transform_input
async def add_user_context(message, context):
    """Add user information to the message before sending to LangGraph."""
    return f"[User {context.user_id} in channel {context.channel_id}]\n{message}"


# Add another input transformer (they chain!)
@bot.transform_input
async def add_channel_type(message, context):
    """Add channel type information."""
    channel_type = "DM" if context.is_dm else "channel"
    return f"[Via {channel_type}]\n{message}"


# Add output transformer to format responses
@bot.transform_output
async def add_footer(response, context):
    """Add a footer to all responses."""
    return f"{response}\n\n_Powered by LangGraph_"


# Export app
app = bot.app
