"""langgraph2slack - Simple LangGraph to Slack integration.

This package provides an easy way to connect LangGraph applications to Slack
with minimal configuration. Just create a SlackBot, optionally add transformers,
and export the FastAPI app to langgraph.json.

Example:
    from langgraph2slack import SlackBot

    bot = SlackBot()

    @bot.transform_input
    async def add_context(message, context):
        return f"User {context.user_name}: {message}"

    app = bot.app  # Export to langgraph.json
"""

from .bot import SlackBot
from .config import BotConfig, MessageContext

# Version is dynamically loaded from package metadata (pyproject.toml)
# Use `uv version --bump patch|minor|major` to update version
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("langgraph2slack")
except PackageNotFoundError:
    # Fallback for development (package not installed)
    __version__ = "0.0.0.dev"

__all__ = ["SlackBot", "BotConfig", "MessageContext"]
