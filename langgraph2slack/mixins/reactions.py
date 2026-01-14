"""Mixin for Slack reaction operations.

This mixin provides shared reaction functionality to avoid code duplication
between SlackBot and StreamingHandler.
"""

import logging
import asyncio

logger = logging.getLogger(__name__)


class ReactionMixin:
    """Mixin providing Slack reaction operations.

    This class centralizes reaction logic that was previously duplicated
    across SlackBot and StreamingHandler. It requires a Slack client to be
    passed during initialization.

    Usage:
        class MyClass:
            def __init__(self, slack_client):
                self._reactions = ReactionMixin(slack_client)

            async def some_method(self):
                await self._reactions.add(channel_id, message_ts, "eyes")
    """

    def __init__(self, slack_client):
        """Initialize the ReactionMixin.

        Args:
            slack_client: Slack API client with reactions_add/reactions_remove methods
        """
        self._slack_client = slack_client

    async def add(
        self,
        channel_id: str,
        message_ts: str,
        emoji: str,
    ) -> None:
        """Add emoji reaction to a Slack message.

        Args:
            channel_id: Slack channel ID
            message_ts: Slack message timestamp
            emoji: Emoji name (without colons, e.g., "eyes", "hourglass")
        """
        try:
            await self._slack_client.reactions_add(
                channel=channel_id,
                timestamp=message_ts,
                name=emoji,
            )
            logger.debug(f"Added reaction :{emoji}: to message {channel_id}:{message_ts}")
        except Exception as e:
            logger.warning(f"Failed to add reaction :{emoji}:: {e}")

    async def remove(
        self,
        channel_id: str,
        message_ts: str,
        emoji: str,
    ) -> None:
        """Remove emoji reaction from a Slack message.

        Args:
            channel_id: Slack channel ID
            message_ts: Slack message timestamp
            emoji: Emoji name (without colons, e.g., "eyes", "hourglass")
        """
        try:
            await self._slack_client.reactions_remove(
                channel=channel_id,
                timestamp=message_ts,
                name=emoji,
            )
            logger.debug(f"Removed reaction :{emoji}: from message {channel_id}:{message_ts}")
        except Exception as e:
            logger.warning(f"Failed to remove reaction :{emoji}:: {e}")

    async def add_parallel(
        self,
        reactions: list[dict],
        channel_id: str,
        message_ts: str,
    ) -> None:
        """Add multiple reactions in parallel with error handling.

        This method adds all reactions concurrently using asyncio.gather,
        which is much faster than sequential addition when multiple reactions
        are configured.

        Args:
            reactions: List of reaction config dicts (with "emoji" key)
            channel_id: Slack channel ID
            message_ts: Slack message timestamp
        """
        if not reactions:
            return

        try:
            # Add all reactions concurrently
            await asyncio.gather(
                *[self.add(channel_id, message_ts, r.get("emoji")) for r in reactions],
                return_exceptions=True,  # Don't fail entire batch if one fails
            )
        except Exception as e:
            logger.error(f"Parallel reactions failed: {e}", exc_info=True)
