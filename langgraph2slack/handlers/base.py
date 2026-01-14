"""Base handler class with shared functionality between streaming and non-streaming handlers.

This module contains common logic used by both MessageHandler and StreamingHandler
to reduce code duplication and ensure consistency.
"""

import logging
import uuid
from typing import Dict, List

from ..config import MessageContext
from ..transformers import TransformerChain
from ..utils import create_feedback_block, extract_markdown_images

logger = logging.getLogger(__name__)


class BaseHandler:
    """Base class for message handlers with shared functionality.

    This class provides common methods and utilities used by both
    streaming and non-streaming handlers.
    """

    def __init__(
        self,
        assistant_id: str,
        input_transformers: TransformerChain,
        output_transformers: TransformerChain,
        show_feedback_buttons: bool = True,
        show_thread_id: bool = True,
        extract_images: bool = True,
        max_image_blocks: int = 5,
    ):
        """Initialize base handler.

        Args:
            assistant_id: LangGraph assistant ID
            input_transformers: Chain of input transformers
            output_transformers: Chain of output transformers
            show_feedback_buttons: Whether to show feedback buttons (default: True)
            show_thread_id: Whether to show thread_id in footer (default: True)
            extract_images: Extract image markdown and render as blocks (default: True)
            max_image_blocks: Maximum number of image blocks to include (default: 5)
        """
        self.assistant_id = assistant_id
        self.input_transformers = input_transformers
        self.output_transformers = output_transformers
        self.show_feedback_buttons = show_feedback_buttons
        self.show_thread_id = show_thread_id
        self.extract_images = extract_images
        self.max_image_blocks = max_image_blocks

    async def _apply_input_transforms(
        self,
        message: str,
        context: MessageContext,
    ) -> str:
        """Apply input transformers to message.

        Args:
            message: Raw message text from Slack
            context: Message context with user/channel info

        Returns:
            Transformed message
        """
        transformed = await self.input_transformers.apply(message, context)
        logger.debug(f"After input transforms: {transformed[:100]}...")
        return transformed

    def _determine_thread_timestamp(self, context: MessageContext) -> str:
        """Determine thread timestamp for LangGraph thread ID.

        If message is in a thread, use thread_ts; otherwise use message_ts.

        Args:
            context: Message context with user/channel info

        Returns:
            Thread timestamp string
        """
        return context.thread_ts or context.message_ts

    def _create_thread_id(self, channel_id: str, timestamp: str) -> str:
        """Create LangGraph thread ID from Slack identifiers.

        Uses UUID5 to generate a deterministic, valid thread ID.
        This ensures same Slack thread always maps to same LangGraph thread.

        Args:
            channel_id: Slack channel ID (e.g., C123ABC456)
            timestamp: Slack thread or message timestamp (e.g., 1234567890.123456)

        Returns:
            UUID string for LangGraph thread ID

        Example:
            >>> _create_thread_id("C123ABC", "1234567890.123456")
            "a1b2c3d4-e5f6-5789-a1b2-c3d4e5f67890"
        """
        # Generate deterministic UUID from Slack identifiers
        # Format: slack.{channel_id}.{timestamp}
        identifier = f"slack.{channel_id}.{timestamp}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, identifier))

    def _create_blocks(
        self,
        response_text: str,
        thread_id: str,
    ) -> List[Dict]:
        """Create Slack blocks for images and feedback.

        Extracts markdown images from response (if enabled), limits them to max_image_blocks,
        and adds feedback blocks.

        Args:
            response_text: Complete response text (may contain markdown images)
            thread_id: LangGraph thread ID for feedback tracking

        Returns:
            List of Slack block dicts (image blocks + feedback blocks)
        """
        # Extract markdown images from response if enabled
        if self.extract_images:
            image_blocks = extract_markdown_images(response_text, max_images=self.max_image_blocks)
        else:
            image_blocks = []

        # Create feedback blocks
        feedback_blocks = create_feedback_block(
            thread_id=thread_id,
            show_feedback_buttons=self.show_feedback_buttons,
            show_thread_id=self.show_thread_id,
        )

        # Combine and return
        blocks = image_blocks + feedback_blocks
        logger.info(
            f"Created {len(image_blocks)} image blocks and "
            f"{len(feedback_blocks)} feedback blocks"
        )

        return blocks
