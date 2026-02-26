"""Base handler class with shared functionality between streaming and non-streaming handlers.

This module contains common logic used by both MessageHandler and StreamingHandler
to reduce code duplication and ensure consistency.
"""

import logging
import uuid
from typing import Dict, List, Optional

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
        show_tool_calls: bool = False,
        show_tool_call_details: bool = True,
        tool_call_store: Optional[Dict] = None,
        structured_output: bool = False,
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
            show_tool_calls: Show plan block with tool call status (default: False)
            show_tool_call_details: Show truncated input/output preview + View Details button (default: True)
            tool_call_store: Shared dict mapping plan_ts -> list[ActiveToolCall] for the
                             "View Full Details" modal. Passed by reference from SlackBot.
            structured_output: When True, pass the full LangGraph state dict to output
                               transformers instead of extracting only the last message text.
                               Use this when your graph has custom state keys (e.g. image_url,
                               table) that you want to access in a transform_output transformer.
        """
        self.assistant_id = assistant_id
        self.input_transformers = input_transformers
        self.output_transformers = output_transformers
        self.show_feedback_buttons = show_feedback_buttons
        self.show_thread_id = show_thread_id
        self.extract_images = extract_images
        self.max_image_blocks = max_image_blocks
        self.show_tool_calls = show_tool_calls
        self.show_tool_call_details = show_tool_call_details
        self.structured_output = structured_output
        # Shared store: plan_ts -> list[ActiveToolCall]; same dict reference across bot + handlers
        self.tool_call_store: Dict = tool_call_store if tool_call_store is not None else {}

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
        custom_blocks: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Create Slack blocks for images and feedback.

        When custom_blocks is provided (returned by a block-returning transform_output
        transformer), skips image extraction and uses those blocks directly, appending
        only the feedback/thread-id blocks.

        When custom_blocks is None (default), extracts markdown images from response
        (if enabled) and adds feedback blocks — existing behavior.

        Args:
            response_text: Complete response text (may contain markdown images).
                           Used for image extraction when custom_blocks is None.
            thread_id: LangGraph thread ID for feedback tracking
            custom_blocks: Pre-built Slack blocks from a transformer (optional).
                           When provided, these replace the default image extraction.

        Returns:
            List of Slack block dicts
        """
        # Create feedback blocks (always appended regardless of custom_blocks)
        feedback_blocks = create_feedback_block(
            thread_id=thread_id,
            show_feedback_buttons=self.show_feedback_buttons,
            show_thread_id=self.show_thread_id,
        )

        if custom_blocks is not None:
            # Use transformer-provided blocks; skip image extraction
            blocks = custom_blocks + feedback_blocks
            logger.info(
                f"Using {len(custom_blocks)} custom blocks from transformer and "
                f"{len(feedback_blocks)} feedback blocks"
            )
        else:
            # Default path: extract markdown images from response if enabled
            if self.extract_images:
                image_blocks = extract_markdown_images(
                    response_text, max_images=self.max_image_blocks
                )
            else:
                image_blocks = []

            blocks = image_blocks + feedback_blocks
            logger.info(
                f"Created {len(image_blocks)} image blocks and "
                f"{len(feedback_blocks)} feedback blocks"
            )

        return blocks

    def _extract_text_fallback(self, run_output: Dict) -> str:
        """Extract last message content as a plain text fallback.

        Used when a transformer returns custom blocks — we still need a short
        text string for Slack's notification preview (the ``text=`` parameter
        in chat_update / say).

        Args:
            run_output: Full LangGraph state dict from runs.join()

        Returns:
            Last AI message content as a string, or empty string if unavailable.
        """
        try:
            messages = run_output.get("messages", [])
            if not messages:
                return ""
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
            return str(content)
        except Exception:
            return ""
