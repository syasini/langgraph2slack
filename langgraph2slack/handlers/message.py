"""Message handling logic for non-streaming communication.

This module handles synchronous message processing where we wait for the
complete LangGraph response before sending to Slack.
"""

import logging
from typing import Dict, List, Optional, Tuple

from ..config import MessageContext
from ..transformers import TransformerChain
from ..utils import clean_markdown
from .base import BaseHandler

logger = logging.getLogger(__name__)


class MessageHandler(BaseHandler):
    """Handles non-streaming message processing between Slack and LangGraph.

    This handler waits for the complete response from LangGraph before
    sending to Slack. Simpler but higher latency than streaming.

    Flow:
    1. Apply input transformers to Slack message
    2. Send to LangGraph and wait for complete response
    3. Extract response message
    4. Apply output transformers
    5. Format for Slack and return
    """

    def __init__(
        self,
        langgraph_client,
        assistant_id: str,
        input_transformers: TransformerChain,
        output_transformers: TransformerChain,
        show_feedback_buttons: bool = True,
        show_thread_id: bool = True,
        extract_images: bool = True,
        max_image_blocks: int = 5,
        metadata_builder=None,
    ):
        """Initialize message handler.

        Args:
            langgraph_client: LangGraph SDK client instance
            assistant_id: LangGraph assistant ID
            input_transformers: Chain of input transformers
            output_transformers: Chain of output transformers
            show_feedback_buttons: Whether to show feedback buttons (default: True)
            show_thread_id: Whether to show thread_id in footer (default: True)
            extract_images: Extract image markdown and render as blocks (default: True)
            max_image_blocks: Maximum number of image blocks to include (default: 5)
            metadata_builder: Async function to build metadata dict from MessageContext
        """
        # Initialize base class
        super().__init__(
            assistant_id=assistant_id,
            input_transformers=input_transformers,
            output_transformers=output_transformers,
            show_feedback_buttons=show_feedback_buttons,
            show_thread_id=show_thread_id,
            extract_images=extract_images,
            max_image_blocks=max_image_blocks,
        )
        # Store handler-specific attributes
        self.client = langgraph_client
        self.metadata_builder = metadata_builder

    async def process_message(
        self,
        message: str,
        context: MessageContext,
    ) -> Tuple[str, List[Dict], Optional[str], Optional[str]]:
        """Process a message through the full pipeline.

        Main entry point for non-streaming message processing.

        Args:
            message: Raw message text from Slack
            context: Message context with user/channel info

        Returns:
            Tuple of (formatted_text, blocks, thread_id, run_id)
            - formatted_text: Processed response ready to send to Slack
            - blocks: List of Slack blocks (images + feedback)
            - thread_id: LangGraph thread ID
            - run_id: LangGraph run ID for feedback
        """
        logger.info(
            f"Processing message from user {context.user_id} in channel {context.channel_id}"
        )

        # Step 1: Apply input transformers
        transformed_input = await self._apply_input_transforms(message, context)

        # Step 2: Create thread ID for conversation continuity
        thread_timestamp = self._determine_thread_timestamp(context)
        langgraph_thread = self._create_thread_id(context.channel_id, thread_timestamp)
        logger.info(f"Using LangGraph thread: {langgraph_thread}")

        # Step 3: Send to LangGraph and wait for complete response
        langgraph_response, run_id = await self._invoke_langgraph(
            transformed_input, langgraph_thread, context
        )

        # Step 4: Extract the actual message content
        response_text = self._extract_message_content(langgraph_response)
        logger.debug(f"LangGraph response: {response_text[:100]}...")

        # Step 5: Apply output transformers
        # Each transformer can modify the response (add footer, filter, etc.)
        transformed_output = await self.output_transformers.apply(response_text, context)

        # Step 6: Convert markdown to Slack's format
        slack_formatted = clean_markdown(transformed_output)

        # Step 7: Extract markdown images and create blocks
        blocks = self._create_blocks(transformed_output, langgraph_thread)

        return slack_formatted, blocks, langgraph_thread, run_id

    async def _invoke_langgraph(
        self, message: str, thread_id: str, context: MessageContext
    ) -> Tuple[dict, str]:
        """Invoke LangGraph and wait for complete response.

        Args:
            message: Message to send to LangGraph
            thread_id: Thread ID for conversation continuity
            context: Message context for metadata building

        Returns:
            Tuple of (completed_run, run_id)
            - completed_run: Complete LangGraph response with state
            - run_id: The run ID for feedback submission

        Raises:
            Exception: If LangGraph invocation fails
        """
        # Build metadata if builder is provided
        metadata = await self.metadata_builder(context) if self.metadata_builder else {}

        try:
            # Use LangGraph SDK to create a run and wait for completion
            run = await self.client.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id,
                input={"messages": [{"role": "user", "content": message}]},
                if_not_exists="create",
                metadata=metadata,
            )

            run_id = run["run_id"]
            logger.info(f"Created run with ID: {run_id}")

            # Wait for the run to complete
            completed_run = await self.client.runs.join(thread_id, run_id)

            # Return the completed run which contains the final state values
            return completed_run, run_id

        except Exception as e:
            logger.error(f"LangGraph invocation failed: {e}", exc_info=True)
            raise

    def _extract_message_content(self, langgraph_response: dict) -> str:
        """Extract message text from LangGraph response.

        runs.join() returns the graph state directly with a "messages" list.
        We want the content from the last message (the assistant's response).

        Args:
            langgraph_response: Full response dict from LangGraph run

        Returns:
            Message content as string

        Note:
            If extraction fails, returns a friendly error message
            instead of raising an exception.
        """
        try:
            # Get messages directly from response (runs.join() returns state directly)
            messages = langgraph_response.get("messages", [])

            if not messages:
                logger.warning("No messages found in LangGraph response")
                return "I apologize, but I couldn't generate a response."

            # Get the last message (assistant's response)
            last_message = messages[-1]

            # Extract content - handle both string and structured formats
            content = last_message.get("content", "")

            if isinstance(content, str):
                # Simple string content
                return content

            elif isinstance(content, list):
                # Content is a list of blocks (e.g., text + images)
                # Extract text blocks and concatenate
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                return "".join(text_parts)

            else:
                # Unexpected format - convert to string as fallback
                logger.warning(f"Unexpected content type: {type(content)}")
                return str(content)

        except Exception as e:
            logger.error(f"Failed to extract message content: {e}", exc_info=True)
            return "I encountered an error while processing the response."
