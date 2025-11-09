"""Streaming response handling for low-latency communication.

This module implements true streaming where LangGraph chunks are immediately
forwarded to Slack as they arrive, minimizing latency.
"""

import logging
import asyncio
from typing import Optional

from ..config import MessageContext
from ..transformers import TransformerChain
from ..utils import clean_markdown
from .base import BaseHandler

logger = logging.getLogger(__name__)


class StreamingHandler(BaseHandler):
    """Handles streaming message processing with immediate chunk forwarding.

    This handler provides true low-latency streaming:
    - LangGraph chunk arrives → immediately sent to Slack
    - No waiting for complete response
    - Better user experience with instant feedback

    Flow:
    1. Apply input transformers
    2. Start Slack stream
    3. Stream from LangGraph, forwarding each chunk immediately
    4. Stop Slack stream with optional images/blocks
    """

    def __init__(
        self,
        langgraph_client,
        slack_client,
        assistant_id: str,
        input_transformers: TransformerChain,
        output_transformers: TransformerChain,
        reply_in_thread: bool = True,
        show_feedback_buttons: bool = True,
        show_thread_id: bool = True,
        extract_images: bool = True,
        max_image_blocks: int = 5,
        metadata_builder=None,
        message_types: list[str] = None,
    ):
        """Initialize streaming handler.

        Args:
            langgraph_client: LangGraph SDK client instance
            slack_client: Slack Bolt AsyncApp client
            assistant_id: LangGraph assistant ID
            input_transformers: Chain of input transformers
            output_transformers: Chain of output transformers
            reply_in_thread: Reply in thread vs main channel (default: True)
            show_feedback_buttons: Whether to show feedback buttons (default: True)
            show_thread_id: Whether to show thread_id in footer (default: True)
            extract_images: Extract image markdown and render as blocks (default: True)
            max_image_blocks: Maximum number of image blocks to include (default: 5)
            metadata_builder: Async function to build metadata dict from MessageContext
            message_types: List of message types to process (default: ["AIMessageChunk"])
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
        self.langgraph_client = langgraph_client
        self.slack_client = slack_client
        self.reply_in_thread = reply_in_thread
        self.metadata_builder = metadata_builder
        self.message_types = message_types if message_types is not None else ["AIMessageChunk"]

    async def process_message(
        self,
        message: str,
        context: MessageContext,
        bot_reactions: list[dict] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Process message with streaming.

        Main entry point for streaming message processing.
        Sends response directly to Slack via streaming.

        Args:
            message: Raw message text from Slack
            context: Message context with user/channel info
            bot_reactions: List of reaction configs for bot message (default: None)

        Returns:
            Tuple of (stream_ts, thread_id, run_id) for feedback tracking
        """
        if bot_reactions is None:
            bot_reactions = []
        logger.info(f"Streaming message from user {context.user_id} in channel {context.channel_id}")

        # Step 1: Apply input transformers
        transformed_input = await self._apply_input_transforms(message, context)

        # Step 2: Create thread ID
        thread_timestamp = self._determine_thread_timestamp(context)
        langgraph_thread = self._create_thread_id(context.channel_id, thread_timestamp)
        logger.info(f"Using LangGraph thread: {langgraph_thread}")

        # Step 3: Get team ID for Slack streaming API
        team_id = await self._get_team_id()

        # Step 4: Determine thread_ts based on reply_in_thread setting
        if self.reply_in_thread:
            # Always reply in thread (use message ts if not already in thread)
            slack_thread_ts = context.thread_ts or context.message_ts
        else:
            # Only reply in thread if message was already in a thread
            slack_thread_ts = context.thread_ts

        # Step 5: Start Slack stream
        stream_ts = await self._start_slack_stream(
            channel_id=context.channel_id,
            thread_ts=slack_thread_ts,
            user_id=context.user_id,
            team_id=team_id,
        )
        logger.info(f"Started Slack stream with ts: {stream_ts}")

        # Add bot-processing reactions to the streaming message
        bot_processing_reactions = [r for r in bot_reactions if r.get("target") == "bot" and r.get("when") == "processing"]
        for reaction in bot_processing_reactions:
            await self._add_reaction(context.channel_id, stream_ts, reaction.get("emoji"))

        try:
            # Step 6: Stream from LangGraph and forward to Slack
            # CRITICAL: Each chunk is sent immediately as it arrives
            complete_response, run_id = await self._stream_from_langgraph_to_slack(
                message=transformed_input,
                langgraph_thread=langgraph_thread,
                slack_channel=context.channel_id,
                slack_stream_ts=stream_ts,
                context=context,
            )

            # Step 7: Stop stream with optional image blocks
            await self._stop_slack_stream(
                channel_id=context.channel_id,
                stream_ts=stream_ts,
                complete_response=complete_response,
                thread_id=langgraph_thread,
            )

            # Add bot-complete reactions after streaming completes
            bot_complete_reactions = [r for r in bot_reactions if r.get("target") == "bot" and r.get("when") == "complete"]
            for reaction in bot_complete_reactions:
                await self._add_reaction(context.channel_id, stream_ts, reaction.get("emoji"))

            logger.info(f"Completed streaming for thread {langgraph_thread}")

            return stream_ts, langgraph_thread, run_id

        finally:
            # Remove all non-persistent bot reactions
            for reaction in bot_reactions:
                if reaction.get("target") == "bot" and not reaction.get("persist", False):
                    await self._remove_reaction(context.channel_id, stream_ts, reaction.get("emoji"))

    async def _stream_from_langgraph_to_slack(
        self,
        message: str,
        langgraph_thread: str,
        slack_channel: str,
        slack_stream_ts: str,
        context: MessageContext,
    ) -> tuple[str, Optional[str]]:
        """Stream from LangGraph to Slack with immediate forwarding.

        THIS IS THE CRITICAL LOW-LATENCY PART:
        - Iterate over LangGraph stream
        - Each chunk arrives → immediately send to Slack
        - No buffering, no waiting

        Args:
            message: Transformed message to send to LangGraph
            langgraph_thread: LangGraph thread ID
            slack_channel: Slack channel ID
            slack_stream_ts: Slack stream timestamp
            context: Message context for output transforms

        Returns:
            Tuple of (complete_response, run_id)
        """
        complete_response = ""
        chunk_count = 0
        run_id = None

        # Build metadata if builder is provided
        metadata = await self.metadata_builder(context) if self.metadata_builder else {}

        try:
            # Start streaming from LangGraph
            # stream_mode="messages-tuple" gives us incremental message updates as tuples
            async for chunk in self.langgraph_client.runs.stream(
                thread_id=langgraph_thread,
                assistant_id=self.assistant_id,
                input={
                    "messages": [{"role": "user", "content": message}]
                },
                stream_mode=["messages-tuple"],
                multitask_strategy="interrupt",
                if_not_exists="create",
                metadata=metadata,
            ):
                chunk_count += 1
                logger.debug(f"Chunk #{chunk_count}: event={chunk.event}")

                # Capture run_id from metadata chunks (appears before message chunks)
                if run_id is None:
                    # Check if this is a metadata event with run_id
                    if hasattr(chunk, "data") and isinstance(chunk.data, dict) and "run_id" in chunk.data:
                        run_id = chunk.data["run_id"]
                        logger.info(f"Captured run_id: {run_id}")

                # Only process message chunks (skip metadata/other events)
                if chunk.event != "messages":
                    logger.debug(f"Chunk #{chunk_count}: skipping non-message event")
                    continue

                # Extract message data from chunk - it's a tuple!
                message_data, _msg_metadata = chunk.data
                msg_type = message_data.get("type", "")
                logger.debug(f"Chunk #{chunk_count}: message type={msg_type}")

                # Skip messages not in the configured message_types list
                if msg_type not in self.message_types:
                    logger.debug(f"Chunk #{chunk_count}: skipping message type '{msg_type}' (not in {self.message_types})")
                    continue

                # Get content from the chunk
                content = message_data.get("content", "")
                logger.debug(f"Chunk #{chunk_count}: content preview={str(content)[:100]}")

                if not content:
                    logger.debug(f"Chunk #{chunk_count}: no content")
                    continue

                # Handle both string and list content
                if isinstance(content, list):
                    content = "".join([block.get("text", "") for block in content if block.get("type") == "text"])
                    logger.debug(f"Chunk #{chunk_count}: extracted from list, length={len(content)}")

                # Skip empty content
                if not content.strip():
                    logger.debug(f"Chunk #{chunk_count}: content is empty after strip")
                    continue

                # Track complete response for image extraction
                # IMPORTANT: Accumulate chunks, don't replace!
                complete_response += content
                logger.debug(f"Chunk #{chunk_count}: sending {len(content)} chars to Slack (total accumulated: {len(complete_response)})")

                # CRITICAL: Immediately send to Slack
                # This is where low latency happens - no waiting!
                await self._append_to_slack_stream(
                    channel_id=slack_channel,
                    stream_ts=slack_stream_ts,
                    content=content,
                )

            logger.info(f"Stream completed: {chunk_count} chunks, {len(complete_response)} chars")

        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            # Try to append error message to stream
            try:
                await self._append_to_slack_stream(
                    channel_id=slack_channel,
                    stream_ts=slack_stream_ts,
                    content="\n\n_Error: Unable to complete response_",
                )
            except:
                pass  # Best effort

        # Apply output transformers to complete response
        # Note: We transform the complete response, not individual chunks
        # This ensures transformers see the full context
        if complete_response:
            logger.debug(f"Applying output transforms to {len(complete_response)} chars")
            complete_response = await self.output_transformers.apply(
                complete_response,
                context
            )
            logger.debug(f"After transforms: {len(complete_response)} chars")

        if run_id:
            logger.info(f"Returning complete response with run_id: {run_id}")
        else:
            logger.warning("No run_id captured during streaming!")

        return complete_response, run_id

    async def _start_slack_stream(
        self,
        channel_id: str,
        thread_ts: Optional[str],
        user_id: str,
        team_id: str,
    ) -> str:
        """Start a Slack stream.

        Args:
            channel_id: Slack channel ID
            thread_ts: Slack thread timestamp (None if not in thread)
            user_id: Slack user ID (recipient)
            team_id: Slack team/workspace ID

        Returns:
            Stream timestamp (message_ts) for subsequent operations

        Raises:
            Exception: If stream start fails
        """
        try:
            response = await self.slack_client.client.chat_startStream(
                channel=channel_id,
                recipient_team_id=team_id,
                recipient_user_id=user_id,
                thread_ts=thread_ts,
            )
            return response["ts"]

        except Exception as e:
            logger.error(f"Failed to start Slack stream: {e}", exc_info=True)
            raise

    async def _append_to_slack_stream(
        self,
        channel_id: str,
        stream_ts: str,
        content: str,
    ) -> None:
        """Append content to an active Slack stream.

        Args:
            channel_id: Slack channel ID
            stream_ts: Stream timestamp
            content: Content to append (will be converted to Slack markdown)
        """
        try:
            # Clean markdown for Slack format
            slack_content = clean_markdown(content)

            # Append to stream
            await self.slack_client.client.chat_appendStream(
                channel=channel_id,
                ts=stream_ts,
                markdown_text=slack_content,
            )

        except Exception as e:
            # Log but don't raise - we want to continue streaming
            logger.warning(f"Failed to append to stream: {e}")

    async def _stop_slack_stream(
        self,
        channel_id: str,
        stream_ts: str,
        complete_response: str,
        thread_id: str = None,
    ) -> None:
        """Stop Slack stream and add optional blocks (images, buttons).

        The text has already been streamed, so we only add image blocks (if extract_images=True)
        and feedback/thread_id blocks.

        Note: Slack's chat.stopStream doesn't support blocks in threads, so we:
        1. Stop the stream without blocks
        2. Update the message with blocks using chat.update

        Args:
            channel_id: Slack channel ID
            stream_ts: Stream timestamp
            complete_response: Complete accumulated response (for image extraction)
            thread_id: Optional LangGraph thread ID to include in feedback footer
        """
        try:
            # Create blocks (images + feedback, NO text block since we already streamed it)
            blocks = self._create_blocks(complete_response, thread_id)

            # Stop the stream without blocks
            await self.slack_client.client.chat_stopStream(
                channel=channel_id,
                ts=stream_ts,
            )

            logger.debug("Stream stopped")

            # If we have blocks to add, update the message
            if blocks:
                # Small delay to ensure Slack has processed the stream stop
                await asyncio.sleep(0.5)

                # Remove image markdown from text (since we're showing them as image blocks)
                import re
                text_without_images = re.sub(r"!\[([^\]]*)\]\(.+?\)", "", complete_response)

                # Get the current message text for the fallback
                slack_text = clean_markdown(text_without_images)

                # Create a text section block to preserve the streamed content
                text_block = {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": slack_text
                    }
                }

                # Prepend text block to preserve streamed content, then add images + feedback
                all_blocks = [text_block] + blocks

                try:
                    # Update the message with text + blocks (images + feedback)
                    await self.slack_client.client.chat_update(
                        channel=channel_id,
                        ts=stream_ts,
                        text=slack_text,  # Fallback text for notifications
                        blocks=all_blocks,
                    )

                    logger.info(f"Added {len(all_blocks)} blocks to message")

                except Exception as block_error:
                    # If updating with blocks fails (e.g., image download issues),
                    # fall back to just feedback blocks without images
                    logger.warning(f"Failed to add blocks: {block_error}")
                    logger.debug("Retrying with feedback blocks only (no images)")

                    # Get only feedback blocks (no image blocks)
                    from ..utils import create_feedback_block
                    feedback_only_blocks = create_feedback_block(
                        thread_id=thread_id,
                        show_feedback_buttons=self.show_feedback_buttons,
                        show_thread_id=self.show_thread_id,
                    )

                    # Add text block back
                    fallback_blocks = [text_block] + feedback_only_blocks

                    # Try again with just text + feedback
                    try:
                        await self.slack_client.client.chat_update(
                            channel=channel_id,
                            ts=stream_ts,
                            text=slack_text,
                            blocks=fallback_blocks,
                        )
                        logger.info("Added feedback blocks only")
                    except Exception as fallback_error:
                        logger.error(f"Failed even with feedback-only blocks: {fallback_error}")

        except Exception as e:
            logger.error(f"Failed to stop stream: {e}", exc_info=True)

    async def _get_team_id(self) -> str:
        """Get Slack team/workspace ID.

        Returns:
            Team ID string

        Raises:
            Exception: If auth test fails
        """
        try:
            auth_info = await self.slack_client.client.auth_test()
            return auth_info["team_id"]

        except Exception as e:
            logger.error(f"Failed to get team ID: {e}", exc_info=True)
            raise

    async def _add_reaction(
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
            await self.slack_client.client.reactions_add(
                channel=channel_id,
                timestamp=message_ts,
                name=emoji,
            )
            logger.debug(f"Added reaction :{emoji}: to message {channel_id}:{message_ts}")
        except Exception as e:
            logger.warning(f"Failed to add reaction :{emoji}:: {e}")

    async def _remove_reaction(
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
            await self.slack_client.client.reactions_remove(
                channel=channel_id,
                timestamp=message_ts,
                name=emoji,
            )
            logger.debug(f"Removed reaction :{emoji}: from message {channel_id}:{message_ts}")
        except Exception as e:
            logger.warning(f"Failed to remove reaction :{emoji}:: {e}")
