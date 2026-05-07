"""Streaming response handling for low-latency communication.

This module implements true streaming where LangGraph chunks are immediately
forwarded to Slack as they arrive, minimizing latency.
"""

import asyncio
import logging
import re
import time
from typing import Optional

from ..config import MessageContext
from ..transformers import TransformerChain
from ..tool_calls import ToolCallTracker
from ..utils import (
    clean_markdown,
    create_markdown_text_block,
    create_mrkdwn_text_block,
    create_plan_block,
    create_tool_details_button,
    replace_markdown_images_with_links,
)
from ..mixins import ReactionMixin
from .base import BaseHandler

logger = logging.getLogger(__name__)


def _is_image_block_error(error: Exception) -> bool:
    """Return True when Slack rejected an image block download/render."""
    error_str = str(error).lower()
    return "downloading image" in error_str or (
        "invalid_blocks" in error_str and "image_url" in error_str
    )


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
        config_builder=None,
        message_types: list[str] = None,
        stream_buffer_time: float = 0.1,
        stream_buffer_max_chunks: int = 10,
        show_tool_calls: bool = False,
        show_tool_call_details: bool = True,
        tool_call_store: Optional[dict] = None,
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
            config_builder: Async function to build config dict from MessageContext
            message_types: List of message types to process (default: ["AIMessageChunk"])
            stream_buffer_time: Time in seconds to buffer chunks before flushing (default: 0.1).
                Buffer flushes when EITHER this time elapses OR stream_buffer_max_chunks is reached.
            stream_buffer_max_chunks: Maximum chunks to buffer before force-flushing (default: 10).
                Buffer flushes when EITHER stream_buffer_time elapses OR this limit is reached.
            show_tool_calls: Show live plan block with tool call status (default: False)
            show_tool_call_details: Show truncated input/output + View Details button (default: True)
            tool_call_store: Shared dict (from SlackBot) mapping plan_ts -> list[ActiveToolCall]
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
            show_tool_calls=show_tool_calls,
            show_tool_call_details=show_tool_call_details,
            tool_call_store=tool_call_store,
        )
        # Store handler-specific attributes
        self.langgraph_client = langgraph_client
        self.slack_client = slack_client
        self.reply_in_thread = reply_in_thread
        self.metadata_builder = metadata_builder
        self.config_builder = config_builder
        self.message_types = message_types if message_types is not None else ["AIMessageChunk"]

        # Streaming buffer configuration
        self.stream_buffer_time = stream_buffer_time
        self.stream_buffer_max_chunks = stream_buffer_max_chunks

        # Slack team_id cache (lazy initialization)
        self._team_id: Optional[str] = None
        self._team_id_lock = asyncio.Lock()
        self._team_id_initialized = False

        # Background task tracking for proper cleanup
        self._background_tasks: set[asyncio.Task] = set()

        # Initialize reaction mixin for shared reaction operations
        self._reactions = ReactionMixin(self.slack_client.client)

    def _create_background_task(self, coro) -> asyncio.Task:
        """Create a background task with automatic cleanup.

        Tracks the task in _background_tasks set and automatically removes it
        when done. This ensures proper lifecycle management and prevents memory leaks.

        Args:
            coro: Coroutine to run as background task

        Returns:
            Created asyncio.Task
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def cleanup(self) -> None:
        """Cancel all background tasks and wait for cleanup.

        Call this during graceful shutdown to ensure all background operations
        complete or are properly cancelled.
        """
        if self._background_tasks:
            logger.info(f"Cancelling {len(self._background_tasks)} background tasks...")
            for task in self._background_tasks:
                task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            logger.info("All background tasks cancelled")

    async def process_message(
        self,
        message: str,
        context: MessageContext,
        bot_reactions: list[dict] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Process message with streaming.

        Main entry point for streaming message processing.
        Sends response directly to Slack via streaming.

        Unlike ``MessageHandler.process_message()`` which returns (text, blocks, ...)
        for the caller to send, this handler manages the full Slack lifecycle internally
        (start stream → append chunks → stop stream → update with blocks). It returns
        only identifiers needed for feedback tracking.

        Args:
            message: Raw message text from Slack
            context: Message context with user/channel info
            bot_reactions: List of reaction configs for bot message (default: None)

        Returns:
            Tuple of (stream_ts, thread_id, run_id) for feedback tracking
        """
        if bot_reactions is None:
            bot_reactions = []
        logger.info(
            f"Streaming message from user {context.user_id} in channel {context.channel_id}"
        )

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

        # Start bot-processing reactions in background (don't block LangGraph)
        bot_processing_reactions = [
            r for r in bot_reactions if r.get("target") == "bot" and r.get("when") == "processing"
        ]
        if bot_processing_reactions:
            self._create_background_task(
                self._reactions.add_parallel(
                    bot_processing_reactions, context.channel_id, stream_ts
                )
            )

        try:
            # Step 6: Stream from LangGraph and forward to Slack
            # CRITICAL: Each chunk is sent immediately as it arrives
            # transformed: output of transformers (str or list[dict] blocks)
            # complete_response: raw accumulated text (for fallback/image extraction)
            transformed, complete_response, run_id = await self._stream_from_langgraph_to_slack(
                message=transformed_input,
                langgraph_thread=langgraph_thread,
                slack_channel=context.channel_id,
                slack_stream_ts=stream_ts,
                context=context,
                slack_thread_ts=slack_thread_ts,
            )

            # Step 7: Stop stream with optional image/custom blocks
            await self._stop_slack_stream(
                channel_id=context.channel_id,
                stream_ts=stream_ts,
                complete_response=complete_response,
                transformed=transformed,
                thread_id=langgraph_thread,
                slack_thread_ts=slack_thread_ts,
            )

            # Add bot-complete reactions after streaming completes (parallel)
            bot_complete_reactions = [
                r for r in bot_reactions if r.get("target") == "bot" and r.get("when") == "complete"
            ]
            if bot_complete_reactions:
                await self._reactions.add_parallel(
                    bot_complete_reactions, context.channel_id, stream_ts
                )

            logger.info(f"Completed streaming for thread {langgraph_thread}")

            return stream_ts, langgraph_thread, run_id

        finally:
            # Remove all non-persistent bot reactions
            for reaction in bot_reactions:
                if reaction.get("target") == "bot" and not reaction.get("persist", False):
                    await self._reactions.remove(
                        context.channel_id, stream_ts, reaction.get("emoji")
                    )

    def _should_flush_buffer(
        self,
        buffer: list[str],
        last_flush_time: float,
    ) -> tuple[bool, Optional[str]]:
        """Determine if buffer should be flushed to Slack.

        Flushes when EITHER condition is met:
        1. Enough time has elapsed since last flush (stream_buffer_time)
        2. Buffer has accumulated too many chunks (stream_buffer_max_chunks)

        These work together as safety limits - whichever is reached first triggers the flush.
        Typical behavior: time limit triggers for normal streaming, chunk limit catches edge cases.

        Args:
            buffer: List of content chunks accumulated
            last_flush_time: Timestamp of last flush

        Returns:
            Tuple of (should_flush, reason):
                - should_flush: True if buffer should be flushed
                - reason: Human-readable reason for flushing (for logging), or None if no flush
        """
        if not buffer:
            return False, None

        # Calculate time elapsed
        time_since_flush = time.time() - last_flush_time

        # Check flush conditions (time OR chunk count)
        if time_since_flush >= self.stream_buffer_time:
            return True, f"time limit ({time_since_flush:.2f}s elapsed)"
        elif len(buffer) >= self.stream_buffer_max_chunks:
            return True, f"chunk limit ({len(buffer)} chunks)"

        return False, None

    async def _flush_buffer(
        self,
        buffer: list[str],
        channel_id: str,
        stream_ts: str,
    ) -> None:
        """Flush accumulated buffer to Slack stream.

        Args:
            buffer: List of content chunks to flush
            channel_id: Slack channel ID
            stream_ts: Stream timestamp
        """
        if not buffer:
            return

        # Combine all buffered chunks
        combined_content = "".join(buffer)

        # Send to Slack
        await self._append_to_slack_stream(
            channel_id=channel_id,
            stream_ts=stream_ts,
            content=combined_content,
        )

        # Clear buffer
        buffer.clear()

    async def _flush_buffer_and_return_time(
        self,
        buffer: list[str],
        channel_id: str,
        stream_ts: str,
    ) -> float:
        """Flush buffer and return current time.

        This helper ensures that the flush time is always updated correctly,
        making it impossible to forget updating last_flush_time after a flush.

        Args:
            buffer: List of content chunks to flush
            channel_id: Slack channel ID
            stream_ts: Stream timestamp

        Returns:
            Current timestamp after flush
        """
        await self._flush_buffer(buffer, channel_id, stream_ts)
        return time.time()

    async def _post_plan_block(
        self,
        channel_id: str,
        thread_ts: Optional[str],
        tracker: "ToolCallTracker",
        run_id: Optional[str],
    ) -> Optional[str]:
        """Post an initial plan block message showing tool call status.

        Args:
            channel_id: Slack channel ID
            thread_ts: Slack thread timestamp (plan block goes in the same thread as the response)
            tracker: ToolCallTracker with current tool call state
            run_id: Optional LangGraph run ID for LangSmith source links

        Returns:
            Slack message timestamp (plan_ts) or None if posting failed
        """
        try:
            calls = tracker.get_calls()
            plan_block = create_plan_block(
                calls, self.show_tool_call_details, run_id, "Working on it..."
            )
            response = await self.slack_client.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text="Working on it...",
                blocks=[plan_block],
            )
            plan_ts = response.get("ts")
            if plan_ts:
                self.tool_call_store[plan_ts] = calls
            return plan_ts
        except Exception as e:
            logger.error(f"Failed to post plan block: {e}", exc_info=True)
            return None

    async def _update_plan_block(
        self,
        channel_id: str,
        plan_ts: str,
        tracker: "ToolCallTracker",
        run_id: Optional[str],
    ) -> None:
        """Update the plan block message with current tool call status.

        Args:
            channel_id: Slack channel ID
            plan_ts: Slack message timestamp of the existing plan block
            tracker: ToolCallTracker with current tool call state
            run_id: Optional LangGraph run ID for LangSmith source links
        """
        try:
            calls = tracker.get_calls()
            plan_block = create_plan_block(
                calls, self.show_tool_call_details, run_id, "Working on it..."
            )
            await self.slack_client.client.chat_update(
                channel=channel_id,
                ts=plan_ts,
                text="Working on it...",
                blocks=[plan_block],
            )
            self.tool_call_store[plan_ts] = calls
        except Exception as e:
            logger.warning(f"Failed to update plan block: {e}")

    async def _finalize_plan_block(
        self,
        channel_id: str,
        plan_ts: str,
        tracker: "ToolCallTracker",
        run_id: Optional[str],
    ) -> None:
        """Do final plan block update: change title to 'Done' and add View Details button.

        Called after all streaming is complete. Adds the 'View Full Details' button
        that lets users open a modal with untruncated tool call inputs and outputs.

        Args:
            channel_id: Slack channel ID
            plan_ts: Slack message timestamp of the existing plan block
            tracker: ToolCallTracker with final tool call state
            run_id: Optional LangGraph run ID for LangSmith source links
        """
        try:
            calls = tracker.get_calls()
            plan_block = create_plan_block(calls, self.show_tool_call_details, run_id, "Done")
            blocks = [plan_block]
            if self.show_tool_call_details:
                blocks.append(create_tool_details_button(plan_ts))
            await self.slack_client.client.chat_update(
                channel=channel_id,
                ts=plan_ts,
                text="Done",
                blocks=blocks,
            )
            self.tool_call_store[plan_ts] = calls
        except Exception as e:
            logger.warning(f"Failed to finalize plan block: {e}")

    async def _stream_from_langgraph_to_slack(
        self,
        message: str,
        langgraph_thread: str,
        slack_channel: str,
        slack_stream_ts: str,
        context: MessageContext,
        slack_thread_ts: Optional[str] = None,
    ) -> tuple:
        """Stream from LangGraph to Slack with buffered forwarding.

        Implements time-based buffering to reduce API call overhead:
        - Accumulate chunks in buffer
        - Flush based on time, size, or chunk count
        - Reduces API calls by 5-10x for typical responses

        Args:
            message: Transformed message to send to LangGraph
            langgraph_thread: LangGraph thread ID
            slack_channel: Slack channel ID
            slack_stream_ts: Slack stream timestamp
            context: Message context for output transforms

        Returns:
            Tuple of (transformed, complete_response, run_id)
            - transformed: Transformer output — str (possibly modified text) or list[dict] (custom blocks)
            - complete_response: Raw accumulated text from the stream (used as fallback)
            - run_id: LangGraph run ID, or None if not captured
        """
        complete_response = ""
        chunk_count = 0
        run_id = None

        # Tool call tracking (only active when show_tool_calls=True)
        tracker = ToolCallTracker() if self.show_tool_calls else None
        plan_ts: Optional[str] = None
        if self.show_tool_calls:
            logger.info(
                f"Tool call tracking enabled (thread_ts={slack_thread_ts}, details={self.show_tool_call_details})"
            )

        # Initialize buffer for time-based flushing
        buffer = []
        last_flush_time = time.time()

        # Build metadata if builder is provided
        metadata = await self.metadata_builder(context) if self.metadata_builder else {}

        # Build config if builder is provided
        config = await self.config_builder(context) if self.config_builder else None

        try:
            # Start streaming from LangGraph
            # stream_mode="messages-tuple" gives us incremental message updates as tuples
            async for chunk in self.langgraph_client.runs.stream(
                thread_id=langgraph_thread,
                assistant_id=self.assistant_id,
                input={"messages": [{"role": "user", "content": message}]},
                stream_mode=["messages-tuple"],
                multitask_strategy="interrupt",
                if_not_exists="create",
                metadata=metadata,
                config=config,
            ):
                chunk_count += 1
                logger.debug(f"Chunk #{chunk_count}: event={chunk.event}")

                # Capture run_id from metadata chunks (appears before message chunks)
                if run_id is None:
                    # Check if this is a metadata event with run_id
                    if (
                        hasattr(chunk, "data")
                        and isinstance(chunk.data, dict)
                        and "run_id" in chunk.data
                    ):
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
                if tracker is not None and msg_type not in ("AIMessageChunk",):
                    logger.warning(
                        f"[show_tool_calls] Non-streaming msg type: {msg_type!r} "
                        f"keys={list(message_data.keys())} "
                        f"tool_calls={bool(message_data.get('tool_calls'))} "
                        f"tool_call_id={message_data.get('tool_call_id')!r}"
                    )

                # Tool call detection (runs before message type filter so tool messages are handled
                # even when "tool" is not in self.message_types).
                # Handles two cases:
                # 1. Streaming: AIMessageChunk with tool_call_chunks (incremental JSON fragments)
                # 2. Complete: "ai"/"AIMessage" with tool_calls dict (ReAct agent pattern)
                if tracker is not None:
                    if msg_type == "AIMessageChunk":
                        tc_chunks = message_data.get("tool_call_chunks", [])
                        if tc_chunks:
                            logger.info(
                                f"Tool call chunks detected: {[tc.get('name') or tc.get('id') or '...' for tc in tc_chunks]}"
                            )
                            had_new_call = False
                            for tc_chunk in tc_chunks:
                                if tracker.handle_chunk(tc_chunk):
                                    had_new_call = True
                            if had_new_call and tracker.has_calls():
                                if plan_ts is None:
                                    plan_ts = await self._post_plan_block(
                                        slack_channel, slack_thread_ts, tracker, run_id
                                    )
                                else:
                                    await self._update_plan_block(
                                        slack_channel, plan_ts, tracker, run_id
                                    )

                    elif msg_type in ("ai", "AIMessage"):
                        # Complete AIMessage from ReAct agent node - tool call decision arrived
                        # as a full message (not streaming chunks). This is the typical pattern
                        # with create_react_agent / create_agent.
                        import json as _json
                        tool_calls = message_data.get("tool_calls", [])
                        if tool_calls:
                            logger.info(
                                f"Tool calls detected in AIMessage: {[tc.get('name') for tc in tool_calls]}"
                            )
                            had_new_call = False
                            for idx, tc in enumerate(tool_calls):
                                call_id = tc.get("id") or f"call_{idx}"
                                name = tc.get("name", "unknown_tool")
                                args = tc.get("args", {})
                                args_str = (
                                    _json.dumps(args) if isinstance(args, dict) else str(args)
                                )
                                # Feed as a single complete chunk
                                is_new = tracker.handle_chunk(
                                    {"id": call_id, "name": name, "args": args_str, "index": idx}
                                )
                                if is_new:
                                    had_new_call = True
                            if had_new_call and tracker.has_calls():
                                if plan_ts is None:
                                    plan_ts = await self._post_plan_block(
                                        slack_channel, slack_thread_ts, tracker, run_id
                                    )
                                else:
                                    await self._update_plan_block(
                                        slack_channel, plan_ts, tracker, run_id
                                    )

                    elif msg_type == "tool":
                        tool_call_id = message_data.get("tool_call_id", "")
                        tool_name = message_data.get("name", "")
                        result = message_data.get("content", "")
                        if isinstance(result, list):
                            result = " ".join(
                                b.get("text", "") for b in result if isinstance(b, dict)
                            )
                        logger.info(
                            f"Tool result received for call_id={tool_call_id!r} name={tool_name!r}: {str(result)[:80]}"
                        )

                        # Retroactive call entry: if we never saw the tool call start
                        # (e.g. call start arrived as an unhandled message type), synthesize
                        # an entry now so the plan block can still be shown.
                        if tool_call_id and tool_call_id not in tracker.calls:
                            logger.warning(
                                f"Tool result for {tool_call_id!r} arrived but call was never tracked — "
                                "retroactively adding call entry (tool call start may have had unexpected format)"
                            )
                            tracker.handle_chunk(
                                {
                                    "id": tool_call_id,
                                    "name": tool_name or tool_call_id,
                                    "args": "",
                                    "index": len(tracker.calls),
                                }
                            )

                        tracker.handle_result(tool_call_id, str(result))

                        if plan_ts:
                            await self._update_plan_block(
                                slack_channel, plan_ts, tracker, run_id
                            )
                        else:
                            # Plan block was never posted — post it now retroactively.
                            # This handles the case where the tool call start event was
                            # missed (unexpected message format) but we still want to
                            # surface the tool call in Slack.
                            logger.warning(
                                "Tool result received but plan block not yet posted — posting retroactively"
                            )
                            plan_ts = await self._post_plan_block(
                                slack_channel, slack_thread_ts, tracker, run_id
                            )

                # Skip messages not in the configured message_types list
                if msg_type not in self.message_types:
                    logger.debug(
                        f"Chunk #{chunk_count}: skipping message type '{msg_type}' (not in {self.message_types})"
                    )
                    continue

                # Get content from the chunk
                content = message_data.get("content", "")
                logger.debug(f"Chunk #{chunk_count}: content preview={str(content)[:100]}")

                if not content:
                    logger.debug(f"Chunk #{chunk_count}: no content")
                    continue

                # Handle both string and list content
                if isinstance(content, list):
                    content = "".join(
                        [block.get("text", "") for block in content if block.get("type") == "text"]
                    )
                    logger.debug(
                        f"Chunk #{chunk_count}: extracted from list, length={len(content)}"
                    )

                # Skip empty content
                if not content.strip():
                    logger.debug(f"Chunk #{chunk_count}: content is empty after strip")
                    continue

                # Track complete response for image extraction
                # IMPORTANT: Accumulate chunks, don't replace!
                complete_response += content
                logger.debug(
                    f"Chunk #{chunk_count}: buffering {len(content)} chars (total accumulated: {len(complete_response)})"
                )

                # Add to buffer
                buffer.append(content)

                # Check if we should flush buffer
                should_flush, reason = self._should_flush_buffer(buffer, last_flush_time)
                if should_flush:
                    logger.debug(f"Flushing buffer: {len(buffer)} chunks (reason: {reason})")
                    last_flush_time = await self._flush_buffer_and_return_time(
                        buffer, slack_channel, slack_stream_ts
                    )

            # Final flush for any remaining content
            if buffer:
                logger.debug(f"Final flush: {len(buffer)} chunks remaining")
                await self._flush_buffer(buffer, slack_channel, slack_stream_ts)

            logger.info(f"Stream completed: {chunk_count} chunks, {len(complete_response)} chars")
            if tracker is not None:
                if tracker.has_calls():
                    logger.info(
                        f"[show_tool_calls] Detected {len(tracker.calls)} tool call(s): "
                        f"{[c.name for c in tracker.get_calls()]}"
                    )
                else:
                    logger.warning(
                        "[show_tool_calls] Stream ended with NO tool calls detected. "
                        "If tools were called, the call start event had an unexpected format. "
                        "Check for non-AIMessageChunk/non-ai message types in the logs above."
                    )

        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            # Flush any buffered content before showing error
            try:
                if buffer:
                    await self._flush_buffer(buffer, slack_channel, slack_stream_ts)
            except:
                pass  # Best effort

            # Try to append error message to stream
            try:
                await self._append_to_slack_stream(
                    channel_id=slack_channel,
                    stream_ts=slack_stream_ts,
                    content="\n\n_Error: Unable to complete response_",
                )
            except:
                pass  # Best effort

        # Apply output transformers to the accumulated text response.
        # Note: We transform after streaming completes so transformers see the full context.
        if complete_response:
            transformed = await self.output_transformers.apply(complete_response, context)
        else:
            transformed = complete_response  # empty string passthrough

        if run_id:
            logger.info(f"Returning complete response with run_id: {run_id}")
        else:
            logger.warning("No run_id captured during streaming!")

        # Finalize plan block: change title to "Done" and add "View Full Details" button
        if plan_ts and tracker and tracker.has_calls():
            await self._finalize_plan_block(slack_channel, plan_ts, tracker, run_id)

        return transformed, complete_response, run_id

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

    async def _update_message_with_fallback(
        self,
        channel_id: str,
        stream_ts: str,
        primary_blocks: list,
        fallback_text: str,
        thread_id: str = None,
    ) -> None:
        """Try chat_update with primary blocks; on failure, degrade to text + feedback only.

        This centralizes the two-level fallback pattern used by both the custom-blocks
        and default paths in ``_stop_slack_stream()``.

        Args:
            channel_id: Slack channel ID
            stream_ts: Message timestamp to update
            primary_blocks: Blocks to attempt first (custom blocks or text + images + feedback)
            fallback_text: Notification/fallback text (mrkdwn-formatted)
            thread_id: LangGraph thread ID for feedback footer
        """
        try:
            await self.slack_client.client.chat_update(
                channel=channel_id,
                ts=stream_ts,
                text=fallback_text,
                blocks=primary_blocks,
            )
            logger.info(f"Updated message with {len(primary_blocks)} blocks")
        except Exception as block_error:
            logger.warning(f"Failed to update with primary blocks: {block_error}")
            logger.debug("Retrying with text + feedback blocks only")

            from ..utils import create_feedback_block

            feedback_only_blocks = create_feedback_block(
                thread_id=thread_id,
                show_feedback_buttons=self.show_feedback_buttons,
                show_thread_id=self.show_thread_id,
            )
            text_block = create_mrkdwn_text_block(fallback_text)
            fallback_blocks = [text_block] + feedback_only_blocks

            try:
                await self.slack_client.client.chat_update(
                    channel=channel_id,
                    ts=stream_ts,
                    text=fallback_text,
                    blocks=fallback_blocks,
                )
                logger.info("Fallback succeeded: sent text + feedback blocks only")
            except Exception as fallback_error:
                logger.error(f"Failed even with fallback blocks: {fallback_error}")

    async def _update_message_with_incremental_images(
        self,
        channel_id: str,
        stream_ts: str,
        text_block: dict,
        image_blocks: list,
        footer_blocks: list,
        fallback_text: str,
        thread_id: str = None,
    ) -> None:
        """Update safe text/footer first, then add image blocks one at a time.

        This performs one base ``chat.update`` plus one candidate update for
        each extracted image block. ``image_blocks`` is already bounded by
        ``max_image_blocks`` when default image extraction builds the block list.
        """
        base_blocks = [text_block] + footer_blocks
        await self._update_message_with_fallback(
            channel_id,
            stream_ts,
            base_blocks,
            fallback_text,
            thread_id,
        )

        accepted_image_blocks = []
        for image_block in image_blocks:
            candidate_blocks = (
                [text_block] + accepted_image_blocks + [image_block] + footer_blocks
            )
            try:
                await self.slack_client.client.chat_update(
                    channel=channel_id,
                    ts=stream_ts,
                    text=fallback_text,
                    blocks=candidate_blocks,
                )
                accepted_image_blocks.append(image_block)
                logger.info(
                    "Accepted image block: "
                    f"{image_block.get('image_url')}"
                )
            except Exception as image_error:
                if _is_image_block_error(image_error):
                    logger.warning(
                        "Skipping image block rejected by Slack: "
                        f"{image_block.get('image_url')} ({image_error})"
                    )
                    continue

                logger.warning(
                    "Stopping incremental image updates after "
                    f"unexpected Slack error: {image_error}"
                )
                break

    async def _stop_slack_stream(
        self,
        channel_id: str,
        stream_ts: str,
        complete_response: str,
        transformed=None,
        thread_id: str = None,
        slack_thread_ts: Optional[str] = None,
    ) -> None:
        """Stop Slack stream and add optional blocks (images, buttons).

        The text has already been streamed, so we only add image blocks (if extract_images=True)
        and feedback/thread_id blocks.

        When ``transformed`` is a list of dicts (custom blocks from a transformer), those
        blocks replace the default text-section + image blocks layout. The streamed text
        is replaced by the custom blocks via chat.update.

        When ``transformed`` is a list of lists (multi-message payload), the streamed message
        is updated with the first chunk and subsequent chunks are posted as separate messages
        in the same thread.

        Note: Slack's chat.stopStream doesn't support blocks in threads, so we:
        1. Stop the stream without blocks
        2. Update the message with blocks using chat.update

        Args:
            channel_id: Slack channel ID
            stream_ts: Stream timestamp
            complete_response: Raw accumulated text response (used for image extraction
                               and as fallback text when transformed is blocks)
            transformed: Transformer result — either a str (possibly modified text),
                         a list[dict] (custom Slack blocks), or a list[list[dict]]
                         (multi-message payload). Defaults to complete_response.
            thread_id: Optional LangGraph thread ID to include in feedback footer
            slack_thread_ts: Slack thread timestamp for posting follow-up messages
                             in multi-message mode
        """
        # Determine whether the transformer returned custom blocks, multi-message, or text
        if isinstance(transformed, list):
            if transformed and isinstance(transformed[0], list):
                # Multi-message payload: list[list[dict]]
                is_multi_message = True
                custom_blocks = None
            else:
                # Single-message custom blocks: list[dict]
                is_multi_message = False
                custom_blocks = transformed
        else:
            is_multi_message = False
            custom_blocks = None
            # If transformer returned a modified string, use that as the display text
            if isinstance(transformed, str) and transformed:
                display_text = transformed
            else:
                display_text = complete_response

        try:
            # Stop the stream first (blocks not supported in chat_stopStream for threads)
            await self.slack_client.client.chat_stopStream(
                channel=channel_id,
                ts=stream_ts,
            )
            logger.debug("Stream stopped")

            if is_multi_message:
                # ---------------------------------------------------------------
                # Multi-message path: transformer returned list[list[dict]]
                # Update stream message with first chunk; post the rest separately.
                # ---------------------------------------------------------------
                from ..utils import create_feedback_block

                feedback_blocks = create_feedback_block(
                    thread_id=thread_id,
                    show_feedback_buttons=self.show_feedback_buttons,
                    show_thread_id=self.show_thread_id,
                )
                messages = [list(chunk) for chunk in transformed]
                messages[-1] = messages[-1] + feedback_blocks

                fallback_text = clean_markdown(complete_response, for_blocks=True) or " "
                first_blocks, *rest_blocks = messages

                await asyncio.sleep(0.5)
                await self._update_message_with_fallback(
                    channel_id, stream_ts, first_blocks, fallback_text, thread_id
                )

                # Thread for follow-up messages: use existing thread or stream_ts as anchor
                reply_thread = slack_thread_ts or stream_ts
                for chunk_blocks in rest_blocks:
                    try:
                        await self.slack_client.client.chat_postMessage(
                            channel=channel_id,
                            thread_ts=reply_thread,
                            text=fallback_text,
                            blocks=chunk_blocks,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to post follow-up message chunk: {e}")

                logger.info(f"Multi-message: sent {len(messages)} messages")

            elif custom_blocks is not None:
                # ---------------------------------------------------------------
                # Custom blocks path: transformer returned list[dict]
                # The streamed text will be fully replaced by the custom blocks.
                # ---------------------------------------------------------------
                blocks = self._create_blocks(
                    complete_response, thread_id, custom_blocks=custom_blocks
                )

                if blocks:
                    await asyncio.sleep(0.5)
                    fallback_text = clean_markdown(complete_response, for_blocks=True) or " "
                    await self._update_message_with_fallback(
                        channel_id, stream_ts, blocks, fallback_text, thread_id
                    )

            else:
                # ---------------------------------------------------------------
                # Default path: use display_text for text section + image extraction
                # ---------------------------------------------------------------
                blocks = self._create_blocks(display_text, thread_id)

                if blocks:
                    await asyncio.sleep(0.5)

                    # Keep image source URLs visible in the text while also
                    # rendering Slack image blocks below it.
                    text_with_image_links = replace_markdown_images_with_links(
                        display_text
                    )
                    slack_text = clean_markdown(text_with_image_links, for_blocks=True)

                    # Prepend a standard Markdown block so Slack preserves headings,
                    # tables, task lists, dividers, and other Markdown constructs.
                    text_block = create_markdown_text_block(text_with_image_links)
                    image_blocks = [b for b in blocks if b.get("type") == "image"]
                    feedback_thread_blocks = [
                        b for b in blocks if b.get("type") != "image"
                    ]

                    await self._update_message_with_incremental_images(
                        channel_id,
                        stream_ts,
                        text_block,
                        image_blocks,
                        feedback_thread_blocks,
                        slack_text,
                        thread_id,
                    )

        except Exception as e:
            logger.error(f"Failed to stop stream: {e}", exc_info=True)

    async def _get_team_id(self) -> str:
        """Get Slack team/workspace ID (cached after first call).

        Uses lazy initialization with lock to prevent race conditions.
        Fetches team_id once on first call, then caches for all subsequent calls.
        This eliminates redundant auth_test() API calls on every message.

        Returns:
            Team ID string

        Raises:
            Exception: If auth test fails
        """
        if self._team_id_initialized:
            return self._team_id  # Return cached value

        async with self._team_id_lock:
            # Double-check after acquiring lock (another task might have initialized)
            if self._team_id_initialized:
                return self._team_id

            try:
                logger.info("Fetching Slack team_id via auth_test()...")
                auth_info = await self.slack_client.client.auth_test()
                self._team_id = auth_info["team_id"]
                self._team_id_initialized = True
                logger.info(f"Cached Slack team_id: {self._team_id}")
                return self._team_id

            except Exception as e:
                logger.error(f"Failed to get team ID: {e}", exc_info=True)
                raise
