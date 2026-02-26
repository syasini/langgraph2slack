"""Message handling logic for non-streaming communication.

This module handles synchronous message processing where we wait for the
complete LangGraph response before sending to Slack.
"""

import logging
from typing import Dict, List, Optional, Tuple

from ..config import MessageContext
from ..tool_calls import ActiveToolCall
from ..transformers import TransformerChain
from ..utils import clean_markdown, create_plan_block, create_tool_details_button
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
        slack_client=None,
        reply_in_thread: bool = True,
        show_tool_calls: bool = False,
        show_tool_call_details: bool = True,
        tool_call_store: Optional[Dict] = None,
        structured_output: bool = False,
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
            slack_client: Slack Bolt AsyncApp (required when show_tool_calls=True)
            reply_in_thread: Reply in thread vs main channel (default: True)
            show_tool_calls: Show plan block with tool call status (default: False)
            show_tool_call_details: Show truncated input/output + View Details button (default: True)
            tool_call_store: Shared dict mapping plan_ts -> list[ActiveToolCall]
            structured_output: Pass full LangGraph state dict to output transformers (default: False)
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
            structured_output=structured_output,
        )
        # Store handler-specific attributes
        self.client = langgraph_client
        self.metadata_builder = metadata_builder
        self.slack_client = slack_client
        self.reply_in_thread = reply_in_thread

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

        # Step 4: Post plan block with tool call details (if enabled)
        # This goes BEFORE the text response so users see what the agent did first.
        if self.show_tool_calls and self.slack_client:
            slack_thread_ts = (
                context.thread_ts or context.message_ts
                if self.reply_in_thread
                else context.thread_ts
            )
            await self._post_tool_calls_plan(
                langgraph_response, context.channel_id, slack_thread_ts, run_id
            )

        # Step 5: Choose transformer input based on structured_output flag
        if self.structured_output:
            # Pass the full LangGraph state dict so transformers can access
            # custom state keys (e.g. image_url, table) beyond the last message.
            transformer_input = langgraph_response
            logger.debug("structured_output=True: passing full state dict to output transformers")
        else:
            # Default: extract last AI message text (existing behaviour)
            transformer_input = self._extract_message_content(langgraph_response)
            logger.debug(f"LangGraph response: {str(transformer_input)[:100]}...")

        # Step 6: Apply output transformers
        # Each transformer can modify the response (add footer, filter content, etc.)
        # or return a list of Slack block dicts for fully custom layouts.
        transformed_output = await self.output_transformers.apply(transformer_input, context)

        # Step 7: Branch on whether the transformer returned blocks or text
        if isinstance(transformed_output, list):
            # Transformer returned custom Slack blocks — use them directly.
            # Extract text from last message for Slack's notification preview.
            fallback_text = self._extract_text_fallback(langgraph_response)
            slack_formatted = clean_markdown(fallback_text)
            blocks = self._create_blocks(fallback_text, langgraph_thread, custom_blocks=transformed_output)
            logger.info(
                f"Output transformer returned {len(transformed_output)} custom blocks"
            )
        else:
            # Transformer returned a string — standard text + image extraction path.
            slack_formatted = clean_markdown(transformed_output)
            blocks = self._create_blocks(transformed_output, langgraph_thread)

        return slack_formatted, blocks, langgraph_thread, run_id

    def _extract_tool_calls_from_messages(
        self, langgraph_response: dict
    ) -> List[ActiveToolCall]:
        """Extract tool calls and their results from a completed LangGraph run.

        Scans the full message list from the completed run to find:
        - AIMessages with non-empty tool_calls (each call gets an ActiveToolCall)
        - ToolMessages with matching tool_call_id (provides the result)

        Args:
            langgraph_response: Full response dict from LangGraph run (runs.join() result)

        Returns:
            List of ActiveToolCall instances with complete data (name, args, result, status)
        """
        messages = langgraph_response.get("messages", [])
        if not messages:
            return []

        # Build index of ToolMessage results keyed by tool_call_id
        tool_results: Dict[str, str] = {}
        for msg in messages:
            if msg.get("type") == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    )
                if tool_call_id:
                    tool_results[tool_call_id] = str(content)

        # Build ActiveToolCall list from AIMessages with tool_calls
        calls: List[ActiveToolCall] = []
        for msg in messages:
            if msg.get("type") not in ("ai", "AIMessage"):
                continue
            for tc in msg.get("tool_calls", []):
                call_id = tc.get("id", f"call_{len(calls)}")
                name = tc.get("name", "unknown_tool")
                # args can be a dict or a JSON string
                args = tc.get("args", {})
                if isinstance(args, dict):
                    import json
                    args_str = json.dumps(args, indent=2)
                else:
                    args_str = str(args)

                result = tool_results.get(call_id)
                call = ActiveToolCall(
                    call_id=call_id,
                    name=name,
                    args_chunks=[args_str],
                    status="complete" if result is not None else "in_progress",
                    result=result,
                )
                calls.append(call)

        return calls

    async def _post_tool_calls_plan(
        self,
        langgraph_response: dict,
        channel_id: str,
        thread_ts: Optional[str],
        run_id: Optional[str],
    ) -> None:
        """Extract tool calls from the completed run and post a plan block to Slack.

        Args:
            langgraph_response: Full response dict from LangGraph run
            channel_id: Slack channel ID
            thread_ts: Slack thread timestamp (plan block goes in the same thread)
            run_id: Optional LangGraph run ID for LangSmith source links
        """
        calls = self._extract_tool_calls_from_messages(langgraph_response)
        if not calls:
            return

        title = "Done"
        plan_block = create_plan_block(calls, self.show_tool_call_details, run_id, title)
        blocks = [plan_block]
        if self.show_tool_call_details:
            # Button value will be set after we know plan_ts
            # Post first, then update with the correct plan_ts button value
            try:
                response = await self.slack_client.client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=title,
                    blocks=blocks,
                )
                plan_ts = response.get("ts")
                if plan_ts:
                    self.tool_call_store[plan_ts] = calls
                    # Update with the View Details button (now we know plan_ts)
                    blocks_with_button = [plan_block, create_tool_details_button(plan_ts)]
                    await self.slack_client.client.chat_update(
                        channel=channel_id,
                        ts=plan_ts,
                        text=title,
                        blocks=blocks_with_button,
                    )
            except Exception as e:
                logger.warning(f"Failed to post/update tool call plan block: {e}")
        else:
            # No details button needed - single post
            try:
                await self.slack_client.client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=title,
                    blocks=blocks,
                )
            except Exception as e:
                logger.warning(f"Failed to post tool call plan block: {e}")

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
