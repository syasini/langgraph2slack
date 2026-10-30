"""Main SlackBot class that orchestrates everything.

This is the primary interface users interact with. It creates the FastAPI app,
sets up Slack handlers, and coordinates between Slack and LangGraph.
"""

import asyncio
import json
import logging
from typing import Callable, Optional

from fastapi import FastAPI, Request
from langgraph_sdk import get_client
from langsmith import Client
from pydantic import SecretStr
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.async_app import AsyncApp

from .backfill import BackfillCollector
from .config import BotConfig, MessageContext
from .framing import RequestedBy, TriggerInput, default_framer
from .handlers import MessageHandler, StreamingHandler
from .mixins import ReactionMixin
from .normalizer import NormalizedMessage, SlackMessageNormalizer
from .state import BridgeState, DeploymentStateStore, StateStore
from .transformers import DictTransformerChain, TransformerChain, call_hook
from .triggers import BotTriggerPolicy, BridgeIdentity, Trigger, TriggerGate
from .utils import (
    TOOL_CALL_DETAILS_ACTION_ID,
    clean_markdown,
    create_feedback_modal,
    create_markdown_text_block,
    create_mrkdwn_text_block,
    extract_feedback_text,
    replace_markdown_images_with_links,
)

logger = logging.getLogger(__name__)


def _clean_blocks_for_retry(blocks: list, error_str: str) -> list:
    """Return a cleaned copy of blocks suitable for a retry after a known Slack API error.

    - ``only_one_table_allowed``: keeps only the first ``table`` block, drops the rest.
    - ``downloading image`` / generic ``invalid_blocks``: strips all ``image`` blocks
      and degrades ``markdown`` text blocks to legacy ``section``/``mrkdwn`` blocks.

    Returns the blocks unchanged if they appear to be a multi-message payload
    (list[list[dict]]) — that case is handled upstream by the ``is_multi_message`` path.
    """
    # Guard: multi-message payloads (list[list[dict]]) cannot be cleaned this way
    if blocks and isinstance(blocks[0], list):
        return blocks
    if "only_one_table_allowed" in error_str:
        cleaned = []
        seen_table = False
        for b in blocks:
            if b.get("type") == "table":
                if not seen_table:
                    cleaned.append(b)
                    seen_table = True
                # extra table blocks silently dropped; their text is already in surrounding sections
            else:
                cleaned.append(b)
        return cleaned
    if "downloading image" in error_str:
        cleaned = []
        for b in blocks:
            if b.get("type") == "image":
                continue
            if b.get("type") == "markdown":
                cleaned.append(
                    {
                        **b,
                        "text": replace_markdown_images_with_links(b.get("text", "")),
                    }
                )
            else:
                cleaned.append(b)
        return cleaned
    # Default: strip image blocks (e.g. image download failure) and convert
    # markdown blocks to legacy mrkdwn sections if Slack rejected the newer block.
    cleaned = []
    for b in blocks:
        if b.get("type") == "image":
            continue
        if b.get("type") == "markdown":
            text = clean_markdown(
                replace_markdown_images_with_links(b.get("text", "")),
                for_blocks=True,
            )
            cleaned.append(create_mrkdwn_text_block(text))
        else:
            cleaned.append(b)
    return cleaned


class SlackBot:
    """Main bot class for LangGraph-Slack integration.

    This class provides a simple interface to connect LangGraph to Slack.
    It handles all the complexity of Slack events, message routing, and
    LangGraph communication.

    Example:
        bot = SlackBot(assistant_id="my-assistant")

        @bot.transform_input
        async def add_context(message, context):
            return f"User: {context.user_id}\\n{message}"

        app = bot.app  # Export for langgraph.json
    """

    def __init__(
        self,
        assistant_id: Optional[str] = None,
        langgraph_url: Optional[str] = None,
        streaming: bool = True,
        reply_in_thread: bool = True,
        slack_bot_token: Optional[str] = None,
        slack_signing_secret: Optional[str] = None,
        show_feedback_buttons: bool = False,
        enable_feedback_comments: bool = False,
        show_thread_id: bool = False,
        extract_images: bool = True,
        max_image_blocks: int = 5,
        include_metadata: bool = True,
        processing_reaction: Optional[str] = None,
        reactions: Optional[list[dict]] = None,
        message_types: Optional[list[str]] = None,
        stream_buffer_time: float = 0.1,
        stream_buffer_max_chunks: int = 10,
        show_tool_calls: bool = False,
        show_tool_call_details: bool = True,
        multitask_strategy: Optional[str] = None,
        delete_interrupted_messages: bool = True,
        state_store: Optional[StateStore] = None,
        allow_bot_triggers: bool = False,
        bot_trigger_allowlist: Optional[list[str]] = None,
        allowlisted_bots_in_joined_threads: bool = False,
        backfill: bool = True,
        backfill_max_messages: int = 20,
        backfill_max_chars: int = 8000,
    ):
        """Initialize SlackBot.

        Args:
            assistant_id: LangGraph assistant ID (or from env: ASSISTANT_ID)
            langgraph_url: LangGraph URL, None for loopback (or from env: LANGGRAPH_URL)
            streaming: Enable streaming responses (default: True)
            reply_in_thread: Reply in thread vs main channel (default: True)
            slack_bot_token: Override Slack bot token (or from env: SLACK_BOT_TOKEN)
            slack_signing_secret: Override Slack signing secret (or from env: SLACK_SIGNING_SECRET)
            show_feedback_buttons: Whether to show feedback buttons (default: True)
            enable_feedback_comments: Enable text input modal for negative feedback (default: False)
            show_thread_id: Whether to show thread_id in footer (default: True)
            extract_images: Extract image markdown and render as blocks (default: True)
            max_image_blocks: Maximum number of image blocks to include (default: 5)
            include_metadata: Include Slack context as metadata in LangGraph (default: True).
                When True, passes the following fields by default: slack_user_id,
                slack_channel_id, slack_message_ts, slack_thread_ts, slack_channel_type,
                slack_is_dm, slack_is_thread. Use @bot.transform_metadata to customize.
            processing_reaction: (DEPRECATED - use reactions instead) Emoji name to add as reaction while processing.
                Must be a Slack emoji name like "eyes", "hourglass", "robot_face".
                Reaction is removed when done. Equivalent to:
                reactions=[{"emoji": "eyes", "target": "user", "when": "processing", "persist": False}]
            reactions: List of reaction configurations for flexible emoji control (default: None).
                Each reaction config is a dict with:
                - emoji (str, required): Slack emoji name like "eyes", "hourglass", "robot_face"
                - target (str, required): "user" (react to user's message) or "bot" (react to bot's response)
                - when (str, required): "processing" (during) or "complete" (after)
                - persist (bool, optional): Keep reaction after done (default: False)
                Example: reactions=[
                    {"emoji": "eyes", "target": "user", "when": "processing"},
                    {"emoji": "white_check_mark", "target": "bot", "when": "complete", "persist": True}
                ]
            message_types: List of LangGraph message types to process in streaming mode (default: ["AIMessageChunk"]).
                Only applies to streaming mode. Non-streaming mode always processes the final response.
                Available message types from LangChain:
                - "ai": AIMessage (complete assistant response)
                - "AIMessageChunk": AIMessageChunk (streaming assistant response chunks)
                - "human": HumanMessage (user input)
                - "system": SystemMessage (system prompts)
                - "tool": ToolMessage (tool execution results)
                Example: message_types=["AIMessageChunk", "tool"] to stream assistant responses and tool results
            stream_buffer_time: Time in seconds to buffer chunks before flushing to Slack (default: 0.1).
                Buffers chunks for this duration before sending to reduce API call overhead.
                The buffer flushes when EITHER this time elapses OR stream_buffer_max_chunks is reached
                (whichever happens first). Lower values = more responsive but more API calls.
                Recommended range: 0.05-0.2 seconds.
                Example: 0.05 = flush every 50ms (more API calls, lower latency)
                         0.2 = flush every 200ms (fewer API calls, slightly higher latency)
            stream_buffer_max_chunks: Maximum chunks to buffer before force-flushing (default: 10).
                Safety limit to prevent accumulating too many small chunks between time-based flushes.
                The buffer flushes when EITHER stream_buffer_time elapses OR this limit is reached
                (whichever happens first). Useful when LangGraph sends many tiny chunks rapidly.
                Recommended range: 5-20 chunks.
                Example: 5 = flush after 5 chunks (more API calls, handles tiny chunks)
                         20 = flush after 20 chunks (fewer API calls, batches more aggressively)
            show_tool_calls: Show a live plan block with tool call status as they execute (default: False).
                Uses Slack's task_card/plan blocks to display each tool call with in_progress →
                complete status. The plan block is posted as a separate message before the text
                response and updated in real-time during streaming.
            show_tool_call_details: When show_tool_calls=True, include truncated input/output previews
                in each task card and add a "View Full Details" button (default: True).
                Clicking the button opens a modal with full untruncated content (SQL, JSON, etc.).
                Set to False to show only tool name + status.
            multitask_strategy: How to handle a new message that arrives while a run is still
                in flight on the same thread (or from env: MULTITASK_STRATEGY). One of:
                - "interrupt" (default): cancel the in-flight run and start the new one.
                - "enqueue": let the current run finish, then process the new message.
                - "reject": reject the new run while one is active.
                - "rollback": cancel and roll back the in-flight run, then start the new one.
                Applies to both streaming and non-streaming handlers. Note: prior versions sent
                no strategy in non-streaming mode (server default "enqueue"); non-streaming now
                uses this setting too. Defaults to "interrupt" for backward compatibility.
            delete_interrupted_messages: When multitask_strategy="interrupt" and a run is
                interrupted mid-generation, delete the partial Slack message instead of leaving
                it frozen mid-sentence (default: True). This is a frontend-only delete
                (chat.delete); it does not modify LangGraph thread state. It is safe because
                LangGraph's interrupt does not checkpoint a node's partial mid-token output, so
                the thread history holds no partial AI message to be out of sync with. Set to
                False to keep (finalize) the partial message, matching legacy behavior. If your
                graph persists partial output to thread state, prefer False.
            state_store: Where the bridge keeps its memory (feedback mappings, tool-call
                details, joined threads, duplicate records). Default: the deployment's
                LangGraph store, or a bounded in-memory store (with a warning) when no store
                is reachable. Pass any object implementing ``StateStore`` to use your own
                storage.
            allow_bot_triggers: Let other bots and integrations (CI, alerting, other agents)
                trigger the agent by @-mentioning it (default: False). The agent's own messages
                never trigger it. Runs carry the bot's provenance; the bridge enforces nothing
                based on it.
            bot_trigger_allowlist: When bot triggers are allowed, limit them to these bot IDs
                (``B...``) or app IDs (``A...``). Default: any bot.
            allowlisted_bots_in_joined_threads: Let allowlisted bots trigger the agent in
                threads it has joined without mentioning it, e.g. a trusted companion agent
                (default: False; bots must always mention the agent). Requires an allowlist.
            backfill: On the first trigger in a Slack thread, hand the agent the thread's
                earlier messages as untrusted context; on later triggers, what was posted
                since it last replied (default: True).
            backfill_max_messages: At most this many backfill messages, the most recent kept
                (default: 20).
            backfill_max_chars: At most this many characters of backfill text (default: 8000).
        """
        logger.info("Initializing SlackBot...")

        # Load configuration (from env + overrides)
        self.config = self._load_config(
            assistant_id=assistant_id,
            langgraph_url=langgraph_url,
            slack_bot_token=slack_bot_token,
            slack_signing_secret=slack_signing_secret,
        )

        # Validate buffer configuration (raises ValueError if invalid)
        self._validate_buffer_config(stream_buffer_time, stream_buffer_max_chunks)

        # Resolve and validate multitask strategy (constructor overrides env; default "interrupt")
        self.multitask_strategy = self._resolve_multitask_strategy(multitask_strategy)
        self.delete_interrupted_messages = delete_interrupted_messages

        # Who besides humans may trigger the agent (validated: raises ValueError)
        self.bot_triggers = BotTriggerPolicy(
            allow=allow_bot_triggers,
            allowlist=frozenset(bot_trigger_allowlist or ()),
            allowlisted_in_joined_threads=allowlisted_bots_in_joined_threads,
        )

        # Store settings
        self.streaming_enabled = streaming
        self.reply_in_thread = reply_in_thread
        self.show_feedback_buttons = show_feedback_buttons
        self.enable_feedback_comments = enable_feedback_comments
        self.show_thread_id = show_thread_id
        self.extract_images = extract_images
        self.max_image_blocks = max_image_blocks
        self.include_metadata = include_metadata
        self.message_types = message_types if message_types is not None else ["AIMessageChunk"]
        self.show_tool_calls = show_tool_calls
        self.show_tool_call_details = show_tool_call_details

        # Normalize reaction configs (handle backward compatibility)
        self.reactions = self._normalize_reactions(processing_reaction, reactions)

        # Initialize transformer chains
        self._input_transformers = TransformerChain()
        self._output_transformers = TransformerChain()
        self._metadata_transformers = DictTransformerChain()
        self._config_transformers = DictTransformerChain()

        # Initialize LangSmith client for feedback
        self.langsmith_client = Client()

        # Initialize LangGraph client
        self.langgraph_client = get_client(url=self.config.LANGGRAPH_URL)
        logger.info(f"LangGraph client initialized (url={self.config.LANGGRAPH_URL or 'loopback'})")

        # Durable bridge state (feedback mappings, tool-call details, joined threads, ...)
        self.state = BridgeState(
            state_store
            or DeploymentStateStore(self.langgraph_client, scope=self.config.ASSISTANT_ID)
        )

        # Initialize Slack app
        self.slack_app = AsyncApp(
            token=self.config.get_slack_bot_token(),
            signing_secret=self.config.get_slack_signing_secret(),
        )
        logger.info("Slack app initialized")

        # Initialize reaction mixin for shared reaction operations
        self._reactions = ReactionMixin(self.slack_app.client)

        # Slack messages -> markdown; earlier thread messages -> backfill; the run's input
        self._normalizer = SlackMessageNormalizer(self.slack_app.client, self.state)
        self._backfill = BackfillCollector(
            self.slack_app.client,
            self.langgraph_client,
            self.state,
            self._normalizer,
            enabled=backfill,
            max_messages=backfill_max_messages,
            max_chars=backfill_max_chars,
        )
        self._input_formatter: Optional[Callable] = None

        # Create appropriate handler based on streaming setting
        if self.streaming_enabled:
            self.handler = StreamingHandler(
                langgraph_client=self.langgraph_client,
                slack_client=self.slack_app,
                assistant_id=self.config.ASSISTANT_ID,
                input_transformers=self._input_transformers,
                output_transformers=self._output_transformers,
                reply_in_thread=self.reply_in_thread,
                show_feedback_buttons=self.show_feedback_buttons,
                show_thread_id=self.show_thread_id,
                extract_images=self.extract_images,
                max_image_blocks=self.max_image_blocks,
                metadata_builder=self._build_metadata,
                input_builder=self._build_run_input,
                config_builder=self._build_config,
                message_types=self.message_types,
                stream_buffer_time=stream_buffer_time,
                stream_buffer_max_chunks=stream_buffer_max_chunks,
                show_tool_calls=self.show_tool_calls,
                show_tool_call_details=self.show_tool_call_details,
                state=self.state,
                multitask_strategy=self.multitask_strategy,
                delete_interrupted_messages=self.delete_interrupted_messages,
            )
            logger.info("Using StreamingHandler (low-latency streaming)")
        else:
            self.handler = MessageHandler(
                langgraph_client=self.langgraph_client,
                assistant_id=self.config.ASSISTANT_ID,
                input_transformers=self._input_transformers,
                output_transformers=self._output_transformers,
                show_feedback_buttons=self.show_feedback_buttons,
                show_thread_id=self.show_thread_id,
                extract_images=self.extract_images,
                max_image_blocks=self.max_image_blocks,
                metadata_builder=self._build_metadata,
                input_builder=self._build_run_input,
                config_builder=self._build_config,
                slack_client=self.slack_app,
                reply_in_thread=self.reply_in_thread,
                show_tool_calls=self.show_tool_calls,
                show_tool_call_details=self.show_tool_call_details,
                state=self.state,
                multitask_strategy=self.multitask_strategy,
            )
            logger.info("Using MessageHandler (non-streaming)")

        # Setup Slack event handlers
        self._setup_slack_handlers()

        # Create FastAPI app
        self._app = self._create_fastapi_app()

        # Slack metadata cache (lazy initialization)
        self._bot_user_id: Optional[str] = None
        self._bot_id: Optional[str] = None
        self._team_id: Optional[str] = None
        self._slack_metadata_lock = asyncio.Lock()
        self._slack_metadata_initialized = False

        # Decides which Slack payloads start a run
        self._trigger_gate = TriggerGate(
            self.slack_app.client, self._get_identity, self.state, self.bot_triggers
        )

        # Background task tracking for proper cleanup
        self._background_tasks: set[asyncio.Task] = set()

        logger.info("SlackBot initialization complete")

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI app instance.

        This is what you export in your script for langgraph.json:

        Example:
            # server.py
            bot = SlackBot(assistant_id="my-assistant")
            app = bot.app  # Export this

        Returns:
            FastAPI app instance
        """
        return self._app

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

    async def send_proactive_message(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> str:
        """Send a Slack message and pre-populate the LangGraph thread with it as context.

        When a bot sends a proactive message outside the normal request/response flow,
        the user's reply would otherwise start a new, empty LangGraph thread. This method
        prevents that by pre-creating the LangGraph thread and injecting the sent message
        as an AIMessage, so the agent has full context when the user replies.

        Note: Context continuity requires the user to reply *in the Slack thread* (i.e.,
        click "Reply in thread" on this message). Unthreaded DM replies start fresh.

        Args:
            channel: Slack channel or DM ID (e.g., "C123..." for channels, "D456..." for DMs)
            text: Message text to send
            thread_ts: Parent message ts to reply into an existing Slack thread

        Returns:
            ts of the sent Slack message

        Example:
            # Send a proactive DM and continue the conversation in the thread
            ts = await bot.send_proactive_message("D123ABC", "Your report is ready!")
        """
        response = await self.slack_app.client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=thread_ts,
        )
        message_ts = response["ts"]

        continuation_ts = thread_ts or message_ts
        thread_id = self.handler._create_thread_id(channel, continuation_ts)
        await self.state.mark_joined(channel, continuation_ts)
        # Replies under this message continue the conversation: catch up from here
        await self._backfill.advance(channel, continuation_ts, message_ts)

        await self.langgraph_client.threads.create(
            thread_id=thread_id,
            if_exists="do_nothing",
        )
        await self.langgraph_client.threads.update_state(
            thread_id,
            values={"messages": [{"role": "assistant", "content": text}]},
        )

        logger.info(
            f"Proactive message sent to {channel}, " f"LangGraph thread {thread_id} pre-populated"
        )
        return message_ts

    async def send_proactive_dm(self, user_id: str, text: str) -> str:
        """Open a DM with a Slack user and send a proactive message with LangGraph context.

        Convenience wrapper around send_proactive_message that opens the DM channel
        from a user ID automatically.

        Args:
            user_id: Slack user ID (e.g., "U123...")
            text: Message text to send

        Returns:
            ts of the sent Slack message

        Example:
            ts = await bot.send_proactive_dm("U123ABC", "Hey, your plant needs water!")
        """
        response = await self.slack_app.client.conversations_open(users=user_id)
        channel = response["channel"]["id"]
        return await self.send_proactive_message(channel, text)

    def transform_input(self, func: Callable) -> Callable:
        """Decorator to add an input transformer.

        Input transformers modify messages before sending to LangGraph.
        Multiple transformers are applied in registration order.

        Example:
            @bot.transform_input
            async def add_user_context(message: str, context: MessageContext) -> str:
                return f"User {context.user_id} says: {message}"

        Args:
            func: Async function (str, MessageContext) -> str

        Returns:
            The function (for decorator usage)
        """
        return self._input_transformers.add(func)

    def transform_output(self, func: Callable) -> Callable:
        """Decorator to add an output transformer.

        Output transformers receive the LangGraph response text (str) and may return
        either a modified string or a list of Slack block dicts for fully custom layouts.
        Multiple transformers are applied in registration order. When a transformer returns
        blocks, the chain short-circuits and remaining transformers are skipped. Blocks fully
        control the final message layout (no text section is auto-prepended by the bot).

        Note:
            When a transformer returns blocks, the ``extract_images`` setting is bypassed —
            the custom blocks fully replace the default image-extraction behavior. If your
            transformer conditionally returns blocks or str, images will only be extracted
            in the str path.

        Examples:
            # Text transformer — modifies the response string
            @bot.transform_output
            async def add_footer(response: str, context: MessageContext) -> str:
                return f"{response}\\n\\n_Powered by AI_"

            # Block transformer — replaces text with custom Slack blocks
            @bot.transform_output
            async def render_table(response: str) -> list[dict]:
                if "|---" in response:
                    return [{"type": "section", "text": {"type": "mrkdwn", "text": response}}]
                return response  # pass through when no table

        Args:
            func: Async function (str[, MessageContext]) -> str | list[dict]

        Returns:
            The function (for decorator usage)
        """
        return self._output_transformers.add(func)

    def transform_metadata(self, func: Callable) -> Callable:
        """Decorator to add a metadata transformer.

        Metadata transformers customize Slack context data sent to LangGraph.
        If no transformer is provided, all MessageContext fields are sent by default.

        Example:
            @bot.transform_metadata
            async def hash_user_id(context: MessageContext) -> dict:
                import hashlib
                return {
                    "channel_id": context.channel_id,
                    "user_id_hash": hashlib.sha256(context.user_id.encode()).hexdigest()[:16],
                    "is_dm": context.is_dm,
                }

        Args:
            func: Async function (MessageContext) -> dict

        Returns:
            The function (for decorator usage)
        """
        return self._metadata_transformers.add(func)

    def format_input(self, func: Callable) -> Callable:
        """Decorator to replace how the run's input message is built.

        By default the request and any backfill reach the agent as one human message: an
        untrusted context section with the earlier Slack messages, then the request with a
        line naming the requester and how the run was triggered. A formatter receives the
        structured trigger and returns either the message content (str) or a full message
        dict (for example with a list of content blocks).

        Example:
            @bot.format_input
            async def as_xml(trigger: TriggerInput, context: MessageContext) -> str:
                earlier = "\\n".join(f"<msg from='{m.display_name}'>{m.text}</msg>"
                                      for m in trigger.backfill)
                return f"<backfill>{earlier}</backfill><request>{trigger.request}</request>"

        Args:
            func: Function (TriggerInput[, MessageContext]) -> str | dict, sync or async

        Returns:
            The function (for decorator usage)
        """
        self._input_formatter = func
        return func

    def transform_config(self, func: Callable) -> Callable:
        """Decorator to add a config transformer.

        Config transformers build the LangGraph config dict (containing configurable
        fields) passed to runs.create() / runs.stream(). This enables injecting
        per-run configuration like repo name, user context, or feature flags that the
        agent graph reads via get_config().

        Example:
            @bot.transform_config
            async def inject_repo(config: dict, context: MessageContext) -> dict:
                config.setdefault("configurable", {})
                repo = extract_repo_from_text(context.event.get("text", ""))
                if repo:
                    config["configurable"]["repo"] = repo
                config["configurable"]["source"] = "slack"
                return config

        Args:
            func: Async function (config: dict, context: MessageContext) -> dict

        Returns:
            The function (for decorator usage)
        """
        return self._config_transformers.add(func)

    def _load_config(
        self,
        assistant_id: Optional[str],
        langgraph_url: Optional[str],
        slack_bot_token: Optional[str],
        slack_signing_secret: Optional[str],
    ) -> BotConfig:
        """Load configuration from env with optional overrides.

        Args:
            assistant_id: Override assistant ID
            langgraph_url: Override LangGraph URL
            slack_bot_token: Override Slack bot token
            slack_signing_secret: Override Slack signing secret

        Returns:
            BotConfig instance

        Raises:
            ValidationError: If required config is missing
        """
        # Start with env vars
        config = BotConfig()

        # Apply overrides if provided
        if assistant_id is not None:
            config.ASSISTANT_ID = assistant_id
        if langgraph_url is not None:
            config.LANGGRAPH_URL = langgraph_url
        if slack_bot_token is not None:
            config.SLACK_BOT_TOKEN = SecretStr(slack_bot_token)
        if slack_signing_secret is not None:
            config.SLACK_SIGNING_SECRET = SecretStr(slack_signing_secret)

        return config

    def _validate_buffer_config(
        self,
        stream_buffer_time: float,
        stream_buffer_max_chunks: int,
    ) -> None:
        """Validate buffer configuration parameters.

        Prevents common configuration mistakes by enforcing sensible bounds on buffering parameters.
        This catches issues like negative values or extreme settings that would degrade user experience.

        Args:
            stream_buffer_time: Time in seconds to buffer chunks (must be > 0, warn if > 5.0)
            stream_buffer_max_chunks: Maximum chunks to buffer (must be >= 1, warn if > 100)

        Raises:
            ValueError: If parameters are invalid (negative, zero, or nonsensical values)
        """
        # Validate stream_buffer_time
        if stream_buffer_time <= 0:
            raise ValueError(f"stream_buffer_time must be positive, got {stream_buffer_time}")
        if stream_buffer_time > 5.0:
            logger.warning(
                f"stream_buffer_time={stream_buffer_time}s is very high. "
                "This may cause noticeable lag in streaming responses. "
                "Recommended range: 0.05-0.2 seconds."
            )

        # Validate stream_buffer_max_chunks
        if stream_buffer_max_chunks < 1:
            raise ValueError(
                f"stream_buffer_max_chunks must be >= 1, got {stream_buffer_max_chunks}"
            )
        if stream_buffer_max_chunks > 100:
            logger.warning(
                f"stream_buffer_max_chunks={stream_buffer_max_chunks} is very high. "
                "This may cause memory issues or delayed flushing. "
                "Recommended range: 5-20 chunks."
            )

    # Supported LangGraph multitask strategies for concurrent runs on the same thread.
    _VALID_MULTITASK_STRATEGIES = frozenset({"interrupt", "enqueue", "reject", "rollback"})

    def _resolve_multitask_strategy(self, multitask_strategy: Optional[str]) -> str:
        """Resolve and validate the multitask strategy.

        Constructor argument overrides the MULTITASK_STRATEGY env value, which in turn
        overrides the "interrupt" default.

        Args:
            multitask_strategy: Explicit strategy from the constructor (or None)

        Returns:
            The resolved strategy string

        Raises:
            ValueError: If the resolved strategy is not one of the supported values
        """
        strategy = multitask_strategy or self.config.MULTITASK_STRATEGY or "interrupt"
        if strategy not in self._VALID_MULTITASK_STRATEGIES:
            raise ValueError(
                f"Invalid multitask_strategy={strategy!r}. "
                f"Must be one of {sorted(self._VALID_MULTITASK_STRATEGIES)}."
            )
        return strategy

    def _convert_legacy_reaction(self, processing_reaction: str) -> dict:
        """Convert legacy processing_reaction parameter to new reaction format.

        Args:
            processing_reaction: Legacy emoji string

        Returns:
            Normalized reaction config dict
        """
        return {
            "emoji": processing_reaction,
            "target": "user",
            "when": "processing",
            "persist": False,
        }

    def _validate_reaction_fields(self, reaction: dict, index: int) -> tuple[str, str, str]:
        """Validate required fields in a reaction config.

        Args:
            reaction: Reaction config dict to validate
            index: Index in the reactions list (for error messages)

        Returns:
            Tuple of (emoji, target, when) values

        Raises:
            ValueError: If required field is missing or invalid type
        """
        if not isinstance(reaction, dict):
            raise ValueError(
                f"Reaction config at index {index} must be a dict, got {type(reaction)}"
            )

        emoji = reaction.get("emoji")
        target = reaction.get("target")
        when = reaction.get("when")

        if not emoji:
            raise ValueError(f"Reaction config at index {index} missing required field 'emoji'")
        if not target:
            raise ValueError(f"Reaction config at index {index} missing required field 'target'")
        if not when:
            raise ValueError(f"Reaction config at index {index} missing required field 'when'")

        return emoji, target, when

    def _normalize_single_reaction(self, reaction: dict, index: int) -> dict:
        """Normalize and validate a single reaction configuration.

        Args:
            reaction: Reaction config dict
            index: Index in the reactions list (for error messages)

        Returns:
            Normalized reaction config dict

        Raises:
            ValueError: If config is invalid
        """
        # Validate required fields
        emoji, target, when = self._validate_reaction_fields(reaction, index)

        # Validate field values
        if target not in ["user", "bot"]:
            raise ValueError(
                f"Reaction config at index {index} has invalid target='{target}'. "
                f"Must be 'user' or 'bot'."
            )
        if when not in ["processing", "complete"]:
            raise ValueError(
                f"Reaction config at index {index} has invalid when='{when}'. "
                f"Must be 'processing' or 'complete'."
            )

        # Set default persist based on when:
        # - processing: False (auto-remove when done)
        # - complete: True (keep the reaction)
        default_persist = when == "complete"

        # Return normalized config with when-dependent persist default
        return {
            "emoji": emoji,
            "target": target,
            "when": when,
            "persist": reaction.get("persist", default_persist),
        }

    def _normalize_reactions(
        self,
        processing_reaction: Optional[str],
        reactions: Optional[list[dict]],
    ) -> list[dict]:
        """Normalize and validate reaction configurations.

        Handles backward compatibility with processing_reaction parameter.
        Delegates to helper methods for single-responsibility validation.

        Args:
            processing_reaction: Legacy single emoji for user message during processing
            reactions: New flexible reaction configs

        Returns:
            List of normalized reaction configs

        Raises:
            ValueError: If reaction config is invalid
        """
        # Handle backward compatibility: processing_reaction -> reactions
        if processing_reaction is not None:
            if reactions is not None:
                logger.warning(
                    "Both processing_reaction and reactions provided. "
                    "processing_reaction is deprecated. Using reactions only."
                )
            else:
                # Convert legacy format to new format
                legacy_reaction = self._convert_legacy_reaction(processing_reaction)
                logger.info(
                    f"Converted processing_reaction='{processing_reaction}' to new reactions format"
                )
                return [legacy_reaction]

        # Use new reactions format if provided
        if reactions:
            return [self._normalize_single_reaction(r, idx) for idx, r in enumerate(reactions)]

        return []

    def _get_reactions_for(self, target: str, when: str) -> list[dict]:
        """Get reactions matching target and when criteria.

        Args:
            target: "user" or "bot"
            when: "processing" or "complete"

        Returns:
            List of matching reaction configs
        """
        return [r for r in self.reactions if r.get("target") == target and r.get("when") == when]

    async def _ensure_slack_metadata(self) -> None:
        """Ensure Slack metadata (bot_user_id, team_id) is cached.

        Uses lazy initialization with lock to prevent race conditions.
        Fetches metadata once on first call, then caches for all subsequent calls.
        This eliminates redundant auth_test() API calls on every message.
        """
        if self._slack_metadata_initialized:
            return  # Already initialized

        async with self._slack_metadata_lock:
            # Double-check after acquiring lock (another task might have initialized)
            if self._slack_metadata_initialized:
                return

            logger.info("Fetching Slack metadata (bot_user_id, team_id) via auth_test()...")
            auth_info = await self.slack_app.client.auth_test()
            self._bot_user_id = auth_info["user_id"]
            self._bot_id = auth_info.get("bot_id")
            self._team_id = auth_info["team_id"]
            self._slack_metadata_initialized = True
            logger.info(
                f"Cached Slack metadata: bot_user_id={self._bot_user_id}, "
                f"bot_id={self._bot_id}, team_id={self._team_id}"
            )

    async def _get_identity(self) -> BridgeIdentity:
        """The bridge's own Slack identity (cached after the first auth check)."""
        await self._ensure_slack_metadata()
        return BridgeIdentity(user_id=self._bot_user_id, bot_id=self._bot_id)

    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app with Slack routes.

        This app is what gets exported to langgraph.json.

        Returns:
            FastAPI app instance with /events/slack endpoint
        """
        app = FastAPI(title="langgraph2slack")

        # Create Slack request handler
        handler = AsyncSlackRequestHandler(self.slack_app)

        @app.post("/events/slack")
        async def slack_events_endpoint(request: Request):
            """Handle Slack events.

            This endpoint receives all Slack events (messages, mentions, etc.)
            and routes them to appropriate handlers.
            """
            return await handler.handle(request)

        return app

    def _prepare_send_blocks(
        self,
        blocks: list,
        response_text: str,
        has_custom_blocks: bool,
    ) -> list:
        """Prepend a text block to standard block layouts.

        When a transformer returned custom blocks they already encode the full
        layout, so nothing is added. For the default image/feedback layout we
        prepend the response text as a Slack markdown block so standard Markdown
        remains visible alongside the blocks.

        Args:
            blocks: Block list returned by the handler.
            response_text: Formatted response text (used only for standard layout).
            has_custom_blocks: True when blocks came from a transformer.

        Returns:
            Final block list ready to send to Slack.
        """
        if blocks and not has_custom_blocks:
            text_block = create_markdown_text_block(response_text)
            return [text_block] + blocks
        return blocks

    async def _build_metadata(self, context: MessageContext) -> dict:
        """Build metadata dict from Slack context.

        If include_metadata is False, returns empty dict.
        If include_metadata is True and user has custom transformer, uses that.
        Otherwise, returns all MessageContext fields.

        Args:
            context: Message context with Slack event data

        Returns:
            Metadata dict to pass to LangGraph
        """
        if not self.include_metadata:
            return {}

        # Check if user provided custom transformer
        if self._metadata_transformers:
            # Custom transformer - user builds metadata from scratch
            # Pass empty dict as first arg since transformers expect (data, context)
            return await self._metadata_transformers.apply({}, context)

        # Default: return all MessageContext fields (exclude raw event)
        return {
            "slack_user_id": context.user_id,
            "slack_channel_id": context.channel_id,
            "slack_message_ts": context.message_ts,
            "slack_thread_ts": context.thread_ts,
            "slack_channel_type": context.channel_type,
            "slack_is_dm": context.is_dm,
            "slack_is_thread": context.is_thread,
            "slack_trigger_kind": context.trigger_kind,
            "slack_author_type": context.author_type,
            "slack_author_id": context.author_id,
            "slack_bot_id": context.bot_id,
            "slack_app_id": context.app_id,
            "slack_requester_id": context.requester_id,
        }

    async def _build_config(self, context: MessageContext) -> dict:
        """Build config dict for LangGraph agent.

        The config always carries the trigger's provenance under
        ``configurable["slack"]``. Config transformers then run on it and may read,
        change or extend it; the agent graph reads the result via get_config().

        Args:
            context: Message context with Slack event data

        Returns:
            Config dict with "configurable" key
        """
        config = {"configurable": {"slack": context.slack_run_config()}}
        if not self._config_transformers:
            return config
        return await self._config_transformers.apply(config, context)

    def _setup_slack_handlers(self) -> None:
        """Setup Slack event handlers.

        Registers handlers for:
        - message events (DMs and channel messages)
        - app_mention events (when bot is @mentioned)
        """

        # Handler for message events
        @self.slack_app.event("message")
        async def handle_message_event(event: dict, say, ack, body: dict):
            """Handle incoming message events."""
            await ack()  # Acknowledge receipt immediately

            trigger = await self._trigger_gate.evaluate(event, event_id=body.get("event_id"))
            if trigger is None:
                logger.debug(f"Ignoring message: {event.get('text', '')[:50]}")
                return

            logger.info(
                f"Processing {trigger.kind} trigger from "
                f"{trigger.provenance.author_type} {trigger.provenance.author_id}"
            )

            # Create context
            # For an edit, the trigger's message is the edited message, not the event
            message = trigger.message
            context = MessageContext(event=message, trigger=trigger)

            # The request as markdown (without the agent's own mention), who asked, and
            # any backfill from the Slack thread
            normalized = await self._normalizer.normalize(
                message, strip_mention_of=self._bot_user_id
            )
            message_text = normalized.text
            context.requested_by = await self._requested_by(context, normalized)
            context.backfill = tuple(await self._collect_backfill(context, trigger))

            # Start user-processing reactions in background (don't block handler)
            user_processing_reactions = self._get_reactions_for("user", "processing")
            if user_processing_reactions:
                self._create_background_task(
                    self._reactions.add_parallel(
                        user_processing_reactions, context.channel_id, context.message_ts
                    )
                )

            success = False
            try:
                # Process based on handler type
                if self.streaming_enabled:
                    # Streaming: handler sends response directly to Slack
                    # Note: Handler will manage bot-processing and bot-complete reactions internally
                    stream_ts, thread_id, run_id = await self.handler.process_message(
                        message_text,
                        context,
                        bot_reactions=self.reactions,  # Pass reactions for bot message handling
                        # Classic integrations have no user: stream to the bridge's own bot user
                        recipient_user_id=context.user_id or self._bot_user_id,
                    )

                    # Add user-complete reactions after streaming completes (parallel)
                    user_complete_reactions = self._get_reactions_for("user", "complete")
                    if user_complete_reactions:
                        await self._reactions.add_parallel(
                            user_complete_reactions, context.channel_id, context.message_ts
                        )

                    # Store mapping for feedback
                    if run_id and stream_ts:
                        await self._remember_reply(context, stream_ts, thread_id, run_id)
                    else:
                        logger.warning("No run_id captured for streaming message")

                    success = True

                else:
                    # Non-streaming: handler returns response, we send it
                    # Determine thread_ts based on reply_in_thread setting
                    if self.reply_in_thread:
                        # Always reply in thread (use message ts if not already in thread)
                        thread_ts = message.get("thread_ts") or message.get("ts")
                    else:
                        # Only reply in thread if message was already in a thread
                        thread_ts = message.get("thread_ts")

                    # Check if we need bot-processing reactions
                    bot_processing_reactions = self._get_reactions_for("bot", "processing")
                    placeholder_ts = None

                    if bot_processing_reactions:
                        # Send placeholder message to show we're thinking
                        placeholder_result = await say(
                            text="Thinking...",
                            thread_ts=thread_ts,
                        )
                        placeholder_ts = (
                            placeholder_result.get("ts") if placeholder_result else None
                        )

                        # Add bot-processing reactions to placeholder
                        if placeholder_ts:
                            for reaction in bot_processing_reactions:
                                await self._reactions.add(
                                    context.channel_id, placeholder_ts, reaction.get("emoji")
                                )

                    logger.info("Non-streaming mode: calling handler.process_message")
                    response_text, blocks, thread_id, run_id, has_custom_blocks = (
                        await self.handler.process_message(message_text, context)
                    )
                    logger.info(
                        f"Handler returned: response_text length={len(response_text)}, blocks count={len(blocks)}, thread_id={thread_id}, run_id={run_id}, has_custom_blocks={has_custom_blocks}"
                    )

                    logger.info(
                        f"Sending message to Slack: thread_ts={thread_ts}, blocks={len(blocks)} blocks"
                    )

                    # Detect multi-message response: transformer returned list[list[dict]]
                    is_multi_message = (
                        has_custom_blocks and bool(blocks) and isinstance(blocks[0], list)
                    )

                    if is_multi_message:
                        first_blocks, *rest_blocks = blocks
                        # Send/update first chunk
                        if placeholder_ts:
                            await self.slack_app.client.chat_update(
                                channel=context.channel_id,
                                ts=placeholder_ts,
                                text=response_text,
                                blocks=first_blocks,
                            )
                            result = {"ts": placeholder_ts}
                        else:
                            result = await say(
                                text=response_text,
                                thread_ts=thread_ts,
                                blocks=first_blocks,
                            )
                        # Send remaining chunks as follow-up messages in the same thread
                        reply_thread = thread_ts or (result.get("ts") if result else None)
                        for chunk_blocks in rest_blocks:
                            await say(
                                text=" ",
                                thread_ts=reply_thread,
                                blocks=chunk_blocks,
                            )
                        logger.info(f"Multi-message: sent {1 + len(rest_blocks)} messages")
                    else:
                        blocks = self._prepare_send_blocks(blocks, response_text, has_custom_blocks)
                        logger.info(f"Prepared blocks for send: {len(blocks)} total")

                    # Update placeholder or send new message
                    if is_multi_message:
                        pass  # already handled above
                    elif placeholder_ts:
                        # Update the placeholder message
                        logger.info(f"Updating placeholder message {placeholder_ts}")
                        try:
                            await self.slack_app.client.chat_update(
                                channel=context.channel_id,
                                ts=placeholder_ts,
                                text=response_text,
                                blocks=blocks if blocks else None,
                            )
                            result = {"ts": placeholder_ts}
                        except Exception as e:
                            # Retry with a cleaned block list for known recoverable errors
                            error_str = str(e)
                            if "invalid_blocks" in error_str or "downloading image" in error_str:
                                cleaned = _clean_blocks_for_retry(blocks, error_str)
                                logger.warning(
                                    f"Block update failed ({error_str}), retrying with cleaned blocks"
                                )
                                await self.slack_app.client.chat_update(
                                    channel=context.channel_id,
                                    ts=placeholder_ts,
                                    text=response_text,
                                    blocks=cleaned if cleaned else None,
                                )
                                result = {"ts": placeholder_ts}
                            else:
                                raise
                    else:
                        # Send message with blocks (or just text if no blocks)
                        # Try sending with blocks first, fallback if Slack rejects them
                        try:
                            result = await say(
                                text=response_text,  # Fallback text for notifications
                                thread_ts=thread_ts,
                                blocks=blocks if blocks else None,
                            )
                            logger.info(
                                f"Message sent successfully, result ts={result.get('ts') if result else 'None'}"
                            )
                        except Exception as e:
                            # Retry with a cleaned block list for known recoverable errors
                            error_str = str(e)
                            if "invalid_blocks" in error_str or "downloading image" in error_str:
                                cleaned = _clean_blocks_for_retry(blocks, error_str)
                                logger.warning(
                                    f"Block send failed ({error_str}), retrying with cleaned blocks"
                                )
                                result = await say(
                                    text=response_text,
                                    thread_ts=thread_ts,
                                    blocks=cleaned if cleaned else None,
                                )
                                logger.info(
                                    f"Message sent with cleaned blocks, result ts={result.get('ts') if result else 'None'}"
                                )
                            else:
                                # Some other error, re-raise it
                                raise

                    # Store mapping for feedback
                    if run_id and result and result.get("ts"):
                        await self._remember_reply(context, result["ts"], thread_id, run_id)
                    else:
                        logger.warning("No run_id captured for non-streaming message")

                    # Add user-complete reactions (parallel)
                    user_complete_reactions = self._get_reactions_for("user", "complete")
                    if user_complete_reactions:
                        await self._reactions.add_parallel(
                            user_complete_reactions, context.channel_id, context.message_ts
                        )

                    # Remove bot-processing reactions and add bot-complete reactions
                    if result and result.get("ts"):
                        # Remove bot-processing reactions if not persistent
                        for reaction in bot_processing_reactions:
                            if not reaction.get("persist", False):
                                await self._reactions.remove(
                                    context.channel_id, result["ts"], reaction.get("emoji")
                                )

                        # Add bot-complete reactions to the bot's response (parallel)
                        bot_complete_reactions = self._get_reactions_for("bot", "complete")
                        if bot_complete_reactions:
                            await self._reactions.add_parallel(
                                bot_complete_reactions, context.channel_id, result["ts"]
                            )

                    success = True

            except Exception:
                # The run never completed: let a corrective edit of this message trigger again
                await self.state.unmark_started(context.channel_id, context.message_ts)
                raise

            finally:
                if success:
                    await self._backfill.advance(
                        context.channel_id,
                        context.conversation_ts,
                        trigger.occurred_ts or context.message_ts,
                    )
                # Only remove non-persistent user reactions on success
                # On error, keep them as visual indicators that something went wrong
                if success:
                    for reaction in self.reactions:
                        if reaction.get("target") == "user" and not reaction.get("persist", False):
                            await self._reactions.remove(
                                context.channel_id, context.message_ts, reaction.get("emoji")
                            )

        # Handler for app_mention events (when bot is @mentioned)
        @self.slack_app.event("app_mention")
        async def handle_mention_event(event: dict, ack):
            """Handle app mention events."""
            await ack()  # Acknowledge immediately
            # These are also delivered as message events, so no need to duplicate processing

        # Handler for feedback button clicks
        @self.slack_app.action("feedback")
        async def handle_feedback(ack, body, client, logger):
            """Handle feedback button clicks.

            For positive feedback: submit directly to LangSmith.
            For negative feedback (with comments enabled): open modal for text input.
            For negative feedback (without comments): submit directly to LangSmith.
            """
            await ack()

            try:
                action = body.get("actions", [{}])[0]
                feedback_value = action.get("value")  # "positive" or "negative"
                user_id = body.get("user", {}).get("id")

                # Extract message information
                container = body.get("container", {})
                message_ts = container.get("message_ts")
                channel_id = container.get("channel_id")

                logger.info(
                    f"Feedback clicked: User {user_id} gave '{feedback_value}' for message {channel_id}:{message_ts}"
                )

                # Look up run_id from message mapping
                run_id = await self._lookup_run_id(channel_id, message_ts)
                if not run_id:
                    return

                # Handle negative feedback with comments enabled
                if feedback_value == "negative" and self.enable_feedback_comments:
                    # Open modal for text input
                    message_context = json.dumps(
                        {
                            "channel_id": channel_id,
                            "message_ts": message_ts,
                            "run_id": run_id,
                        }
                    )
                    modal_view = create_feedback_modal(message_context)

                    await client.views_open(
                        trigger_id=body["trigger_id"],
                        view=modal_view,
                    )
                    logger.info("Opened feedback modal for negative feedback")
                    return

                # Handle positive feedback or negative feedback without comments
                score = 1.0 if feedback_value == "positive" else 0.0
                comment = f"Slack user feedback: {feedback_value}"

                await self._submit_feedback(run_id, score, comment)

            except Exception as e:
                logger.exception(f"Error handling feedback: {e}")

        # Handler for feedback modal submission
        @self.slack_app.view("feedback_modal")
        async def handle_feedback_modal_submission(ack, body, view, logger):
            """Handle feedback modal submission with text."""
            await ack()

            try:
                # Extract message context from private_metadata
                private_metadata = view.get("private_metadata", "{}")
                context = json.loads(private_metadata)

                run_id = context.get("run_id")
                if not run_id:
                    logger.error("No run_id in modal private_metadata")
                    return

                # Extract feedback text from modal
                view_state = view.get("state", {}).get("values", {})
                feedback_text = extract_feedback_text(view_state)

                logger.info(f"Modal submitted with feedback text (length={len(feedback_text)})")

                # Submit negative feedback with optional comment
                comment = feedback_text if feedback_text else None

                await self._submit_feedback(run_id, score=0.0, comment=comment)

            except Exception as e:
                logger.exception(f"Error handling feedback modal submission: {e}")

        # Handler for "View Full Details" button on plan block messages
        @self.slack_app.action(TOOL_CALL_DETAILS_ACTION_ID)
        async def handle_tool_call_details(ack, body, client, logger):
            """Open a modal with full untruncated tool call inputs and outputs.

            The plan_ts stored in the button value is used to look up the tool call
            data from the bridge state, which the handlers populate during
            streaming or after non-streaming completion.
            """
            await ack()

            try:
                action = body.get("actions", [{}])[0]
                plan_ts = action.get("value", "")

                tool_calls = await self.state.tool_calls(plan_ts)
                if not tool_calls:
                    logger.warning(f"No tool call data found for plan_ts={plan_ts}")
                    await client.views_open(
                        trigger_id=body["trigger_id"],
                        view={
                            "type": "modal",
                            "title": {"type": "plain_text", "text": "Tool Call Details"},
                            "close": {"type": "plain_text", "text": "Close"},
                            "blocks": [
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": "_Details are no longer available._",
                                    },
                                }
                            ],
                        },
                    )
                    return

                # Build modal blocks: one section per tool call
                modal_blocks = []
                for tc in tool_calls:
                    # Tool name as header
                    modal_blocks.append(
                        {
                            "type": "header",
                            "text": {"type": "plain_text", "text": tc.name},
                        }
                    )
                    # Status line
                    status_emoji = {"complete": ":white_check_mark:", "error": ":x:"}.get(
                        tc.status, ":hourglass_flowing_sand:"
                    )
                    modal_blocks.append(
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"{status_emoji} *Status:* {tc.status}",
                            },
                        }
                    )
                    # Full input (if any)
                    args = tc.args_json()
                    if args.strip():
                        modal_blocks.append(
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Input:*\n```{args}```",
                                },
                                "expand": True,
                            }
                        )
                    # Full output (if any)
                    if tc.result:
                        modal_blocks.append(
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"*Output:*\n```{tc.result}```",
                                },
                                "expand": True,
                            }
                        )
                    # Divider between tool calls
                    modal_blocks.append({"type": "divider"})

                await client.views_open(
                    trigger_id=body["trigger_id"],
                    view={
                        "type": "modal",
                        "title": {"type": "plain_text", "text": "Tool Call Details"},
                        "close": {"type": "plain_text", "text": "Close"},
                        "blocks": modal_blocks,
                    },
                )

            except Exception as e:
                logger.exception(f"Error opening tool call details modal: {e}")

        logger.info("Slack event handlers registered")

    async def _requested_by(
        self, context: MessageContext, trigger_message: NormalizedMessage
    ) -> RequestedBy:
        """Who the request line names: the requester, or else the triggering bot."""
        if context.requester_id:
            name = await self._normalizer.user_name(context.requester_id)
            return RequestedBy(name, context.requester_id, "human")
        return RequestedBy(
            trigger_message.display_name, trigger_message.author_id, trigger_message.author_type
        )

    async def _collect_backfill(
        self, context: MessageContext, trigger: Trigger
    ) -> list[NormalizedMessage]:
        return await self._backfill.collect(
            channel_id=context.channel_id,
            conversation_ts=context.conversation_ts,
            trigger_ts=context.message_ts,
            thread_id=self.handler.conversation_thread_id(context),
            identity=await self._get_identity(),
            until_ts=trigger.occurred_ts,
        )

    async def _build_run_input(self, request: str, context: MessageContext) -> dict:
        """The single human message the run receives (ADR-0009)."""
        requested_by = context.requested_by or RequestedBy(
            context.author_id or "unknown",
            context.requester_id or context.author_id,
            context.author_type,
        )
        trigger = TriggerInput(
            request=request,
            trigger_kind=context.trigger_kind,
            requested_by=requested_by,
            provenance=context.provenance,
            backfill=context.backfill,
        )
        if not self._input_formatter:
            return default_framer(trigger)
        result = await call_hook(self._input_formatter, trigger, context)
        return result if isinstance(result, dict) else {"role": "user", "content": result}

    async def _remember_reply(
        self, context: MessageContext, reply_ts: str, thread_id: str, run_id: str
    ) -> None:
        """Remember a reply the bridge posted: its run (for feedback) and its thread (joined).

        Args:
            context: Context of the trigger the reply answers
            reply_ts: Timestamp of the bridge's reply message
            thread_id: LangGraph thread ID
            run_id: LangGraph run ID
        """
        await self.state.remember_run(context.channel_id, reply_ts, thread_id, run_id)
        logger.info(f"Stored feedback mapping: {context.channel_id}:{reply_ts} -> run_id={run_id}")

        # The reply sits in the trigger's thread, or (when posted top-level) starts its own
        reply_thread_ts = context.thread_ts or (
            context.message_ts if self.reply_in_thread else None
        )
        await self.state.mark_joined(context.channel_id, reply_thread_ts or reply_ts)

    async def _lookup_run_id(self, channel_id: str, message_ts: str) -> Optional[str]:
        """Look up the run that produced a Slack message.

        Args:
            channel_id: Slack channel ID
            message_ts: Slack message timestamp

        Returns:
            run_id if found, None otherwise
        """
        run_info = await self.state.run_for_message(channel_id, message_ts)
        if not run_info:
            logger.warning(f"No run mapping found for message {channel_id}:{message_ts}")
            return None
        return run_info["run_id"]

    async def _submit_feedback(
        self,
        run_id: str,
        score: float,
        comment: str = None,
    ) -> None:
        """Submit feedback to LangSmith.

        Runs in executor to avoid blocking async loop.

        Args:
            run_id: LangGraph run ID
            score: Feedback score (1.0 for positive, 0.0 for negative)
            comment: Optional comment text
        """
        logger.info(f"Submitting feedback to LangSmith: run_id={run_id}, score={score}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.langsmith_client.create_feedback(
                run_id=run_id,
                key="user_feedback",
                score=score,
                comment=comment,
            ),
        )

        logger.info(f"Feedback submitted successfully for run {run_id}")
