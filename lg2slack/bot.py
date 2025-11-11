"""Main SlackBot class that orchestrates everything.

This is the primary interface users interact with. It creates the FastAPI app,
sets up Slack handlers, and coordinates between Slack and LangGraph.
"""

import logging
import asyncio
import json
from typing import Optional, Callable, Dict
from fastapi import FastAPI, Request
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from langgraph_sdk import get_client
from langsmith import Client
from pydantic import SecretStr

from .config import BotConfig, MessageContext
from .handlers import MessageHandler, StreamingHandler
from .transformers import TransformerChain
from .utils import is_bot_mention, is_dm, create_feedback_modal, extract_feedback_text

logger = logging.getLogger(__name__)


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
        stream_buffer_max_size: int = 500,
        stream_buffer_max_chunks: int = 10,
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
                Reduces API call overhead by batching multiple chunks. Lower = more responsive but more API calls.
                Recommended: 0.05-0.2 seconds.
            stream_buffer_max_size: Maximum characters to buffer before force-flushing (default: 500).
                Prevents accumulating too much text in buffer. Higher = fewer API calls but longer perceived delay.
            stream_buffer_max_chunks: Maximum number of chunks to buffer before force-flushing (default: 10).
                Prevents accumulating too many small chunks. Higher = fewer API calls but longer perceived delay.
        """
        logger.info("Initializing SlackBot...")

        # Load configuration (from env + overrides)
        self.config = self._load_config(
            assistant_id=assistant_id,
            langgraph_url=langgraph_url,
            slack_bot_token=slack_bot_token,
            slack_signing_secret=slack_signing_secret,
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

        # Normalize reaction configs (handle backward compatibility)
        self.reactions = self._normalize_reactions(processing_reaction, reactions)

        # Initialize transformer chains
        self._input_transformers = TransformerChain()
        self._output_transformers = TransformerChain()
        self._metadata_transformers = TransformerChain()

        # Initialize LangSmith client for feedback
        self.langsmith_client = Client()

        # Map Slack message identifiers to LangGraph run information for feedback
        # Key: "{channel_id}:{message_ts}", Value: {"thread_id": str, "run_id": str}
        self.message_run_mapping: Dict[str, Dict[str, str]] = {}

        # Initialize LangGraph client
        self.langgraph_client = get_client(url=self.config.LANGGRAPH_URL)
        logger.info(f"LangGraph client initialized (url={self.config.LANGGRAPH_URL or 'loopback'})")

        # Initialize Slack app
        self.slack_app = AsyncApp(
            token=self.config.get_slack_bot_token(),
            signing_secret=self.config.get_slack_signing_secret(),
        )
        logger.info("Slack app initialized")

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
                message_types=self.message_types,
                stream_buffer_time=stream_buffer_time,
                stream_buffer_max_size=stream_buffer_max_size,
                stream_buffer_max_chunks=stream_buffer_max_chunks,
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
            )
            logger.info("Using MessageHandler (non-streaming)")

        # Setup Slack event handlers
        self._setup_slack_handlers()

        # Create FastAPI app
        self._app = self._create_fastapi_app()

        # Slack metadata cache (lazy initialization)
        self._bot_user_id: Optional[str] = None
        self._team_id: Optional[str] = None
        self._slack_metadata_lock = asyncio.Lock()
        self._slack_metadata_initialized = False

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

        Output transformers modify LangGraph responses before sending to Slack.
        Multiple transformers are applied in registration order.

        Example:
            @bot.transform_output
            async def add_footer(response: str, context: MessageContext) -> str:
                return f"{response}\\n\\n_Powered by AI_"

        Args:
            func: Async function (str, MessageContext) -> str

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

    def _normalize_reactions(
        self,
        processing_reaction: Optional[str],
        reactions: Optional[list[dict]],
    ) -> list[dict]:
        """Normalize and validate reaction configurations.

        Handles backward compatibility with processing_reaction parameter.

        Args:
            processing_reaction: Legacy single emoji for user message during processing
            reactions: New flexible reaction configs

        Returns:
            List of normalized reaction configs

        Raises:
            ValueError: If reaction config is invalid
        """
        # Start with empty list
        normalized = []

        # Handle backward compatibility: processing_reaction -> reactions
        if processing_reaction is not None:
            if reactions is not None:
                logger.warning(
                    "Both processing_reaction and reactions provided. "
                    "processing_reaction is deprecated. Using reactions only."
                )
            else:
                # Convert legacy format to new format
                normalized.append({
                    "emoji": processing_reaction,
                    "target": "user",
                    "when": "processing",
                    "persist": False,
                })
                logger.info(f"Converted processing_reaction='{processing_reaction}' to new reactions format")
                return normalized

        # Use new reactions format if provided
        if reactions:
            for idx, reaction in enumerate(reactions):
                # Validate required fields
                if not isinstance(reaction, dict):
                    raise ValueError(f"Reaction config at index {idx} must be a dict, got {type(reaction)}")

                emoji = reaction.get("emoji")
                target = reaction.get("target")
                when = reaction.get("when")

                if not emoji:
                    raise ValueError(f"Reaction config at index {idx} missing required field 'emoji'")
                if not target:
                    raise ValueError(f"Reaction config at index {idx} missing required field 'target'")
                if not when:
                    raise ValueError(f"Reaction config at index {idx} missing required field 'when'")

                # Validate values
                if target not in ["user", "bot"]:
                    raise ValueError(
                        f"Reaction config at index {idx} has invalid target='{target}'. "
                        f"Must be 'user' or 'bot'."
                    )
                if when not in ["processing", "complete"]:
                    raise ValueError(
                        f"Reaction config at index {idx} has invalid when='{when}'. "
                        f"Must be 'processing' or 'complete'."
                    )

                # Set default persist based on when:
                # - processing: False (auto-remove when done)
                # - complete: True (keep the reaction)
                default_persist = when == "complete"

                # Add normalized config with when-dependent persist default
                normalized.append({
                    "emoji": emoji,
                    "target": target,
                    "when": when,
                    "persist": reaction.get("persist", default_persist),
                })

        return normalized

    def _get_reactions_for(self, target: str, when: str) -> list[dict]:
        """Get reactions matching target and when criteria.

        Args:
            target: "user" or "bot"
            when: "processing" or "complete"

        Returns:
            List of matching reaction configs
        """
        return [
            r for r in self.reactions
            if r.get("target") == target and r.get("when") == when
        ]

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
            self._team_id = auth_info["team_id"]
            self._slack_metadata_initialized = True
            logger.info(f"Cached Slack metadata: bot_user_id={self._bot_user_id}, team_id={self._team_id}")

    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app with Slack routes.

        This app is what gets exported to langgraph.json.

        Returns:
            FastAPI app instance with /events/slack endpoint
        """
        app = FastAPI(title="lg2slack")

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
        }

    def _setup_slack_handlers(self) -> None:
        """Setup Slack event handlers.

        Registers handlers for:
        - message events (DMs and channel messages)
        - app_mention events (when bot is @mentioned)
        """

        # Handler for message events
        @self.slack_app.event("message")
        async def handle_message_event(event: dict, say, ack):
            """Handle incoming message events."""
            await ack()  # Acknowledge receipt immediately

            # Check if we should process this message
            if not await self._should_process_message(event):
                logger.debug(f"Ignoring message: {event.get('text', '')[:50]}")
                return

            logger.info(f"Processing message from {event.get('user')}")

            # Create context
            context = MessageContext(event=event)

            # Extract message text
            message_text = event.get("text", "")

            # Remove bot mention from text if present
            if self._bot_user_id:
                message_text = message_text.replace(f"<@{self._bot_user_id}>", "").strip()

            # Start user-processing reactions in background (don't block handler)
            user_processing_reactions = self._get_reactions_for("user", "processing")
            if user_processing_reactions:
                asyncio.create_task(
                    self._add_reactions_parallel(
                        user_processing_reactions,
                        context.channel_id,
                        context.message_ts
                    )
                )

            try:
                # Process based on handler type
                if self.streaming_enabled:
                    # Streaming: handler sends response directly to Slack
                    # Note: Handler will manage bot-processing and bot-complete reactions internally
                    stream_ts, thread_id, run_id = await self.handler.process_message(
                        message_text,
                        context,
                        bot_reactions=self.reactions,  # Pass reactions for bot message handling
                    )

                    # Add user-complete reactions after streaming completes (parallel)
                    user_complete_reactions = self._get_reactions_for("user", "complete")
                    if user_complete_reactions:
                        await self._add_reactions_parallel(
                            user_complete_reactions,
                            context.channel_id,
                            context.message_ts
                        )

                    # Store mapping for feedback
                    if run_id and stream_ts:
                        message_key = f"{context.channel_id}:{stream_ts}"
                        self.message_run_mapping[message_key] = {
                            "thread_id": thread_id,
                            "run_id": run_id,
                        }
                        logger.info(f"Stored feedback mapping: {message_key} -> thread_id={thread_id}, run_id={run_id}")
                    else:
                        logger.warning(f"No run_id captured for streaming message")

                else:
                    # Non-streaming: handler returns response, we send it
                    # Determine thread_ts based on reply_in_thread setting
                    if self.reply_in_thread:
                        # Always reply in thread (use message ts if not already in thread)
                        thread_ts = event.get("thread_ts") or event.get("ts")
                    else:
                        # Only reply in thread if message was already in a thread
                        thread_ts = event.get("thread_ts")

                    # Check if we need bot-processing reactions
                    bot_processing_reactions = self._get_reactions_for("bot", "processing")
                    placeholder_ts = None

                    if bot_processing_reactions:
                        # Send placeholder message to show we're thinking
                        placeholder_result = await say(
                            text="Thinking...",
                            thread_ts=thread_ts,
                        )
                        placeholder_ts = placeholder_result.get("ts") if placeholder_result else None

                        # Add bot-processing reactions to placeholder
                        if placeholder_ts:
                            for reaction in bot_processing_reactions:
                                await self._add_reaction(context.channel_id, placeholder_ts, reaction.get("emoji"))

                    logger.info("Non-streaming mode: calling handler.process_message")
                    response_text, blocks, thread_id, run_id = await self.handler.process_message(message_text, context)
                    logger.info(f"Handler returned: response_text length={len(response_text)}, blocks count={len(blocks)}, thread_id={thread_id}, run_id={run_id}")

                    logger.info(f"Sending message to Slack: thread_ts={thread_ts}, blocks={len(blocks)} blocks")

                    # If we have blocks, prepend a text section block with the response
                    if blocks:
                        # Create a text section block for the response
                        text_block = {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": response_text
                            }
                        }
                        # Prepend text block to the beginning
                        blocks = [text_block] + blocks
                        logger.info(f"Added text block, total blocks: {len(blocks)}")

                    # Update placeholder or send new message
                    if placeholder_ts:
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
                            # If error mentions invalid_blocks or downloading image, retry without image blocks
                            error_str = str(e)
                            if "invalid_blocks" in error_str or "downloading image" in error_str:
                                logger.warning(f"Image blocks failed ({error_str}), retrying without images")
                                blocks_without_images = [b for b in blocks if b.get("type") != "image"]
                                await self.slack_app.client.chat_update(
                                    channel=context.channel_id,
                                    ts=placeholder_ts,
                                    text=response_text,
                                    blocks=blocks_without_images if blocks_without_images else None,
                                )
                                result = {"ts": placeholder_ts}
                            else:
                                raise
                    else:
                        # Send message with blocks (or just text if no blocks)
                        # Try sending with blocks first, fallback to text-only if image download fails
                        try:
                            result = await say(
                                text=response_text,  # Fallback text for notifications
                                thread_ts=thread_ts,
                                blocks=blocks if blocks else None,
                            )
                            logger.info(f"Message sent successfully, result ts={result.get('ts') if result else 'None'}")
                        except Exception as e:
                            # If error mentions invalid_blocks or downloading image, retry without image blocks
                            error_str = str(e)
                            if "invalid_blocks" in error_str or "downloading image" in error_str:
                                logger.warning(f"Image blocks failed ({error_str}), retrying without images")
                                # Keep text block and feedback block, remove image blocks
                                blocks_without_images = [b for b in blocks if b.get("type") != "image"]
                                result = await say(
                                    text=response_text,
                                    thread_ts=thread_ts,
                                    blocks=blocks_without_images if blocks_without_images else None,
                                )
                                logger.info(f"Message sent without images, result ts={result.get('ts') if result else 'None'}")
                            else:
                                # Some other error, re-raise it
                                raise

                    # Store mapping for feedback
                    if run_id and result and result.get("ts"):
                        message_key = f"{context.channel_id}:{result['ts']}"
                        self.message_run_mapping[message_key] = {
                            "thread_id": thread_id,
                            "run_id": run_id,
                        }
                        logger.info(f"Stored feedback mapping: {message_key} -> thread_id={thread_id}, run_id={run_id}")
                    else:
                        logger.warning(f"No run_id captured for non-streaming message")

                    # Add user-complete reactions (parallel)
                    user_complete_reactions = self._get_reactions_for("user", "complete")
                    if user_complete_reactions:
                        await self._add_reactions_parallel(
                            user_complete_reactions,
                            context.channel_id,
                            context.message_ts
                        )

                    # Remove bot-processing reactions and add bot-complete reactions
                    if result and result.get("ts"):
                        # Remove bot-processing reactions if not persistent
                        for reaction in bot_processing_reactions:
                            if not reaction.get("persist", False):
                                await self._remove_reaction(context.channel_id, result["ts"], reaction.get("emoji"))

                        # Add bot-complete reactions to the bot's response (parallel)
                        bot_complete_reactions = self._get_reactions_for("bot", "complete")
                        if bot_complete_reactions:
                            await self._add_reactions_parallel(
                                bot_complete_reactions,
                                context.channel_id,
                                result["ts"]
                            )

            finally:
                # Remove all non-persistent user reactions
                for reaction in self.reactions:
                    if reaction.get("target") == "user" and not reaction.get("persist", False):
                        await self._remove_reaction(context.channel_id, context.message_ts, reaction.get("emoji"))

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
                run_id = self._lookup_run_id(channel_id, message_ts)
                if not run_id:
                    return

                # Handle negative feedback with comments enabled
                if feedback_value == "negative" and self.enable_feedback_comments:
                    # Open modal for text input
                    message_context = json.dumps({
                        "channel_id": channel_id,
                        "message_ts": message_ts,
                        "run_id": run_id,
                    })
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

        logger.info("Slack event handlers registered")

    def _lookup_run_id(self, channel_id: str, message_ts: str) -> Optional[str]:
        """Look up run_id from message mapping.

        Args:
            channel_id: Slack channel ID
            message_ts: Slack message timestamp

        Returns:
            run_id if found, None otherwise
        """
        message_key = f"{channel_id}:{message_ts}"
        run_info = self.message_run_mapping.get(message_key)

        if not run_info:
            logger.error(f"No run mapping found for message key: {message_key}")
            logger.error(f"Available mappings: {list(self.message_run_mapping.keys())}")
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
            await self.slack_app.client.reactions_add(
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
            await self.slack_app.client.reactions_remove(
                channel=channel_id,
                timestamp=message_ts,
                name=emoji,
            )
            logger.debug(f"Removed reaction :{emoji}: from message {channel_id}:{message_ts}")
        except Exception as e:
            logger.warning(f"Failed to remove reaction :{emoji}:: {e}")

    async def _add_reactions_parallel(
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
                *[self._add_reaction(channel_id, message_ts, r.get("emoji"))
                  for r in reactions],
                return_exceptions=True  # Don't fail entire batch if one fails
            )
        except Exception as e:
            logger.error(f"Parallel reactions failed: {e}", exc_info=True)

    async def _should_process_message(self, event: dict) -> bool:
        """Determine if we should process this message.

        We process messages if:
        - Bot is mentioned (in a channel)
        - Message is a DM to the bot
        - Message has a user (not from another bot)

        Args:
            event: Slack message event dict

        Returns:
            True if we should process this message
        """
        # Skip messages without a user (e.g., bot messages, system messages)
        if not event.get("user"):
            return False

        # Skip messages from bots (including ourselves)
        if event.get("bot_id"):
            return False

        # Ensure Slack metadata is cached (fetches once, then cached forever)
        await self._ensure_slack_metadata()

        # Check if this is a DM
        if is_dm(event):
            logger.debug("Message is a DM - processing")
            return True

        # Check if we're in a thread where the bot has already participated
        thread_ts = event.get("thread_ts")
        if thread_ts:
            # We're in a thread - check conversation history to see if bot has participated
            channel_id = event.get("channel")

            try:
                # Fetch thread history to check if bot has responded
                thread_history = await self.slack_app.client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    limit=100  # Check last 100 messages in thread
                )

                # Check if any message in the thread is from the bot
                for message in thread_history.get("messages", []):
                    if message.get("user") == self._bot_user_id or message.get("bot_id"):
                        # Check if it's our bot specifically
                        if message.get("user") == self._bot_user_id:
                            logger.debug(f"Bot has participated in thread {thread_ts} - processing without mention")
                            return True
            except Exception as e:
                logger.warning(f"Could not check thread history: {e}")
                # Fall through to mention check if history lookup fails

        # Check if bot is mentioned
        message_text = event.get("text", "")
        if is_bot_mention(message_text, self._bot_user_id):
            logger.debug("Bot is mentioned - processing")
            return True

        # Not a DM, not in an active thread, and bot not mentioned - skip
        return False
