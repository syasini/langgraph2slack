"""Main SlackBot class that orchestrates everything.

This is the primary interface users interact with. It creates the FastAPI app,
sets up Slack handlers, and coordinates between Slack and LangGraph.
"""

import logging
import asyncio
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
from .utils import is_bot_mention, is_dm

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
        show_feedback_buttons: bool = True,
        show_thread_id: bool = True,
        extract_images: bool = True,
        max_image_blocks: int = 5,
        include_metadata: bool = True,
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
            show_thread_id: Whether to show thread_id in footer (default: True)
            extract_images: Extract image markdown and render as blocks (default: True)
            max_image_blocks: Maximum number of image blocks to include (default: 5)
            include_metadata: Include Slack context as metadata in LangGraph (default: False)
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
        self.show_thread_id = show_thread_id
        self.extract_images = extract_images
        self.max_image_blocks = max_image_blocks
        self.include_metadata = include_metadata

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

        # Get bot user ID (needed for mention detection)
        self._bot_user_id: Optional[str] = None

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

            # Process based on handler type
            if self.streaming_enabled:
                # Streaming: handler sends response directly to Slack
                stream_ts, thread_id, run_id = await self.handler.process_message(message_text, context)

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
                logger.info("Non-streaming mode: calling handler.process_message")
                response_text, blocks, thread_id, run_id = await self.handler.process_message(message_text, context)
                logger.info(f"Handler returned: response_text length={len(response_text)}, blocks count={len(blocks)}, thread_id={thread_id}, run_id={run_id}")

                # Determine thread_ts based on reply_in_thread setting
                if self.reply_in_thread:
                    # Always reply in thread (use message ts if not already in thread)
                    thread_ts = event.get("thread_ts") or event.get("ts")
                else:
                    # Only reply in thread if message was already in a thread
                    thread_ts = event.get("thread_ts")

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

        # Handler for app_mention events (when bot is @mentioned)
        @self.slack_app.event("app_mention")
        async def handle_mention_event(event: dict, ack):
            """Handle app mention events."""
            await ack()  # Acknowledge immediately
            # These are also delivered as message events, so no need to duplicate processing

        # Handler for feedback button clicks
        @self.slack_app.action("feedback")
        async def handle_feedback(ack, body, logger):
            """Handle feedback button clicks and submit to LangSmith."""
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

                # Look up the run information
                message_key = f"{channel_id}:{message_ts}"
                run_info = self.message_run_mapping.get(message_key)

                if not run_info:
                    logger.error(f"No run mapping found for message key: {message_key}")
                    logger.error(f"Available mappings: {list(self.message_run_mapping.keys())}")
                    return

                run_id = run_info["run_id"]

                # Convert feedback to score (1.0 for positive, 0.0 for negative)
                score = 1.0 if feedback_value == "positive" else 0.0

                # Submit feedback to LangSmith (run in thread pool to avoid blocking async loop)
                logger.info(f"Submitting feedback to LangSmith: run_id={run_id}, score={score}")

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.langsmith_client.create_feedback(
                        run_id=run_id,
                        key="user_feedback",
                        score=score,
                        comment=f"Slack user feedback: {feedback_value}",
                    ),
                )

                logger.info(f"Feedback submitted successfully to LangSmith for run {run_id}")

            except Exception as e:
                logger.exception(f"Error handling feedback: {e}")

        logger.info("Slack event handlers registered")

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

        # Get bot user ID if we don't have it yet
        if not self._bot_user_id:
            auth_info = await self.slack_app.client.auth_test()
            self._bot_user_id = auth_info["user_id"]
            logger.info(f"Bot user ID: {self._bot_user_id}")

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
