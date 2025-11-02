# lg2slack

[![PyPI version](https://badge.fury.io/py/lg2slack.svg)](https://badge.fury.io/py/lg2slack)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple, minimal package to connect LangGraph applications to Slack with just a few lines of code.

## Installation

```bash
pip install lg2slack
```

## Quick Start

### 1. Create a LangGraph App

First, create a simple LangGraph chatbot that will power your Slack bot:

```python
# agent.py
from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Define the chatbot function
def chatbot(state: MessagesState):
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    return {"messages": [model.invoke(state["messages"])]}

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile with memory to maintain conversation history
app = graph.compile(checkpointer=MemorySaver())
```

This creates a simple chatbot that maintains conversation history across messages.

### 2. Create Your Slack Bot Server

This is where the magic happens.

Create a `slack/server.py` file in your project directory:

```python
# slack/server.py
from lg2slack import SlackBot

bot = SlackBot()

# Export the app for langgraph.json
app = bot.app
```

That's it! Just 3 lines of code.

### 3. Configure Environment Variables

Create a `.env` file with your credentials:

```bash
# Slack credentials (from https://api.slack.com/apps -> Your App)
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# LangGraph agent name
# This is the key you will set in langgraph.json
ASSISTANT_ID=my-assistant
```
You can find `SLACK_SIGNING_SECRET` and `SLACK_BOT_TOKEN` in the following pages on `https://api.slack.com/apps`:

![slack_app_creds](https://raw.githubusercontent.com/syasini/lg2slack/main/media/slack_secrets_page.png)


### 4. Configure LangGraph Deployment

Add your agent and Slack server paths to `langgraph.json`:

```json
{
  "dependencies": ["lg2slack", "."],
  "graphs": {
    "my-assistant": "./agent.py:app"
  },
  "env": ".env",
  "http": {
    "/events/slack": "slack/server:app"
  }
}
```

## Local Testing

![local_demo](https://raw.githubusercontent.com/syasini/lg2slack/main/media/local_demo.gif)

Before deploying to production, test your bot locally using ngrok.

**Important:** You'll need **separate Slack apps** for local development and production deployment, since each environment has its own request URL where Slack sends events.

### 1. Create a Slack App for Local Development

- Go to https://api.slack.com/apps
- Click "Create New App" → "From a manifest"
- Copy the contents of `slack_manifest.yaml` from this repo
- Replace placeholder values:
   - `your-app-name` → Your app name (e.g., "My Bot - Local")
   - `your-deployment-url` → This is your ngrok or Langgraph deployment URL. Leave as placeholder for now
- Install the app to your workspace
- Copy the Bot Token and Signing Secret to your `.env` file

### 2. Start LangGraph Dev Server

```bash
langgraph dev
# Runs on http://localhost:2024 and automatically mounts your FastAPI app
```

Note: You don't need to run a separate server! LangGraph dev automatically imports and serves the FastAPI app from your `langgraph.json`.

### 3. Expose with ngrok

Install ngrok if you haven't already:
```bash
# macOS
brew install ngrok

# Or download from https://ngrok.com/download
```

Start ngrok to expose your local server:
```bash
ngrok http 2024
```

This will output something like:
```
Forwarding  https://abc123.ngrok.io -> http://localhost:2024
```

**Tip:** View all requests in ngrok's web interface at http://localhost:4040

### 4. Update Slack App Event URL

Go to your Slack app settings → Event Subscriptions:
- Request URL: `https://abc123.ngrok.io/events/slack` (use YOUR ngrok URL)
- Slack will verify the URL - you should see a green checkmark

### 5. Test Your Bot

Send a DM to your bot or @mention it in a channel! You'll see requests in both:
- LangGraph dev console
- ngrok web interface (http://localhost:4040)



## Production Deployment

Once local testing looks good, deploy to LangGraph Platform.

### 1. Create a Production Slack App

Create a **new** Slack app for production (separate from your local dev app):

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From a manifest"
3. Use the same manifest, but name it differently (e.g., "My Bot - Production")
4. After deployment, you'll update the request URL to your LangGraph Platform URL

### 2. Update Environment Variables

Update your `.env` file with the **production** Slack app credentials:

```bash
# Production Slack credentials
SLACK_BOT_TOKEN=xoxb-your-production-bot-token
SLACK_SIGNING_SECRET=your-production-signing-secret

# LangGraph configuration
ASSISTANT_ID=my-assistant
```

### 3. Deploy to LangGraph Platform

```bash
langgraph deploy
```

After deployment, you'll receive a URL like: `https://your-deployment.langraph.app`

### 4. Update Production Slack App URL

Go to your **production** Slack app settings → Event Subscriptions:
- Request URL: `https://your-deployment.langraph.app/events/slack`

Your bot is now live! Chat with it by:
- Sending a DM to the bot
- @mentioning the bot in a channel

## Advanced Usage

### Configuration Options
![advanced_features](https://raw.githubusercontent.com/syasini/lg2slack/main/media/advanced_features_demo.gif)


The `SlackBot` class accepts many parameters to customize behavior:

```python
bot = SlackBot(
    # LangGraph settings
    assistant_id="my-assistant",        # Or from env: ASSISTANT_ID
    langgraph_url=None,                 # Or from env: LANGGRAPH_URL (None = loopback)

    # Response settings
    streaming=True,                     # Stream responses token-by-token (default: True)
    reply_in_thread=True,               # Always reply in threads (default: True)

    # Slack credentials (or from env)
    slack_bot_token=None,               # From env: SLACK_BOT_TOKEN
    slack_signing_secret=None,          # From env: SLACK_SIGNING_SECRET

    # Feedback integration
    show_feedback_buttons=False,        # Show thumbs up/down buttons (default: False)
    enable_feedback_comments=False,     # Allow text feedback on negative reactions (default: False)
    show_thread_id=False,               # Show LangGraph thread_id in footer (default: False)

    # Image handling
    extract_images=True,                # Convert markdown images to Slack blocks (default: True)
    max_image_blocks=5,                 # Max images per message (default: 5)

    # Metadata tracking
    include_metadata=True,              # Pass Slack context to LangSmith (default: True)

    # Visual feedback
    processing_reaction="eyes",         # Show emoji while processing (default: None)
                                        # Examples: "eyes", "hourglass", "robot_face"

    # Message filtering (streaming only)
    message_types=["AIMessageChunk"],   # Which message types to stream (default: ["AIMessageChunk"])
                                        # Options: "AIMessageChunk", "ai", "tool", "human", "system"
)
```

### Input/Output Transformers

Customize message processing with transformers:

```python
from lg2slack import SlackBot

bot = SlackBot()

# Transform user input before sending to LangGraph
@bot.transform_input
async def add_context(message: str, context) -> str:
    return f"User {context.user_id} asks: {message}"

# Transform AI output before sending to Slack
@bot.transform_output
async def add_footer(response: str, context) -> str:
    return f"{response}\n\n_Powered by LangGraph_"

app = bot.app
```

**Multiple transformers** are applied in registration order:

```python
@bot.transform_input
async def first_transform(message: str, context) -> str:
    return f"[1] {message}"

@bot.transform_input
async def second_transform(message: str, context) -> str:
    return f"[2] {message}"

# Input "hello" becomes: "[2] [1] hello"
```

### Metadata Transformers

Customize what Slack context gets passed to LangSmith:

```python
bot = SlackBot(include_metadata=True)

@bot.transform_metadata
async def custom_metadata(context) -> dict:
    """Customize metadata sent to LangSmith."""
    return {
        "channel_id": context.channel_id,
        "is_dm": context.is_dm,
        "user_id_hash": hash(context.user_id),  # Hash PII for privacy
    }
```

By default, the following fields are passed:
- `slack_user_id`
- `slack_channel_id`
- `slack_message_ts`
- `slack_thread_ts`
- `slack_channel_type`
- `slack_is_dm`
- `slack_is_thread`

### Streaming Mode Control

Control which message types to stream to users:

```python
# Stream only AI responses (default)
bot = SlackBot(message_types=["AIMessageChunk"])

# Stream AI responses AND tool calls
bot = SlackBot(message_types=["AIMessageChunk", "tool"])

# Stream everything (highly verbose!)
bot = SlackBot(message_types=["AIMessageChunk", "ai", "tool", "system"])
```

### Processing Reaction

Show a visual indicator while the bot is thinking:

```python
# Show hourglass emoji while processing
bot = SlackBot(processing_reaction="hourglass")

# Other options: "eyes", "robot_face", "thinking_face", etc.
# Must be emoji NAME, not the emoji character itself
```

The reaction is automatically removed when the response is ready.

### Image Support

The bot automatically extracts markdown images and renders them as Slack image blocks:

```python
# Enable image extraction (default)
bot = SlackBot(extract_images=True, max_image_blocks=5)

# Disable image extraction
bot = SlackBot(extract_images=False)
```

When enabled, markdown like `![Plant](https://example.com/plant.jpg)` in AI responses will:
1. Appear as text in the message
2. Render as a native Slack image block below the text

### Feedback Integration

Collect user feedback and send it to LangSmith:

```python
bot = SlackBot(
    show_thread_id=True,                # Show thread ID for debugging
    show_feedback_buttons=True,         # Show thumbs up/down
    enable_feedback_comments=True,      # Allow text feedback for negative reactions
    
)
```

## How It Works

### Architecture

```
Slack [user] → lg2slack → [INPUT TRANSFORMERS] → LangGraph [HumanMessage]
                                                     ↓
Slack [bot]  ← lg2slack ← [OUTPUT TRANSFORMERS] ← LangGraph [AIMessage]
```

### Message Flow

1. **User sends message** in Slack (DM or @mention)
2. **Input transformers** process the message
3. **Slack sends event** to `/events/slack` endpoint 
4. **Message passed to LangGraph** as HumanMessage with thread_id
5. **LangGraph processes** and generates response as AIMessage
6. **Streaming mode:** Each token immediately forwarded to Slack
7. **Output transformers** process the complete response
8. **Final message** displayed in Slack with optional feedback buttons
9. **Feedback and Metadata** optionally stored in LangSmith if enabled in `.env`

### Streaming vs Non-Streaming
lg2slack assumes you are generating streaming responses from LangGraph, but you have the option to show them on Slack in streaming on non-streaming modes.

**Streaming mode (default):**
- Low-latency streaming
- Each token forwarded immediately to Slack
- Uses Slack's `chat_startStream`, `chat_appendStream`, `chat_stopStream` APIs
- If images are extracted, the text will be replaced with a markdown block.  

**Non-streaming mode:**
```python
bot = SlackBot(streaming=False)
```
- Waits for complete response
- Sends entire message at once
- Useful for debugging or if streaming causes issues

## Examples

Check out the [`examples/plant_bot`](https://github.com/syasini/lg2slack/tree/main/examples/plant_bot) directory for a complete working example:

- **[plant_agent.py](https://github.com/syasini/lg2slack/blob/main/examples/plant_bot/plant_agent.py)** - LangGraph agent with conditional image search
- **[slack_server.py](https://github.com/syasini/lg2slack/blob/main/examples/plant_bot/slack_server.py)** - SlackBot setup with transformers
- **[langgraph.json](https://github.com/syasini/lg2slack/blob/main/examples/plant_bot/langgraph.json)** - Simple deployment configuration

## Requirements

- Python 3.10+
- LangGraph deployment with `messages` state key
- Slack workspace with bot permissions

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.
