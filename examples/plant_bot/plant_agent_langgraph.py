"""Houseplant recommendation agent using pure LangGraph (StateGraph + ToolNode).

Same functionality as plant_agent.py but built with LangGraph primitives:
- StateGraph with MessagesState
- ToolNode for tool execution
- Conditional routing (call_tools → agent or END)
- Streams AIMessageChunk with tool_call_chunks (streaming format)

This is used to verify that langgraph2slack plan/task cards work with the
pure LangGraph streaming format (AIMessageChunk.tool_call_chunks), which
differs from create_agent's non-streaming complete AIMessage.tool_calls.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Initialize LLM (Haiku 4.5 for speed)
llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7, streaming=True)

# System prompt (identical to plant_agent.py)
SYSTEM_PROMPT = """You are a helpful houseplant expert assistant. CRITICAL: Keep responses SHORT and CONCISE.

You help people with:
- Plant care (watering, light, soil, etc.)
- Plant recommendations for different conditions
- Showing images of plants

BREVITY RULES (MUST FOLLOW):
- Maximum 2-3 sentences per response
- Use bullet points for lists (max 3-4 items)
- NO long explanations or elaborations
- Get straight to the point

When users ask to see what a plant looks like, use the search_plant_images tool.

IMAGE COUNT RULE (CRITICAL):
- MATCH the exact number of images the user requests:
  - "show me an image" / "one image" → max_results=1 (show EXACTLY 1 image)
  - "show me a few" / "some images" → max_results=2 or 3
  - "show me several" / "many images" → max_results=3+
- If user doesn't specify count, default to 2-3 images

IMAGE URL RULE:
- If you have search results with images, include them using markdown: ![plant name](IMAGE_URL)
- Include EXACTLY the number of images from the search results (don't add extra)
- DO NOT fabricate image URLs or use placeholder URLs

REMEMBER: Brief, direct answers only. Quality over quantity.
"""


@tool
def search_plant_images(query: str, max_results: int = 3) -> str:
    """Search for plant images and information.

    Use this tool when users ask to see what a plant looks like
    or want images of specific plants.

    Args:
        query: The plant name or search query
        max_results: Number of image results to return (default: 3)
                    - Use 1 if user says "an image" or "one image"
                    - Use 2-3 if user says "a few" or "some"
                    - Use 3+ if user says "several" or "many"

    Returns:
        Search results containing plant images and information
    """
    tavily = TavilySearch(
        max_results=max_results,
        include_images=True,
        search_depth="basic",
    )
    results = tavily.invoke(f"{query} images")
    return str(results)


tools = [search_plant_images]
llm_with_tools = llm.bind_tools(tools)


def agent(state: MessagesState, config: RunnableConfig) -> dict:
    """Call LLM with tools bound; prepend system prompt on first turn.

    Config is passed explicitly so LangGraph can inject streaming callbacks,
    enabling token-level AIMessageChunk streaming (including tool_call_chunks).
    This matches the pattern used by create_react_agent internally.
    """
    messages = state["messages"]

    # Prepend system prompt if not already present
    from langchain_core.messages import SystemMessage
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm_with_tools.invoke(messages, config)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    """Route to tools if the last message has tool calls, else end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "call_tools"
    return END


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("call_tools", ToolNode(tools))

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue, ["call_tools", END])
builder.add_edge("call_tools", "agent")

# Note: checkpointer is NOT provided - LangGraph Platform handles persistence automatically
graph = builder.compile()
