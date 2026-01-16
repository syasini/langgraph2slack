"""Houseplant recommendation agent using LangChain's create_agent.

Simple, fast agent that:
1. Responds immediately with plant care knowledge
2. Automatically searches for images when needed using tools
3. Matches exact image count requested by user (1 image, a few, many, etc.)
4. Uses MessagesState with thread_id for automatic conversation history

This design uses LangChain's create_agent (ReAct pattern) for simplified
implementation while maintaining low latency for simple questions.
"""

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

# Initialize LLM (Haiku 4.5 for speed)
llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7, streaming=True)

# System prompt
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
- NEVER show more images than requested

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


# Build the graph using create_agent
# Note: checkpointer is NOT provided - LangGraph Platform handles persistence automatically
graph = create_agent(
    model=llm,
    tools=[search_plant_images],
    system_prompt=SYSTEM_PROMPT,
)
