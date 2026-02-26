"""
Test all 8 LangGraph/LangChain output scenarios to understand
how each emits data for a Slack connector.
"""
import asyncio
import json
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

# ============================================================
# FAKE LLM SETUP
# ============================================================
from langchain_core.language_models.fake_chat_models import FakeListChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

TABLE_TEXT = """Here are the results:

| Name | Value | Status |
|------|-------|--------|
| Alpha | 42 | Active |
| Beta | 17 | Pending |
| Gamma | 99 | Done |"""

STRUCTURED_JSON = '{"text": "Here is your summary", "image_url": "https://example.com/chart.png", "table": [{"name": "Alpha", "value": 42}, {"name": "Beta", "value": 17}]}'

fake_text_llm = FakeListChatModel(responses=[TABLE_TEXT])
fake_json_llm = FakeListChatModel(responses=[STRUCTURED_JSON])

# ============================================================
# HELPER
# ============================================================
def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def describe_object(obj, label="object", depth=0):
    indent = "  " * depth
    print(f"{indent}{label}: type={type(obj).__name__}")
    if hasattr(obj, 'type'):
        print(f"{indent}  .type = {repr(obj.type)}")
    if hasattr(obj, 'content'):
        content = obj.content
        print(f"{indent}  .content type = {type(content).__name__}")
        if isinstance(content, str):
            print(f"{indent}  .content = {repr(content[:200])}")
        elif isinstance(content, list):
            print(f"{indent}  .content = list of {len(content)} items")
            for i, item in enumerate(content[:3]):
                print(f"{indent}    [{i}]: {repr(item)[:100]}")
    if hasattr(obj, 'tool_calls') and obj.tool_calls:
        print(f"{indent}  .tool_calls = {repr(obj.tool_calls)[:200]}")
    if hasattr(obj, 'tool_call_chunks') and obj.tool_call_chunks:
        print(f"{indent}  .tool_call_chunks = {repr(obj.tool_call_chunks)[:200]}")
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:10]:
            vtype = type(v).__name__
            if isinstance(v, str):
                print(f"{indent}  [{repr(k)}]: {repr(v[:100])}")
            elif isinstance(v, list):
                print(f"{indent}  [{repr(k)}]: list of {len(v)} items")
            elif isinstance(v, dict):
                print(f"{indent}  [{repr(k)}]: dict with keys {list(v.keys())}")
            else:
                print(f"{indent}  [{repr(k)}]: {vtype}")

# ============================================================
# STREAMING TESTS (direct graph.astream)
# ============================================================

async def test_1_react_agent_text_stream():
    print_section("SCENARIO 1: create_react_agent + text table (STREAMING)")
    from langgraph.prebuilt import create_react_agent
    
    graph = create_react_agent(fake_text_llm, tools=[])
    inputs = {"messages": [HumanMessage(content="Show me a table")]}
    
    chunk_count = 0
    print("--- Streaming chunks (stream_mode='messages') ---")
    async for chunk in graph.astream(inputs, stream_mode="messages"):
        chunk_count += 1
        print(f"\n-- Chunk {chunk_count} --")
        if isinstance(chunk, tuple):
            print(f"  Is tuple: True, len={len(chunk)}")
            msg, metadata = chunk
            describe_object(msg, "msg", depth=1)
            print(f"  metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else type(metadata)}")
        else:
            describe_object(chunk, "chunk", depth=1)
        if chunk_count >= 8:
            print("  [... truncated after 8 chunks]")
            break
    
    print(f"\nTotal chunks seen: {chunk_count}")
    
    print("\n--- Same with stream_mode='messages_tuple' ---")
    chunk_count = 0
    async for chunk in graph.astream(inputs, stream_mode="messages_tuple"):
        chunk_count += 1
        if chunk_count <= 3:
            print(f"\n-- Chunk {chunk_count} --")
            if isinstance(chunk, tuple):
                print(f"  Is tuple: True, len={len(chunk)}")
                msg, metadata = chunk
                describe_object(msg, "msg", depth=1)
            else:
                describe_object(chunk, f"chunk {chunk_count}", depth=1)
    print(f"Total chunks (messages_tuple): {chunk_count}")

async def test_2_react_agent_structured_stream():
    print_section("SCENARIO 2: create_react_agent + structured output (STREAMING)")
    from langgraph.prebuilt import create_react_agent
    from langchain_core.pydantic_v1 import BaseModel, Field

    class StructuredOutput(BaseModel):
        text: str = Field(description="Summary text")
        image_url: str = Field(description="URL of chart image")
        table: List[Dict[str, Any]] = Field(description="Data table")

    # Use with_structured_output  
    structured_llm = fake_json_llm.with_structured_output(StructuredOutput)
    
    try:
        graph = create_react_agent(structured_llm, tools=[])
        inputs = {"messages": [HumanMessage(content="Give me structured output")]}
        
        chunk_count = 0
        async for chunk in graph.astream(inputs, stream_mode="messages"):
            chunk_count += 1
            print(f"\n-- Chunk {chunk_count} --")
            if isinstance(chunk, tuple):
                msg, metadata = chunk
                describe_object(msg, "msg", depth=1)
            else:
                describe_object(chunk, "chunk", depth=1)
            if chunk_count >= 8:
                print("  [truncated]")
                break
        print(f"\nTotal chunks: {chunk_count}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        print("NOTE: with_structured_output may not work with FakeListChatModel")
        
        # Try alternative: directly check what structured output does
        print("\n--- Testing with_structured_output behavior ---")
        try:
            structured_llm2 = fake_json_llm.with_structured_output(StructuredOutput)
            result = await structured_llm2.ainvoke([HumanMessage(content="test")])
            print(f"Direct invocation result: type={type(result).__name__}")
            print(f"Result: {result}")
        except Exception as e2:
            print(f"Direct invocation also failed: {e2}")

async def test_3_stategraph_text_stream():
    print_section("SCENARIO 3: StateGraph + MessagesState + text table (STREAMING)")
    from langgraph.graph import StateGraph, MessagesState, END
    from langgraph.graph.message import add_messages
    
    def chatbot(state: MessagesState):
        response = fake_text_llm.invoke(state["messages"])
        return {"messages": [response]}
    
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")
    graph = builder.compile()
    
    inputs = {"messages": [HumanMessage(content="Show me a table")]}
    
    print("--- stream_mode='messages' ---")
    chunk_count = 0
    async for chunk in graph.astream(inputs, stream_mode="messages"):
        chunk_count += 1
        if chunk_count <= 5:
            print(f"\n-- Chunk {chunk_count} --")
            if isinstance(chunk, tuple):
                msg, metadata = chunk
                describe_object(msg, "msg", depth=1)
                if isinstance(metadata, dict):
                    print(f"  metadata: {metadata}")
            else:
                describe_object(chunk, "chunk", depth=1)
    print(f"\nTotal chunks: {chunk_count}")
    
    print("\n--- stream_mode='values' ---")
    async for chunk in graph.astream(inputs, stream_mode="values"):
        print(f"\n-- Values chunk --")
        describe_object(chunk, "state", depth=1)

async def test_4_stategraph_structured_stream():
    print_section("SCENARIO 4: StateGraph + custom state keys (STREAMING)")
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    
    class CustomState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]
        text: str
        image_url: str
        table: List[Dict[str, Any]]
    
    def process_node(state: CustomState) -> CustomState:
        # Simulate agent returning both messages and structured data
        ai_msg = AIMessage(content="Processing complete. See structured output below.")
        return {
            "messages": [ai_msg],
            "text": "Here is your summary",
            "image_url": "https://example.com/chart.png",
            "table": [{"name": "Alpha", "value": 42}, {"name": "Beta", "value": 17}]
        }
    
    builder = StateGraph(CustomState)
    builder.add_node("process", process_node)
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    graph = builder.compile()
    
    inputs = {"messages": [HumanMessage(content="Give me analysis")]}
    
    print("--- stream_mode='messages' ---")
    async for chunk in graph.astream(inputs, stream_mode="messages"):
        if isinstance(chunk, tuple):
            msg, metadata = chunk
            describe_object(msg, "msg", depth=1)
            print(f"  NOTE: Custom state keys (text/image_url/table) NOT visible in messages stream!")
        else:
            describe_object(chunk, "chunk", depth=1)
    
    print("\n--- stream_mode='values' (shows full state) ---")
    async for chunk in graph.astream(inputs, stream_mode="values"):
        print("\n-- Full state snapshot --")
        describe_object(chunk, "state", depth=1)
        if isinstance(chunk, dict):
            print(f"  State keys: {list(chunk.keys())}")
            if "text" in chunk:
                print(f"  text = {repr(chunk['text'])}")
            if "image_url" in chunk:
                print(f"  image_url = {repr(chunk['image_url'])}")
            if "table" in chunk:
                print(f"  table = {chunk['table']}")
    
    print("\n--- stream_mode='updates' ---")
    async for chunk in graph.astream(inputs, stream_mode="updates"):
        print("\n-- Update chunk --")
        describe_object(chunk, "update", depth=1)
    
    print("\n--- stream_mode=['messages', 'values'] COMBINED ---")
    async for chunk in graph.astream(inputs, stream_mode=["messages", "values"]):
        print(f"\n-- Combined chunk: type={type(chunk).__name__} --")
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, data = chunk
            print(f"  mode = {repr(mode)}")
            describe_object(data, "data", depth=1)

# ============================================================
# NON-STREAMING TESTS (graph.invoke)
# ============================================================

async def test_5_react_agent_text_invoke():
    print_section("SCENARIO 5: create_react_agent + text table (NON-STREAMING invoke)")
    from langgraph.prebuilt import create_react_agent
    
    graph = create_react_agent(fake_text_llm, tools=[])
    inputs = {"messages": [HumanMessage(content="Show me a table")]}
    
    result = graph.invoke(inputs)
    print(f"\nResult type: {type(result).__name__}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    
    if isinstance(result, dict):
        describe_object(result, "full result", depth=0)
        if "messages" in result:
            print(f"\nmessages list length: {len(result['messages'])}")
            last_msg = result["messages"][-1]
            print(f"messages[-1] type: {type(last_msg).__name__}")
            describe_object(last_msg, "messages[-1]", depth=1)
            
            # This is how langgraph2slack extracts text:
            if isinstance(last_msg, dict):
                extracted = last_msg.get("content", "")
                print(f"\n[Connector] last_msg.get('content') = {repr(extracted[:100])}")
            elif hasattr(last_msg, 'content'):
                print(f"\n[Connector] last_msg.content = {repr(last_msg.content[:100])}")
                print(f"[Connector] WARNING: connector uses .get() but this is an object, not dict!")

async def test_6_react_agent_structured_invoke():
    print_section("SCENARIO 6: create_react_agent + structured output (NON-STREAMING invoke)")
    from langgraph.prebuilt import create_react_agent
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    class StructuredOutput(BaseModel):
        text: str
        image_url: str
        table: List[Dict[str, Any]]
    
    structured_llm = fake_json_llm.with_structured_output(StructuredOutput)
    
    try:
        graph = create_react_agent(structured_llm, tools=[])
        result = graph.invoke({"messages": [HumanMessage(content="Give structured output")]})
        print(f"\nResult type: {type(result).__name__}")
        describe_object(result, "full result", depth=0)
        if isinstance(result, dict) and "messages" in result:
            last_msg = result["messages"][-1]
            print(f"\nmessages[-1] type: {type(last_msg).__name__}")
            describe_object(last_msg, "messages[-1]", depth=1)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

async def test_7_stategraph_text_invoke():
    print_section("SCENARIO 7: StateGraph + MessagesState + text table (NON-STREAMING invoke)")
    from langgraph.graph import StateGraph, MessagesState
    
    def chatbot(state: MessagesState):
        response = fake_text_llm.invoke(state["messages"])
        return {"messages": [response]}
    
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.set_entry_point("chatbot")
    builder.set_finish_point("chatbot")
    graph = builder.compile()
    
    result = graph.invoke({"messages": [HumanMessage(content="Show me a table")]})
    print(f"\nResult type: {type(result).__name__}")
    print(f"Result keys: {list(result.keys())}")
    
    describe_object(result, "full result", depth=0)
    
    if isinstance(result, dict) and "messages" in result:
        msgs = result["messages"]
        print(f"\nMessages list: {len(msgs)} messages")
        for i, msg in enumerate(msgs):
            print(f"\n  messages[{i}]: type={type(msg).__name__}")
            if hasattr(msg, 'content'):
                print(f"    .content = {repr(msg.content[:100])}")
            elif isinstance(msg, dict):
                print(f"    (dict) content = {repr(msg.get('content', '')[:100])}")

async def test_8_stategraph_structured_invoke():
    print_section("SCENARIO 8: StateGraph + custom state keys (NON-STREAMING invoke)")
    from langgraph.graph import StateGraph
    from langgraph.graph.message import add_messages
    
    class CustomState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]
        text: str
        image_url: str
        table: List[Dict[str, Any]]
    
    def process_node(state: CustomState) -> dict:
        ai_msg = AIMessage(content="Processing complete.")
        return {
            "messages": [ai_msg],
            "text": "Here is your summary",
            "image_url": "https://example.com/chart.png",
            "table": [{"name": "Alpha", "value": 42}, {"name": "Beta", "value": 17}]
        }
    
    builder = StateGraph(CustomState)
    builder.add_node("process", process_node)
    builder.set_entry_point("process")
    builder.set_finish_point("process")
    graph = builder.compile()
    
    result = graph.invoke({
        "messages": [HumanMessage(content="Give me analysis")],
        "text": "",
        "image_url": "",
        "table": []
    })
    
    print(f"\nResult type: {type(result).__name__}")
    print(f"Result keys: {list(result.keys())}")
    describe_object(result, "full result", depth=0)
    
    print("\n--- KEY FINDING: Custom state keys in invoke result ---")
    if isinstance(result, dict):
        print(f"text = {repr(result.get('text', 'MISSING'))}")
        print(f"image_url = {repr(result.get('image_url', 'MISSING'))}")
        print(f"table = {result.get('table', 'MISSING')}")
        
        print("\n--- How connector handles messages[-1] ---")
        if "messages" in result:
            last = result["messages"][-1]
            print(f"messages[-1] type = {type(last).__name__}")
            if hasattr(last, 'content'):
                print(f"messages[-1].content = {repr(last.content)}")
            print(f"\nConnector CANNOT see custom state keys (text/image_url/table)")
            print(f"because it only looks at messages[-1].content")


# ============================================================
# ALSO TEST: What does SDK serialization look like?
# ============================================================

async def test_serialization():
    print_section("BONUS: How messages serialize to dict (simulating SDK format)")
    from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
    from langchain_core.messages.utils import convert_to_openai_messages
    
    ai_msg = AIMessage(content=TABLE_TEXT)
    print(f"AIMessage.content type: {type(ai_msg.content)}")
    print(f"AIMessage.content preview: {repr(ai_msg.content[:100])}")
    
    # How LangGraph SDK serializes messages to dicts
    try:
        msg_dict = ai_msg.dict()
        print(f"\nai_msg.dict() keys: {list(msg_dict.keys())}")
        print(f"  type: {repr(msg_dict.get('type'))}")
        print(f"  content type: {type(msg_dict.get('content')).__name__}")
        print(f"  content preview: {repr(str(msg_dict.get('content'))[:100])}")
    except Exception as e:
        print(f"dict() failed: {e}")
    
    # Also test AIMessageChunk
    chunk = AIMessageChunk(content="Hello ")
    print(f"\nAIMessageChunk.content = {repr(chunk.content)}")
    print(f"AIMessageChunk type = {chunk.type}")
    
    chunk_dict = chunk.dict()
    print(f"AIMessageChunk.dict() keys: {list(chunk_dict.keys())}")
    print(f"  type: {repr(chunk_dict.get('type'))}")


# ============================================================
# MAIN
# ============================================================

async def main():
    print("\n" + "="*60)
    print("  LANGGRAPH OUTPUT FORMAT INVESTIGATION")
    print("="*60)
    
    tests = [
        ("1", test_1_react_agent_text_stream),
        ("2", test_2_react_agent_structured_stream),
        ("3", test_3_stategraph_text_stream),
        ("4", test_4_stategraph_structured_stream),
        ("5", test_5_react_agent_text_invoke),
        ("6", test_6_react_agent_structured_invoke),
        ("7", test_7_stategraph_text_invoke),
        ("8", test_8_stategraph_structured_invoke),
        ("B", test_serialization),
    ]
    
    for num, test in tests:
        try:
            await test()
        except Exception as e:
            print(f"\nERROR in scenario {num}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        print()

if __name__ == "__main__":
    asyncio.run(main())
