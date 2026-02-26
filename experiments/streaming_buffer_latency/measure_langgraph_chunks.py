#!/usr/bin/env python3
"""Measure actual chunk patterns from LangGraph streaming.

This script creates a minimal LangGraph chatbot and measures:
1. How many chunks LangGraph sends
2. Size of each chunk (characters)
3. Time between chunk arrivals
4. Whether chunk patterns match our benchmark assumptions

This validates whether our benchmark (tiny word-by-word chunks)
accurately represents real LangGraph streaming behavior.

Requirements:
- ANTHROPIC_API_KEY environment variable
- LangGraph SDK installed

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python experiments/streaming_buffer_latency/measure_langgraph_chunks.py
"""

import asyncio
import time
import os
from dataclasses import dataclass
from typing import List
import statistics

# Check if running in LangGraph environment
try:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_anthropic import ChatAnthropic
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    print("⚠️  LangGraph not installed. Install with: uv pip install langgraph langchain-anthropic")

# For SDK client testing
try:
    from langgraph_sdk import get_client
    HAS_SDK = True
except ImportError:
    HAS_SDK = False


@dataclass
class ChunkMeasurement:
    """Measurement of a single chunk."""
    chunk_num: int
    content: str
    char_count: int
    arrival_time: float  # Seconds since stream start
    time_since_last: float  # Seconds since previous chunk


class LangGraphChunkAnalyzer:
    """Analyze chunk patterns from LangGraph streaming."""

    def __init__(self, anthropic_api_key: str):
        """Initialize analyzer.

        Args:
            anthropic_api_key: Anthropic API key
        """
        self.api_key = anthropic_api_key

    def create_simple_chatbot(self) -> StateGraph:
        """Create a minimal chatbot for testing.

        Returns:
            Compiled LangGraph graph
        """
        # Create LLM (using Haiku for faster responses, matching plant_bot)
        llm = ChatAnthropic(
            model="claude-haiku-4-5",
            api_key=self.api_key,
            streaming=True,
        )

        def chatbot(state: MessagesState):
            """Simple chatbot node."""
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        # Build graph
        graph = StateGraph(MessagesState)
        graph.add_node("chatbot", chatbot)
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)

        # Compile with memory
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    async def measure_chunks_direct(
        self, test_prompt: str = "Write a haiku about latency."
    ) -> List[ChunkMeasurement]:
        """Measure chunks by invoking graph directly.

        Args:
            test_prompt: Prompt to send to chatbot

        Returns:
            List of chunk measurements
        """
        print(f"\n{'='*70}")
        print("DIRECT GRAPH INVOCATION (In-Process)")
        print(f"{'='*70}")
        print(f"Prompt: {test_prompt}\n")

        graph = self.create_simple_chatbot()
        measurements = []
        stream_start = time.time()
        last_chunk_time = stream_start
        chunk_count = 0
        complete_response = ""

        # Stream from graph using messages mode (returns tuples of (msg, metadata))
        async for msg, metadata in graph.astream(
            {"messages": [{"role": "user", "content": test_prompt}]},
            config={"configurable": {"thread_id": "test-direct"}},
            stream_mode="messages",
        ):
            # Only process AI message chunks
            if not hasattr(msg, 'type') or msg.type != "AIMessageChunk":
                continue

            chunk_count += 1
            content = msg.content

            # Handle list content (Claude returns list of content blocks)
            if isinstance(content, list):
                content = "".join(
                    [block["text"] if isinstance(block, dict) else block.text
                     for block in content
                     if (isinstance(block, dict) and "text" in block) or hasattr(block, "text")]
                )

            if not content:
                continue

            # Track cumulative
            complete_response += content
            current_time = time.time()
            arrival_time = current_time - stream_start
            time_since_last = current_time - last_chunk_time

            measurement = ChunkMeasurement(
                chunk_num=chunk_count,
                content=content,
                char_count=len(content),
                arrival_time=arrival_time,
                time_since_last=time_since_last,
            )
            measurements.append(measurement)

            # Print real-time
            print(
                f"[{arrival_time:.3f}s] Chunk #{chunk_count}: "
                f"{len(content):3d} chars, "
                f"+{time_since_last*1000:5.1f}ms, "
                f"'{content[:50]}...'" if len(content) > 50 else f"'{content}'"
            )

            last_chunk_time = current_time

        total_time = time.time() - stream_start
        print(f"\n✓ Stream completed in {total_time:.2f}s")
        print(f"  Total chunks: {len(measurements)}")
        print(f"  Total characters: {len(complete_response)}")

        return measurements

    async def measure_chunks_sdk(
        self,
        langgraph_url: str = "http://localhost:2024",
        assistant_id: str = "test_agent",
        test_prompt: str = "Write a short poem about streaming data.",
    ) -> List[ChunkMeasurement]:
        """Measure chunks via LangGraph SDK (remote).

        Args:
            langgraph_url: LangGraph server URL
            assistant_id: Assistant/graph ID
            test_prompt: Prompt to send

        Returns:
            List of chunk measurements
        """
        print(f"\n{'='*70}")
        print("SDK CLIENT (Remote LangGraph Server)")
        print(f"{'='*70}")
        print(f"URL: {langgraph_url}")
        print(f"Assistant: {assistant_id}")
        print(f"Prompt: {test_prompt}\n")

        client = get_client(url=langgraph_url)
        measurements = []
        stream_start = time.time()
        last_chunk_time = stream_start
        chunk_count = 0
        complete_response = ""

        # Stream from SDK
        async for chunk in client.runs.stream(
            thread_id="test-sdk",
            assistant_id=assistant_id,
            input={"messages": [{"role": "user", "content": test_prompt}]},
            stream_mode=["messages-tuple"],
        ):
            # Only process message events
            if chunk.event != "messages":
                continue

            # Extract message data
            message_data, _metadata = chunk.data
            if message_data.get("type") != "AIMessageChunk":
                continue

            content = message_data.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    [block.get("text", "") for block in content if block.get("type") == "text"]
                )

            if not content:
                continue

            chunk_count += 1
            current_time = time.time()
            arrival_time = current_time - stream_start
            time_since_last = current_time - last_chunk_time

            # Track cumulative
            complete_response += content

            measurement = ChunkMeasurement(
                chunk_num=chunk_count,
                content=content,
                char_count=len(content),
                arrival_time=arrival_time,
                time_since_last=time_since_last,
            )
            measurements.append(measurement)

            # Print real-time
            print(
                f"[{arrival_time:.3f}s] Chunk #{chunk_count}: "
                f"{len(content):3d} chars, "
                f"+{time_since_last*1000:5.1f}ms, "
                f"'{content[:50]}...'" if len(content) > 50 else f"'{content}'"
            )

            last_chunk_time = current_time

        total_time = time.time() - stream_start
        print(f"\n✓ Stream completed in {total_time:.2f}s")
        print(f"  Total chunks: {len(measurements)}")
        print(f"  Total characters: {len(complete_response)}")

        return measurements

    def analyze_measurements(self, measurements: List[ChunkMeasurement], scenario: str):
        """Analyze and print statistics about chunk patterns.

        Args:
            measurements: List of chunk measurements
            scenario: Description of test scenario
        """
        if not measurements:
            print("\n⚠️  No measurements to analyze")
            return

        print(f"\n{'='*70}")
        print(f"ANALYSIS: {scenario}")
        print(f"{'='*70}\n")

        # Character counts
        char_counts = [m.char_count for m in measurements]
        inter_arrival_times = [m.time_since_last for m in measurements[1:]]  # Skip first

        print("Chunk Size Statistics:")
        print(f"  Total chunks: {len(measurements)}")
        print(f"  Total characters: {sum(char_counts)}")
        print(f"  Avg chunk size: {statistics.mean(char_counts):.1f} chars")
        print(f"  Median chunk size: {statistics.median(char_counts):.1f} chars")
        print(f"  Min chunk size: {min(char_counts)} chars")
        print(f"  Max chunk size: {max(char_counts)} chars")
        print(f"  StdDev: {statistics.stdev(char_counts):.1f} chars" if len(char_counts) > 1 else "  StdDev: N/A")

        print("\nInter-Arrival Time Statistics:")
        if inter_arrival_times:
            print(f"  Avg time between chunks: {statistics.mean(inter_arrival_times)*1000:.1f}ms")
            print(f"  Median time: {statistics.median(inter_arrival_times)*1000:.1f}ms")
            print(f"  Min time: {min(inter_arrival_times)*1000:.1f}ms")
            print(f"  Max time: {max(inter_arrival_times)*1000:.1f}ms")

        # Categorize chunk sizes
        tiny = sum(1 for c in char_counts if c <= 5)
        small = sum(1 for c in char_counts if 5 < c <= 20)
        medium = sum(1 for c in char_counts if 20 < c <= 50)
        large = sum(1 for c in char_counts if c > 50)

        print("\nChunk Size Distribution:")
        print(f"  Tiny (≤5 chars):     {tiny:3d} chunks ({tiny/len(measurements)*100:.1f}%)")
        print(f"  Small (6-20 chars):  {small:3d} chunks ({small/len(measurements)*100:.1f}%)")
        print(f"  Medium (21-50 chars):{medium:3d} chunks ({medium/len(measurements)*100:.1f}%)")
        print(f"  Large (>50 chars):   {large:3d} chunks ({large/len(measurements)*100:.1f}%)")

        # Compare to benchmark assumptions
        print("\n" + "="*70)
        print("COMPARISON TO BENCHMARK ASSUMPTIONS")
        print("="*70)

        benchmark_tiny_chunks = 29  # Our test used 29 tiny word chunks
        benchmark_avg_size = 5  # "Let ", "me ", "help " etc.

        actual_tiny_chunks = tiny
        actual_avg_size = statistics.mean(char_counts)

        print(f"\nBenchmark assumed:")
        print(f"  • {benchmark_tiny_chunks} tiny chunks (~{benchmark_avg_size} chars each)")
        print(f"  • Word-by-word streaming")

        print(f"\nActual LangGraph behavior:")
        print(f"  • {len(measurements)} total chunks (~{actual_avg_size:.1f} chars each)")
        print(f"  • {tiny} tiny chunks (≤5 chars)")

        if actual_avg_size < 10:
            print("\n✓ LangGraph DOES send tiny chunks - benchmark is accurate!")
        elif actual_avg_size < 30:
            print("\n⚠️  LangGraph sends small-medium chunks - benchmark overstates the problem")
        else:
            print("\n✗ LangGraph sends large chunks - benchmark significantly overstates the problem")

        # Estimate buffering benefit
        if inter_arrival_times:
            avg_interval = statistics.mean(inter_arrival_times)
            buffer_time = 0.1  # 100ms buffer

            # How many chunks arrive in 100ms?
            chunks_per_buffer = max(1, int(buffer_time / avg_interval)) if avg_interval > 0 else 1
            api_calls_unbuffered = len(measurements)
            api_calls_buffered = max(1, len(measurements) // chunks_per_buffer)

            print(f"\nEstimated Buffering Impact (100ms buffer):")
            print(f"  Unbuffered API calls: {api_calls_unbuffered}")
            print(f"  Buffered API calls: ~{api_calls_buffered}")
            print(f"  Reduction: {(1 - api_calls_buffered/api_calls_unbuffered)*100:.1f}%")


async def main():
    """Run chunk pattern measurements."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Get your key from: https://console.anthropic.com/")
        return

    if not HAS_LANGGRAPH:
        print("ERROR: LangGraph not installed")
        print("Install with: uv pip install langgraph langchain-anthropic")
        return

    analyzer = LangGraphChunkAnalyzer(api_key)

    print("="*70)
    print("LANGGRAPH CHUNK PATTERN ANALYZER")
    print("="*70)
    print("\nThis measures real chunk patterns from LangGraph streaming")
    print("to validate whether our benchmark accurately represents production behavior.\n")

    # Test 1: Direct graph invocation
    print("Test 1: Measuring chunks from direct graph.astream()...")
    direct_measurements = await analyzer.measure_chunks_direct(
        test_prompt="Write a haiku about latency and performance optimization."
    )
    analyzer.analyze_measurements(direct_measurements, "Direct Graph Invocation")

    # Test 2: SDK client (if available)
    langgraph_url = os.getenv("LANGGRAPH_URL", "http://localhost:2024")
    assistant_id = os.getenv("ASSISTANT_ID", "plant_agent")

    if HAS_SDK:
        print("\n\nTest 2: Measuring chunks from LangGraph SDK client...")
        try:
            sdk_measurements = await analyzer.measure_chunks_sdk(
                langgraph_url=langgraph_url,
                assistant_id=assistant_id,
                test_prompt="Write a haiku about latency and performance optimization."
            )
            analyzer.analyze_measurements(sdk_measurements, "LangGraph SDK Client")
        except Exception as e:
            print(f"\n⚠️  SDK test failed: {e}")
            print(f"   Make sure LangGraph server is running at {langgraph_url}")
            print(f"   Start with: cd examples/plant_bot && langgraph dev")
    else:
        print("\n⚠️  Skipping SDK test (langgraph_sdk not installed)")
        print("   Install with: uv pip install langgraph-sdk")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Based on the measurements above:

1. If LangGraph sends tiny chunks (≤10 chars avg):
   → Benchmark is accurate, buffering provides massive benefit
   → PR #11 should be merged

2. If LangGraph sends medium chunks (10-30 chars avg):
   → Buffering still helps, but benefit is less dramatic
   → Consider adjusting buffer_time or max_chunks

3. If LangGraph sends large chunks (>30 chars avg):
   → Buffering may not provide significant benefit
   → Re-evaluate whether buffering is needed

Compare these real measurements to the benchmark's assumptions!
    """)


if __name__ == "__main__":
    asyncio.run(main())
