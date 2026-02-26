#!/usr/bin/env python3
"""Real-world Slack streaming latency test.

This script uses the ACTUAL Slack API to measure streaming behavior:
1. Send many small chunks rapidly (simulating unbuffered)
2. Send fewer larger chunks (simulating buffered)
3. Measure time deltas to see if Slack queues sequential calls

Requirements:
- SLACK_BOT_TOKEN environment variable
- A test channel to send messages to
- Bot must have chat:write permission

Usage:
    export SLACK_BOT_TOKEN="xoxb-your-token"
    export SLACK_TEST_CHANNEL="C01234567890"  # Find this from Slack
    python tests/benchmark_real_slack_streaming.py
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import List
import statistics

from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError


@dataclass
class StreamingTest:
    """Results from a streaming test."""
    name: str
    chunks_sent: int
    api_calls_made: int
    total_time: float  # Total time from start to finish
    call_durations: List[float]  # How long each API call took
    inter_call_gaps: List[float]  # Time between starting successive calls


class SlackStreamingBenchmark:
    """Benchmark Slack's streaming API behavior."""

    def __init__(self, slack_token: str, test_channel: str):
        """Initialize benchmark.

        Args:
            slack_token: Slack bot token
            test_channel: Channel ID to send test messages to
        """
        # Initialize Slack app
        self.app = AsyncApp(token=slack_token)
        self.client = self.app.client
        self.test_channel = test_channel

    async def _get_team_id(self) -> str:
        """Get team ID for streaming API."""
        auth_response = await self.client.auth_test()
        return auth_response["team_id"]

    async def _get_bot_user_id(self) -> str:
        """Get bot's user ID."""
        auth_response = await self.client.auth_test()
        return auth_response["user_id"]

    async def test_unbuffered_streaming(self, message_prefix: str) -> StreamingTest:
        """Test sending many tiny chunks rapidly (simulates unbuffered).

        This simulates the ORIGINAL behavior where every tiny chunk
        from LangGraph triggers an immediate Slack API call.

        Args:
            message_prefix: Prefix for test message (to identify it)

        Returns:
            StreamingTest with results
        """
        print(f"\n{'='*60}")
        print(f"TEST: Unbuffered (many tiny chunks)")
        print(f"{'='*60}")

        # Simulate tiny chunks like "Let ", "me ", "help ", "you ", etc.
        tiny_chunks = [
            "Let ", "me ", "help ", "you ", "with ", "that. ",
            "This ", "is ", "a ", "test ", "of ", "unbuffered ",
            "streaming. ", "Each ", "word ", "is ", "a ", "separate ",
            "API ", "call. ", "We'll ", "see ", "if ", "they ",
            "queue ", "up ", "on ", "Slack's ", "side. ",
        ]

        team_id = await self._get_team_id()
        bot_user_id = await self._get_bot_user_id()
        call_durations = []
        inter_call_gaps = []

        # Start stream
        start_time = time.time()
        print(f"[{0:.3f}s] Starting stream...")

        # Post a regular message first to get a thread_ts
        initial_msg = await self.client.chat_postMessage(
            channel=self.test_channel,
            text="🧪 Starting streaming test..."
        )
        thread_ts = initial_msg["ts"]

        start_response = await self.client.chat_startStream(
            channel=self.test_channel,
            recipient_team_id=team_id,
            recipient_user_id=bot_user_id,  # Bot's own user ID
            thread_ts=thread_ts,  # Reply in thread
        )
        stream_ts = start_response["ts"]
        print(f"[{time.time() - start_time:.3f}s] Stream started: {stream_ts}")

        # Send chunks as fast as possible (no artificial delays)
        last_call_start = time.time()
        api_call_count = 0

        for i, chunk in enumerate(tiny_chunks):
            call_start = time.time()

            # Record gap since last call started
            if i > 0:
                gap = call_start - last_call_start
                inter_call_gaps.append(gap)

            # Send chunk
            await self.client.chat_appendStream(
                channel=self.test_channel,
                ts=stream_ts,
                markdown_text=chunk,
            )
            api_call_count += 1

            call_duration = time.time() - call_start
            call_durations.append(call_duration)

            print(f"[{time.time() - start_time:.3f}s] Chunk #{i+1}: '{chunk.strip()}' "
                  f"(call took {call_duration*1000:.1f}ms)")

            last_call_start = call_start

        # Stop stream
        await self.client.chat_stopStream(
            channel=self.test_channel,
            ts=stream_ts,
        )

        total_time = time.time() - start_time
        print(f"[{total_time:.3f}s] Stream stopped")
        print(f"\nSummary:")
        print(f"  • Total chunks: {len(tiny_chunks)}")
        print(f"  • API calls: {api_call_count}")
        print(f"  • Total time: {total_time*1000:.1f}ms")
        print(f"  • Avg call duration: {statistics.mean(call_durations)*1000:.1f}ms")
        print(f"  • Avg inter-call gap: {statistics.mean(inter_call_gaps)*1000:.1f}ms")

        return StreamingTest(
            name="Unbuffered (many tiny chunks)",
            chunks_sent=len(tiny_chunks),
            api_calls_made=api_call_count,
            total_time=total_time,
            call_durations=call_durations,
            inter_call_gaps=inter_call_gaps,
        )

    async def test_buffered_streaming(
        self, message_prefix: str, buffer_time: float = 0.1
    ) -> StreamingTest:
        """Test sending fewer, larger chunks (simulates buffering).

        This simulates the PROPOSED buffered behavior where chunks
        are accumulated for ~100ms before sending.

        Args:
            message_prefix: Prefix for test message
            buffer_time: How long to buffer before flushing (seconds)

        Returns:
            StreamingTest with results
        """
        print(f"\n{'='*60}")
        print(f"TEST: Buffered ({buffer_time*1000:.0f}ms buffer)")
        print(f"{'='*60}")

        # Same content, but we'll buffer it
        tiny_chunks = [
            "Let ", "me ", "help ", "you ", "with ", "that. ",
            "This ", "is ", "a ", "test ", "of ", "buffered ",
            "streaming. ", "Chunks ", "are ", "accumulated ", "for ",
            f"{buffer_time*1000:.0f}ms ", "before ", "sending. ",
            "This ", "should ", "reduce ", "API ", "calls. ",
        ]

        team_id = await self._get_team_id()
        bot_user_id = await self._get_bot_user_id()
        call_durations = []
        inter_call_gaps = []

        # Start stream
        start_time = time.time()
        print(f"[{0:.3f}s] Starting stream...")

        # Post a regular message first to get a thread_ts
        initial_msg = await self.client.chat_postMessage(
            channel=self.test_channel,
            text="🧪 Starting streaming test..."
        )
        thread_ts = initial_msg["ts"]

        start_response = await self.client.chat_startStream(
            channel=self.test_channel,
            recipient_team_id=team_id,
            recipient_user_id=bot_user_id,  # Bot's own user ID
            thread_ts=thread_ts,  # Reply in thread
        )
        stream_ts = start_response["ts"]
        print(f"[{time.time() - start_time:.3f}s] Stream started: {stream_ts}")

        # Simulate buffering
        buffer = []
        last_flush_time = time.time()
        last_call_start = None
        api_call_count = 0
        chunks_processed = 0

        for i, chunk in enumerate(tiny_chunks):
            # Simulate chunk arriving (very small delay to be realistic)
            await asyncio.sleep(0.01)  # 10ms between chunks from LangGraph

            buffer.append(chunk)
            chunks_processed += 1

            # Check if should flush (time-based)
            time_since_flush = time.time() - last_flush_time
            should_flush = time_since_flush >= buffer_time

            if should_flush and buffer:
                combined = "".join(buffer)
                call_start = time.time()

                # Record gap
                if last_call_start is not None:
                    gap = call_start - last_call_start
                    inter_call_gaps.append(gap)

                # Send buffered content
                await self.client.chat_appendStream(
                    channel=self.test_channel,
                    ts=stream_ts,
                    markdown_text=combined,
                )
                api_call_count += 1

                call_duration = time.time() - call_start
                call_durations.append(call_duration)

                print(f"[{time.time() - start_time:.3f}s] Flush #{api_call_count}: "
                      f"{len(buffer)} chunks, {len(combined)} chars "
                      f"(call took {call_duration*1000:.1f}ms)")

                buffer.clear()
                last_flush_time = time.time()
                last_call_start = call_start

        # Final flush
        if buffer:
            combined = "".join(buffer)
            call_start = time.time()

            if last_call_start is not None:
                gap = call_start - last_call_start
                inter_call_gaps.append(gap)

            await self.client.chat_appendStream(
                channel=self.test_channel,
                ts=stream_ts,
                markdown_text=combined,
            )
            api_call_count += 1

            call_duration = time.time() - call_start
            call_durations.append(call_duration)

            print(f"[{time.time() - start_time:.3f}s] Final flush: "
                  f"{len(buffer)} chunks, {len(combined)} chars")

            last_call_start = call_start

        # Stop stream
        await self.client.chat_stopStream(
            channel=self.test_channel,
            ts=stream_ts,
        )

        total_time = time.time() - start_time
        print(f"[{total_time:.3f}s] Stream stopped")
        print(f"\nSummary:")
        print(f"  • Total chunks: {chunks_processed}")
        print(f"  • API calls: {api_call_count}")
        print(f"  • Total time: {total_time*1000:.1f}ms")
        print(f"  • Avg call duration: {statistics.mean(call_durations)*1000:.1f}ms")
        if inter_call_gaps:
            print(f"  • Avg inter-call gap: {statistics.mean(inter_call_gaps)*1000:.1f}ms")

        return StreamingTest(
            name=f"Buffered ({buffer_time*1000:.0f}ms)",
            chunks_sent=chunks_processed,
            api_calls_made=api_call_count,
            total_time=total_time,
            call_durations=call_durations,
            inter_call_gaps=inter_call_gaps,
        )

    def print_comparison(self, tests: List[StreamingTest]):
        """Print comparison of test results."""
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}\n")

        print(f"{'Test':<30} {'API Calls':<12} {'Total Time':<15} {'Avg Call':<15}")
        print("-" * 80)

        for test in tests:
            avg_call = statistics.mean(test.call_durations) * 1000
            print(
                f"{test.name:<30} "
                f"{test.api_calls_made:<12} "
                f"{test.total_time*1000:>10.1f} ms   "
                f"{avg_call:>10.1f} ms"
            )

        print()

        if len(tests) >= 2:
            unbuffered = tests[0]
            buffered = tests[1]

            api_reduction = (
                1 - buffered.api_calls_made / unbuffered.api_calls_made
            ) * 100

            time_diff = (buffered.total_time - unbuffered.total_time) * 1000

            print("ANALYSIS:")
            print(f"  • API call reduction: {api_reduction:.1f}%")
            print(f"  • Time difference: {time_diff:+.1f}ms")

            if api_reduction > 50:
                print(f"    ✓ Buffering significantly reduces API calls")
            else:
                print(f"    ⚠ API call reduction is modest")

            if time_diff < 0:
                print(f"    ✓ Buffering is FASTER (by {abs(time_diff):.0f}ms)")
            elif time_diff < 100:
                print(f"    ✓ Buffering is only slightly slower ({time_diff:.0f}ms)")
            else:
                print(f"    ⚠ Buffering adds noticeable latency ({time_diff:.0f}ms)")

            # Check if call durations increase (indicating queuing)
            unbuffered_avg_call = statistics.mean(unbuffered.call_durations)
            buffered_avg_call = statistics.mean(buffered.call_durations)

            print(f"\n  Average API call duration:")
            print(f"    • Unbuffered: {unbuffered_avg_call*1000:.1f}ms")
            print(f"    • Buffered: {buffered_avg_call*1000:.1f}ms")

            if unbuffered_avg_call > buffered_avg_call * 1.2:
                print(f"    ⚠ Unbuffered calls are SLOWER - suggests Slack queuing!")
                print(f"      This supports your hypothesis that rapid calls queue up.")
            else:
                print(f"    ℹ Call durations are similar - queuing may not be the issue")


async def main():
    """Run benchmarks."""
    # Get credentials from environment
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    test_channel = os.getenv("SLACK_TEST_CHANNEL")

    if not slack_token:
        print("ERROR: SLACK_BOT_TOKEN environment variable not set")
        print("Get your token from: https://api.slack.com/apps")
        return

    if not test_channel:
        print("ERROR: SLACK_TEST_CHANNEL environment variable not set")
        print("Right-click a channel in Slack → Copy → Copy link")
        print("The channel ID is the last part: C01234567890")
        return

    benchmark = SlackStreamingBenchmark(slack_token, test_channel)

    print("Slack Streaming Latency Benchmark")
    print("This will send test messages to your Slack channel")
    print(f"Channel: {test_channel}")
    print()

    input("Press Enter to start tests...")

    tests = []

    # Test 1: Unbuffered (many rapid calls)
    unbuffered_result = await benchmark.test_unbuffered_streaming("Test 1")
    tests.append(unbuffered_result)

    await asyncio.sleep(2)  # Pause between tests

    # Test 2: Buffered (fewer, larger calls)
    buffered_result = await benchmark.test_buffered_streaming("Test 2", buffer_time=0.1)
    tests.append(buffered_result)

    # Print comparison
    benchmark.print_comparison(tests)


if __name__ == "__main__":
    asyncio.run(main())
