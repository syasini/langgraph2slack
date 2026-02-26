"""
Test whether Slack allows multiple table blocks in the same message,
and whether behavior differs between chat.postMessage and chat.update
on different message types (regular vs streaming-created).

Run from examples/plant_bot/:
    uv run python ../../experiments/test_multi_table_slack.py

Results tell us whether to convert ALL markdown tables to block-kit tables
or cap at one.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "examples/plant_bot/.env")

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
# Use a test channel — DM the bot to itself works too.
# Override with TEST_CHANNEL env var, otherwise we'll find the bot's own DM.
TEST_CHANNEL = os.environ.get("TEST_CHANNEL")

TWO_TABLE_BLOCKS = [
    {
        "type": "table",
        "column_settings": [{"is_wrapped": True}, {"is_wrapped": True}],
        "rows": [
            [{"type": "raw_text", "text": "Plant"}, {"type": "raw_text", "text": "Water Freq"}],
            [{"type": "raw_text", "text": "Monstera"}, {"type": "raw_text", "text": "Weekly"}],
            [{"type": "raw_text", "text": "Cactus"}, {"type": "raw_text", "text": "Monthly"}],
        ],
    },
    {
        "type": "section",
        "text": {"type": "mrkdwn", "text": "Here is the light requirements table:"},
    },
    {
        "type": "table",
        "column_settings": [{"is_wrapped": True}, {"is_wrapped": True}],
        "rows": [
            [{"type": "raw_text", "text": "Plant"}, {"type": "raw_text", "text": "Light"}],
            [{"type": "raw_text", "text": "Monstera"}, {"type": "raw_text", "text": "Indirect"}],
            [{"type": "raw_text", "text": "Cactus"}, {"type": "raw_text", "text": "Full sun"}],
        ],
    },
]


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


async def get_bot_dm_channel(client) -> str:
    """Open a DM with the bot itself (self-DM) to use as test channel."""
    auth = await client.auth_test()
    bot_user_id = auth["user_id"]
    dm = await client.conversations_open(users=[bot_user_id])
    return dm["channel"]["id"]


async def test_post_message_two_tables(client, channel: str):
    """Test 1: chat.postMessage with two table blocks."""
    sep("TEST 1: chat.postMessage with two table blocks")
    try:
        result = await client.chat_postMessage(
            channel=channel,
            text="Two-table test (postMessage)",
            blocks=TWO_TABLE_BLOCKS,
        )
        print(f"  SUCCESS  ts={result['ts']}")
        return result["ts"]
    except Exception as e:
        print(f"  FAIL  {e}")
        return None


async def test_update_regular_message_two_tables(client, channel: str):
    """Test 2: chat.update on a regular message with two table blocks."""
    sep("TEST 2: chat.update on a regular say()-style message → two tables")
    # First post a placeholder
    placeholder = await client.chat_postMessage(
        channel=channel,
        text="Thinking...",
    )
    ts = placeholder["ts"]
    print(f"  Placeholder posted: ts={ts}")
    try:
        await client.chat_update(
            channel=channel,
            ts=ts,
            text="Two-table test (update on regular msg)",
            blocks=TWO_TABLE_BLOCKS,
        )
        print(f"  SUCCESS  Updated regular message with two table blocks")
    except Exception as e:
        print(f"  FAIL  {e}")


async def test_update_streaming_message_two_tables(client, channel: str):
    """Test 3: chat.update on a streaming-created message with two table blocks."""
    sep("TEST 3: chat.update on a chat_startStream message → two tables")
    try:
        # chat_startStream requires a thread_ts — post a seed message first
        seed = await client.chat_postMessage(channel=channel, text="(test thread seed)")
        thread_ts = seed["ts"]
        print(f"  Seed message ts={thread_ts}")

        stream = await client.chat_startStream(
            channel=channel,
            thread_ts=thread_ts,
            assistant_color="#36a64f",
        )
        ts = stream["ts"]
        print(f"  Stream started: ts={ts}")

        # Append some text to simulate streaming
        await client.chat_appendStream(channel=channel, ts=ts, markdown_text="Plant water frequencies: ")
        await client.chat_appendStream(channel=channel, ts=ts, markdown_text="Monstera weekly, Cactus monthly.")
        await client.chat_stopStream(channel=channel, ts=ts)
        print(f"  Stream stopped")

        # Now try chat_update with two table blocks
        import asyncio as _asyncio
        await _asyncio.sleep(0.5)
        await client.chat_update(
            channel=channel,
            ts=ts,
            text="Two-table test (update on streaming msg)",
            blocks=TWO_TABLE_BLOCKS,
        )
        print(f"  SUCCESS  Updated streaming message with two table blocks")
    except Exception as e:
        print(f"  FAIL  {e}")


async def main():
    from slack_sdk.web.async_client import AsyncWebClient

    client = AsyncWebClient(token=SLACK_BOT_TOKEN)

    if TEST_CHANNEL:
        channel = TEST_CHANNEL
        print(f"Using TEST_CHANNEL={channel}")
    else:
        channel = await get_bot_dm_channel(client)
        print(f"Using bot self-DM channel: {channel}")

    ts1 = await test_post_message_two_tables(client, channel)
    await test_update_regular_message_two_tables(client, channel)
    await test_update_streaming_message_two_tables(client, channel)

    print("\n" + "="*60)
    print("  DONE — check results above")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
