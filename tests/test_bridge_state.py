"""Durable bridge state: feedback, tool-call details, joined threads and duplicate
suppression survive restarts, expire on their own, and degrade gracefully without a store.

Restarts are simulated by building a second SlackBot against the same fake deployment.
"""

import logging

import pytest

from langgraph2slack import InMemoryStateStore
from langgraph2slack.state import DeploymentStateStore
from langgraph2slack.utils import TOOL_CALL_DETAILS_ACTION_ID

from .harness import AGENT_USER_ID

MENTION = f"<@{AGENT_USER_ID}>"

TOOL_CALL_REPLY = [
    {
        "type": "ai",
        "content": "",
        "tool_calls": [{"id": "call_1", "name": "search_logs", "args": {"query": "timeout"}}],
    },
    {"type": "tool", "tool_call_id": "call_1", "name": "search_logs", "content": "3 matches"},
    {"type": "AIMessageChunk", "content": "Found 3 timeouts."},
]


@pytest.fixture(params=[True, False], ids=["streaming", "non-streaming"])
def streaming(request):
    return request.param


async def reply_to_dm(bridge, text="hello"):
    await bridge.send_event(bridge.message("D0DM", text, channel_type="im"))
    replies = bridge.agent_messages("D0DM")
    assert replies, "the agent did not reply"
    return replies[-1]["ts"]


# ============================================================================
# Feedback and tool-call details survive a restart
# ============================================================================


async def test_feedback_works_after_a_restart(make_bridge, streaming):
    first = make_bridge(streaming=streaming, show_feedback_buttons=True)
    reply_ts = await reply_to_dm(first)
    run_id = first.runs[0].run_id

    restarted = make_bridge(streaming=streaming, show_feedback_buttons=True)
    await restarted.click("feedback", "positive", channel="D0DM", message_ts=reply_ts)

    assert len(restarted.feedback_submitted) == 1
    assert restarted.feedback_submitted[0]["run_id"] == run_id
    assert restarted.feedback_submitted[0]["score"] == 1.0


async def test_feedback_on_unknown_message_submits_nothing(make_bridge):
    bridge = make_bridge(show_feedback_buttons=True)

    await bridge.click("feedback", "positive", channel="D0DM", message_ts="1.000001")

    assert bridge.feedback_submitted == []


async def test_tool_call_details_open_after_a_restart(make_bridge, deployment, streaming):
    deployment.reply = TOOL_CALL_REPLY
    first = make_bridge(streaming=streaming, show_tool_calls=True)
    await reply_to_dm(first)
    (plan,) = [m for m in first.slack.calls_to("chat.postMessage") if "blocks" in m.args]
    plan_ts = next(m["ts"] for m in first.agent_messages("D0DM") if m["text"] == plan.args["text"])

    restarted = make_bridge(streaming=streaming, show_tool_calls=True)
    await restarted.click(TOOL_CALL_DETAILS_ACTION_ID, plan_ts, channel="D0DM", message_ts=plan_ts)

    (opened,) = restarted.slack.calls_to("views.open")
    rendered = str(opened.args["view"])
    assert "search_logs" in rendered
    assert "3 matches" in rendered
    assert "no longer available" not in rendered


# ============================================================================
# Joined threads are remembered
# ============================================================================


async def test_joined_thread_is_remembered_without_reading_slack_history(make_bridge):
    bridge = make_bridge()
    await bridge.send_event(
        bridge.message("C0CHAN", f"{MENTION} hi", channel_type="channel", ts="1600000000.000001")
    )

    restarted = make_bridge()
    restarted.slack.calls.clear()
    await restarted.send_event(
        restarted.message(
            "C0CHAN", "follow-up", channel_type="channel", thread_ts="1600000000.000001"
        )
    )

    assert [r.text for r in restarted.runs] == ["hi", "follow-up"]
    assert restarted.slack.calls_to("conversations.replies") == []


async def test_proactive_message_thread_counts_as_joined(make_bridge):
    bridge = make_bridge()
    ts = await bridge.bot.send_proactive_message("C0CHAN", "Deploy finished")
    bridge.slack.calls.clear()

    await bridge.send_event(
        bridge.message("C0CHAN", "thanks, any errors?", channel_type="channel", thread_ts=ts)
    )

    assert [r.text for r in bridge.runs] == ["thanks, any errors?"]
    assert bridge.slack.calls_to("conversations.replies") == []


async def test_thread_found_in_slack_history_is_remembered(make_bridge):
    bridge = make_bridge()
    bridge.slack.add_message("C0CHAN", {"ts": "1600000000.000001", "user": "U0HUMAN", "text": "p"})
    bridge.slack.add_message(
        "C0CHAN",
        {"user": AGENT_USER_ID, "text": "earlier reply", "thread_ts": "1600000000.000001"},
    )
    reply = dict(channel_type="channel", thread_ts="1600000000.000001")

    await bridge.send_event(bridge.message("C0CHAN", "one", **reply))
    await bridge.send_event(bridge.message("C0CHAN", "two", **reply))

    assert [r.text for r in bridge.runs] == ["one", "two"]
    assert len(bridge.slack.calls_to("conversations.replies")) == 1


# ============================================================================
# Duplicate suppression
# ============================================================================


async def test_redelivered_event_starts_no_second_run(make_bridge):
    bridge = make_bridge()
    event = bridge.message("D0DM", "hello", channel_type="im", ts="1600000000.000001")

    await bridge.send_event(event, event_id="Ev123")
    await bridge.send_event(event, event_id="Ev123")

    assert len(bridge.runs) == 1


async def test_redelivery_to_a_restarted_bridge_starts_no_second_run(make_bridge):
    event = {
        "type": "message",
        "channel": "D0DM",
        "channel_type": "im",
        "user": "U0HUMAN",
        "text": "hello",
        "ts": "1600000000.000001",
    }

    await make_bridge().send_event(event, event_id="Ev123")
    restarted = make_bridge()
    await restarted.send_event(event, event_id="Ev123")

    assert len(restarted.runs) == 1


async def test_distinct_events_each_start_a_run(make_bridge):
    bridge = make_bridge()

    await bridge.send_event(bridge.message("D0DM", "one", channel_type="im"), event_id="Ev1")
    await bridge.send_event(bridge.message("D0DM", "two", channel_type="im"), event_id="Ev2")

    assert [r.text for r in bridge.runs] == ["one", "two"]


async def test_duplicate_records_expire(make_bridge, deployment):
    bridge = make_bridge()
    event = bridge.message("D0DM", "hello", channel_type="im", ts="1600000000.000001")
    await bridge.send_event(event, event_id="Ev123")

    deployment.store.advance(days=2)
    await bridge.send_event(event, event_id="Ev123")

    assert len(bridge.runs) == 2


async def test_each_kind_of_state_has_its_own_ttl(make_bridge, deployment):
    deployment.reply = TOOL_CALL_REPLY
    bridge = make_bridge(show_tool_calls=True)
    await bridge.send_event(
        bridge.message("C0CHAN", f"{MENTION} hi", channel_type="channel"), event_id="Ev1"
    )

    ttls = {
        name: {item["ttl"] for item in deployment.store.namespace(name).values()}
        for name in ["feedback", "tool_calls", "joined_conversations", "duplicate_records"]
    }
    assert all(len(t) == 1 and None not in t for t in ttls.values()), ttls
    assert ttls["duplicate_records"] != ttls["joined_conversations"]


# ============================================================================
# Without a store, and with a custom store
# ============================================================================


async def test_without_a_store_the_bridge_falls_back_to_memory_with_a_warning(
    make_bridge, deployment, caplog
):
    deployment.store.reachable = False
    bridge = make_bridge(show_feedback_buttons=True)

    with caplog.at_level(logging.WARNING):
        reply_ts = await reply_to_dm(bridge)
        await bridge.click("feedback", "negative", channel="D0DM", message_ts=reply_ts)

    assert bridge.feedback_submitted[0]["run_id"] == bridge.runs[0].run_id
    assert any("in-memory" in r.getMessage() for r in caplog.records)


async def test_without_a_store_redeliveries_are_still_suppressed(make_bridge, deployment):
    deployment.store.reachable = False
    bridge = make_bridge()
    event = bridge.message("D0DM", "hello", channel_type="im", ts="1600000000.000001")

    await bridge.send_event(event, event_id="Ev123")
    await bridge.send_event(event, event_id="Ev123")

    assert len(bridge.runs) == 1


async def test_custom_state_store_is_used(make_bridge, deployment):
    custom = InMemoryStateStore()
    first = make_bridge(state_store=custom, show_feedback_buttons=True)
    reply_ts = await reply_to_dm(first)

    restarted = make_bridge(state_store=custom, show_feedback_buttons=True)
    await restarted.click("feedback", "positive", channel="D0DM", message_ts=reply_ts)

    assert restarted.feedback_submitted[0]["run_id"] == first.runs[0].run_id
    assert deployment.store.items == {}


# ============================================================================
# The in-memory store itself
# ============================================================================


async def test_in_memory_store_expires_items():
    now = [0.0]
    store = InMemoryStateStore(clock=lambda: now[0])
    await store.put("ns", "k", {"v": 1}, ttl=10)

    assert await store.get("ns", "k") == {"v": 1}
    now[0] = 10
    assert await store.get("ns", "k") is None


async def test_in_memory_store_is_bounded():
    store = InMemoryStateStore(max_items=2)
    for key in ["a", "b", "c"]:
        await store.put("ns", key, {})

    assert await store.get("ns", "a") is None
    assert await store.get("ns", "c") == {}


# ============================================================================
# Review follow-ups: keys, scope, TTL support, probe retry
# ============================================================================


async def test_redelivery_without_event_id_is_suppressed_by_message_and_kind(make_bridge):
    bridge = make_bridge()
    event = bridge.message("D0DM", "hello", channel_type="im", ts="1600000000.000001")

    await bridge.send_event(event, event_id="")
    await bridge.send_event(event, event_id="")

    assert len(bridge.runs) == 1


async def test_top_level_reply_starts_a_joined_thread(make_bridge):
    bridge = make_bridge(reply_in_thread=False)
    await bridge.send_event(bridge.message("C0CHAN", f"{MENTION} hi", channel_type="channel"))
    reply_ts = bridge.agent_messages("C0CHAN")[-1]["ts"]
    bridge.slack.calls.clear()

    await bridge.send_event(
        bridge.message("C0CHAN", "about that", channel_type="channel", thread_ts=reply_ts)
    )

    assert [r.text for r in bridge.runs] == ["hi", "about that"]
    assert bridge.slack.calls_to("conversations.replies") == []


async def test_bridges_for_different_assistants_keep_separate_state(make_bridge):
    first = make_bridge(assistant_id="triage-agent")
    await first.send_event(
        first.message("C0CHAN", f"{MENTION} hi", channel_type="channel", ts="1600000000.000001")
    )

    other = make_bridge(assistant_id="review-agent")
    other.slack.fail("conversations.replies")
    await other.send_event(
        other.message("C0CHAN", "follow-up", channel_type="channel", thread_ts="1600000000.000001")
    )

    assert [r.assistant_id for r in other.runs] == ["triage-agent"]


async def test_bridge_records_are_never_indexed(make_bridge, deployment):
    bridge = make_bridge(show_feedback_buttons=True)
    await reply_to_dm(bridge)

    assert deployment.store.items
    assert all(item["index"] is False for item in deployment.store.items.values())


async def test_store_without_ttl_support_is_used_without_expiry(make_bridge, deployment, caplog):
    deployment.store.supports_ttl = False
    first = make_bridge(show_feedback_buttons=True)

    with caplog.at_level(logging.WARNING):
        reply_ts = await reply_to_dm(first)

    restarted = make_bridge(show_feedback_buttons=True)
    await restarted.click("feedback", "positive", channel="D0DM", message_ts=reply_ts)
    assert restarted.feedback_submitted[0]["run_id"] == first.runs[0].run_id
    assert any("TTL" in r.getMessage() for r in caplog.records)


async def test_unreachable_store_is_probed_again_later(deployment):
    now = [0.0]
    store = DeploymentStateStore(deployment, retry_after=300, clock=lambda: now[0])
    deployment.store.reachable = False
    await store.put("ns", "k", {"in": "memory"})

    deployment.store.reachable = True
    now[0] = 100
    await store.put("ns", "k2", {})
    assert deployment.store.namespace("ns") == {}

    now[0] = 301
    await store.put("ns", "k3", {"in": "store"})
    assert list(deployment.store.namespace("ns")) == ["k3"]
