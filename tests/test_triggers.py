"""Trigger gate behavior, tested through the bridge's public seam.

Each test delivers a raw Slack payload through Bolt and checks whether a run started,
and with what input, metadata and config.
"""

import pytest

from .harness import AGENT_BOT_ID, AGENT_USER_ID

MENTION = f"<@{AGENT_USER_ID}>"
HUMAN = "U0HUMAN"


@pytest.fixture(params=[True, False], ids=["streaming", "non-streaming"])
def bridge(request, make_bridge):
    return make_bridge(streaming=request.param)


@pytest.fixture
def streaming_bridge(make_bridge):
    return make_bridge(streaming=True)


def agent_has_posted_in(slack, channel, thread_ts):
    slack.add_message(channel, {"ts": thread_ts, "user": HUMAN, "text": "parent"})
    slack.add_message(
        channel,
        {"user": AGENT_USER_ID, "bot_id": AGENT_BOT_ID, "text": "hi", "thread_ts": thread_ts},
    )


# ============================================================================
# Which Slack messages are triggers
# ============================================================================


async def test_human_dm_starts_a_run(bridge):
    await bridge.send_event(bridge.message("D0DM", "hello agent", channel_type="im"))

    assert len(bridge.runs) == 1
    assert bridge.runs[0].text == "hello agent"


@pytest.mark.parametrize(
    "text",
    [f"{MENTION} help me", f"help me {MENTION}", f"please {MENTION} help me"],
    ids=["start", "end", "middle"],
)
async def test_channel_mention_starts_a_run_with_the_mention_stripped(streaming_bridge, text):
    await streaming_bridge.send_event(
        streaming_bridge.message("C0CHAN", text, channel_type="channel")
    )

    assert len(streaming_bridge.runs) == 1
    assert MENTION not in streaming_bridge.runs[0].text
    assert "help me" in streaming_bridge.runs[0].text


async def test_private_channel_mention_starts_a_run(streaming_bridge):
    await streaming_bridge.send_event(
        streaming_bridge.message("G0PRIV", f"{MENTION} hi", channel_type="group")
    )

    assert len(streaming_bridge.runs) == 1


@pytest.mark.parametrize(
    "text",
    ["just chatting", "", "<@U0SOMEONE> can you look?"],
    ids=["no-mention", "empty", "other-user-mentioned"],
)
async def test_channel_message_without_agent_mention_is_ignored(streaming_bridge, text):
    await streaming_bridge.send_event(
        streaming_bridge.message("C0CHAN", text, channel_type="channel")
    )

    assert streaming_bridge.runs == []


async def test_private_channel_message_without_mention_is_ignored(streaming_bridge):
    await streaming_bridge.send_event(
        streaming_bridge.message("G0PRIV", "hello team", channel_type="group")
    )

    assert streaming_bridge.runs == []


async def test_message_without_a_user_is_ignored(streaming_bridge):
    await streaming_bridge.send_event(
        {"type": "message", "channel": "D0DM", "channel_type": "im", "text": "system"}
    )

    assert streaming_bridge.runs == []


async def test_bot_message_is_ignored_by_default(streaming_bridge):
    await streaming_bridge.send_event(
        {
            "type": "message",
            "subtype": "bot_message",
            "channel": "C0CHAN",
            "channel_type": "channel",
            "bot_id": "B0CI",
            "text": f"{MENTION} build failed",
        }
    )

    assert streaming_bridge.runs == []


async def test_agent_never_triggers_on_its_own_message(streaming_bridge):
    await streaming_bridge.send_event(
        {
            "type": "message",
            "channel": "D0DM",
            "channel_type": "im",
            "user": AGENT_USER_ID,
            "bot_id": AGENT_BOT_ID,
            "text": f"{MENTION} talking to myself",
        }
    )

    assert streaming_bridge.runs == []


@pytest.mark.parametrize(
    "subtype", ["channel_join", "message_deleted", "channel_topic", "me_message"]
)
async def test_unhandled_message_subtypes_are_ignored(streaming_bridge, subtype):
    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", f"{MENTION} has joined the channel", channel_type="channel", subtype=subtype
        )
    )

    assert streaming_bridge.runs == []


@pytest.mark.parametrize("subtype", ["file_share", "thread_broadcast"])
async def test_handled_human_subtypes_trigger(streaming_bridge, subtype):
    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", f"{MENTION} look at this", channel_type="channel", subtype=subtype
        )
    )

    assert len(streaming_bridge.runs) == 1


async def test_bridge_identity_is_looked_up_once(streaming_bridge):
    await streaming_bridge.send_event(streaming_bridge.message("D0DM", "one", channel_type="im"))
    calls_after_first = len(streaming_bridge.slack.calls_to("auth.test"))

    await streaming_bridge.send_event(streaming_bridge.message("D0DM", "two", channel_type="im"))

    assert len(streaming_bridge.runs) == 2
    assert len(streaming_bridge.slack.calls_to("auth.test")) == calls_after_first


# ============================================================================
# Joined threads
# ============================================================================


async def test_human_reply_in_joined_thread_triggers_without_mention(streaming_bridge):
    agent_has_posted_in(streaming_bridge.slack, "C0CHAN", "1600000000.000001")

    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", "follow-up", channel_type="channel", thread_ts="1600000000.000001"
        )
    )

    assert len(streaming_bridge.runs) == 1
    assert streaming_bridge.runs[0].text == "follow-up"


async def test_reply_in_thread_the_agent_has_not_joined_is_ignored(streaming_bridge):
    slack = streaming_bridge.slack
    slack.add_message("C0CHAN", {"ts": "1600000000.000001", "user": HUMAN, "text": "parent"})
    slack.add_message(
        "C0CHAN",
        {"bot_id": "B0OTHER", "user": "U0OTHERBOT", "text": "hi", "thread_ts": "1600000000.000001"},
    )

    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", "follow-up", channel_type="channel", thread_ts="1600000000.000001"
        )
    )

    assert streaming_bridge.runs == []


async def test_mention_in_unjoined_thread_triggers(streaming_bridge):
    streaming_bridge.slack.add_message(
        "C0CHAN", {"ts": "1600000000.000001", "user": HUMAN, "text": "parent"}
    )

    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN",
            f"{MENTION} take a look",
            channel_type="channel",
            thread_ts="1600000000.000001",
        )
    )

    assert len(streaming_bridge.runs) == 1


async def test_failed_thread_history_lookup_falls_back_to_mention_rule(streaming_bridge):
    streaming_bridge.slack.fail("conversations.replies")

    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", "follow-up", channel_type="channel", thread_ts="1600000000.000001"
        )
    )
    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", f"{MENTION} hi", channel_type="channel", thread_ts="1600000000.000001"
        )
    )

    assert [r.text for r in streaming_bridge.runs] == ["hi"]


# ============================================================================
# Provenance
# ============================================================================


@pytest.mark.parametrize(
    "event, kind",
    [
        ({"channel": "D0DM", "channel_type": "im", "text": "hello"}, "direct_message"),
        ({"channel": "C0CHAN", "channel_type": "channel", "text": f"{MENTION} hi"}, "mention"),
    ],
    ids=["dm", "mention"],
)
async def test_run_metadata_carries_human_provenance(bridge, event, kind):
    await bridge.send_event({"type": "message", "user": HUMAN, **event})

    metadata = bridge.runs[0].metadata
    assert metadata["slack_trigger_kind"] == kind
    assert metadata["slack_author_type"] == "human"
    assert metadata["slack_author_id"] == HUMAN
    assert metadata["slack_requester_id"] == HUMAN
    assert metadata["slack_bot_id"] is None
    assert metadata["slack_app_id"] is None
    # Existing keys are unchanged
    assert metadata["slack_user_id"] == HUMAN
    assert metadata["slack_channel_id"] == event["channel"]


async def test_joined_thread_trigger_kind(streaming_bridge):
    agent_has_posted_in(streaming_bridge.slack, "C0CHAN", "1600000000.000001")

    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", "follow-up", channel_type="channel", thread_ts="1600000000.000001"
        )
    )

    assert streaming_bridge.runs[0].metadata["slack_trigger_kind"] == "joined_thread"
    assert streaming_bridge.runs[0].slack_config["trigger_kind"] == "joined_thread"


async def test_mention_in_joined_thread_counts_as_mention(streaming_bridge):
    agent_has_posted_in(streaming_bridge.slack, "C0CHAN", "1600000000.000001")

    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", f"{MENTION} again", channel_type="channel", thread_ts="1600000000.000001"
        )
    )

    assert streaming_bridge.runs[0].metadata["slack_trigger_kind"] == "mention"


async def test_run_config_always_carries_slack_provenance(bridge):
    await bridge.send_event(
        bridge.message(
            "C0CHAN",
            f"{MENTION} hi",
            channel_type="channel",
            ts="1600000000.000009",
            thread_ts="1600000000.000001",
        )
    )

    assert bridge.runs[0].slack_config == {
        "trigger_kind": "mention",
        "author_type": "human",
        "author_id": HUMAN,
        "bot_id": None,
        "app_id": None,
        "requester_id": HUMAN,
        "channel_id": "C0CHAN",
        "channel_type": "channel",
        "message_ts": "1600000000.000009",
        "thread_ts": "1600000000.000001",
        "conversation_ts": "1600000000.000001",
    }


async def test_top_level_trigger_starts_a_conversation_rooted_at_itself(streaming_bridge):
    await streaming_bridge.send_event(
        streaming_bridge.message(
            "C0CHAN", f"{MENTION} hi", channel_type="channel", ts="1600000000.000009"
        )
    )

    slack_config = streaming_bridge.runs[0].slack_config
    assert slack_config["thread_ts"] is None
    assert slack_config["conversation_ts"] == "1600000000.000009"


async def test_provenance_is_in_config_even_when_metadata_is_off(make_bridge):
    bridge = make_bridge(include_metadata=False)

    await bridge.send_event(bridge.message("D0DM", "hello", channel_type="im"))

    assert bridge.runs[0].metadata == {}
    assert bridge.runs[0].slack_config["requester_id"] == HUMAN


async def test_config_transformers_see_and_can_change_provenance(bridge):
    seen = {}

    @bridge.bot.transform_config
    async def tweak(config, context):
        seen.update(config["configurable"]["slack"])
        config["configurable"]["slack"]["author_id"] = "redacted"
        config["configurable"]["repo"] = "example/repo"
        return config

    await bridge.send_event(bridge.message("D0DM", "hello", channel_type="im"))

    assert seen["author_id"] == HUMAN
    configurable = bridge.runs[0].config["configurable"]
    assert configurable["slack"]["author_id"] == "redacted"
    assert configurable["repo"] == "example/repo"


async def test_metadata_transformers_see_provenance_on_the_context(bridge):
    @bridge.bot.transform_metadata
    async def custom(context):
        return {
            "kind": context.trigger_kind,
            "author_type": context.author_type,
            "requester": context.requester_id,
        }

    await bridge.send_event(bridge.message("D0DM", "hello", channel_type="im"))

    assert bridge.runs[0].metadata == {
        "kind": "direct_message",
        "author_type": "human",
        "requester": HUMAN,
    }
