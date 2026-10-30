"""The "Ask agent about this" message shortcut starts a run on any message."""

import pytest

from .harness import AGENT_APP_ID, AGENT_USER_ID, TEAM_ID

HUMAN = "U0HUMAN"
THREAD = "1700000100.000001"
ALERT = {
    "type": "message",
    "subtype": "bot_message",
    "bot_id": "B0ALERTS",
    "username": "alerts",
    "ts": THREAD,
    "text": "Disk 92% full on db-1",
}
DEFAULT_INSTRUCTION = "Take a look at this message and help with it."


@pytest.fixture(params=[True, False], ids=["streaming", "non-streaming"])
def streaming(request):
    return request.param


def shortcut(
    message,
    *,
    callback_id="ask_agent",
    user=HUMAN,
    trigger_id="13345224609.738474920",
    action_ts="1700000500.000001",
    channel="C0CHAN",
):
    return {
        "type": "message_action",
        "callback_id": callback_id,
        "trigger_id": trigger_id,
        "action_ts": action_ts,
        "team": {"id": TEAM_ID},
        "api_app_id": AGENT_APP_ID,
        "user": {"id": user, "team_id": TEAM_ID},
        "channel": {"id": channel, "name": "general"},
        "message_ts": message["ts"],
        "message": message,
        "response_url": "https://hooks.slack.com/app/T0TEAM/1/abc",
    }


async def test_shortcut_on_a_bot_post_starts_a_run_in_its_thread(make_bridge, slack, streaming):
    bridge = make_bridge(streaming=streaming)  # bot triggers stay off: a human asked
    slack.add_message("C0CHAN", ALERT)

    status = await bridge.send_interaction(shortcut(ALERT))

    assert status == 200
    (run,) = bridge.runs
    assert run.request == DEFAULT_INSTRUCTION
    assert [m.text for m in run.backfill] == ["Disk 92% full on db-1"]
    assert run.slack_config["trigger_kind"] == "shortcut"
    # The shortcut is a human's action; the alert's own author shows in the backfill
    assert run.slack_config["requester_id"] == HUMAN
    assert run.slack_config["author_type"] == "human"
    assert run.slack_config["author_id"] == HUMAN
    assert run.backfill[0].author_type == "bot"
    assert run.backfill[0].author_id == "B0ALERTS"
    assert run.slack_config["conversation_ts"] == THREAD
    assert run.text.splitlines()[-2].startswith(
        f"Request from {HUMAN} ({HUMAN}, human) via message shortcut"
    )
    assert bridge.agent_messages("C0CHAN")[-1]["thread_ts"] == THREAD


async def test_shortcut_reply_streams_to_the_person_who_used_it(make_bridge, slack):
    bridge = make_bridge(streaming=True)
    slack.add_message("C0CHAN", ALERT)

    await bridge.send_interaction(shortcut(ALERT, user="U0ASKER"))

    (start,) = slack.calls_to("chat.startStream")
    assert start.args["recipient_user_id"] == "U0ASKER"


async def test_shortcut_on_a_reply_uses_that_thread_and_its_earlier_messages(make_bridge, slack):
    bridge = make_bridge()
    slack.add_message("C0CHAN", ALERT)
    reply = slack.add_message(
        "C0CHAN",
        {"ts": "1700000200.000001", "user": "U0OMAR", "text": "rotated logs", "thread_ts": THREAD},
    )

    await bridge.send_interaction(shortcut(reply))

    (run,) = bridge.runs
    assert run.slack_config["conversation_ts"] == THREAD
    assert run.slack_config["message_ts"] == "1700000200.000001"
    assert [m.text for m in run.backfill] == ["Disk 92% full on db-1", "rotated logs"]
    assert bridge.agent_messages("C0CHAN")[-1]["thread_ts"] == THREAD


async def test_shortcut_callback_and_instruction_are_configurable(make_bridge, slack):
    bridge = make_bridge(
        message_shortcut_callback_id="triage",
        message_shortcut_instruction="Triage this alert.",
    )
    slack.add_message("C0CHAN", ALERT)

    await bridge.send_interaction(shortcut(ALERT, callback_id="ask_agent"))
    await bridge.send_interaction(shortcut(ALERT, callback_id="triage", trigger_id="t2"))

    assert [r.request for r in bridge.runs] == ["Triage this alert."]


async def test_redelivered_shortcut_starts_no_second_run(make_bridge, slack):
    bridge = make_bridge()
    slack.add_message("C0CHAN", ALERT)

    await bridge.send_interaction(shortcut(ALERT, trigger_id="t1"))
    await bridge.send_interaction(shortcut(ALERT, trigger_id="t1"))

    assert len(bridge.runs) == 1


async def test_later_replies_in_the_thread_continue_the_conversation(make_bridge, slack):
    bridge = make_bridge()
    slack.add_message("C0CHAN", ALERT)
    await bridge.send_interaction(shortcut(ALERT))

    await bridge.send_event(
        bridge.message(
            "C0CHAN",
            "can you clean it up?",
            channel_type="channel",
            ts="1700000600.000001",
            thread_ts=THREAD,
        )
    )

    assert [r.request for r in bridge.runs][-1] == "can you clean it up?"
    assert bridge.runs[-1].slack_config["trigger_kind"] == "joined_thread"
    assert bridge.runs[-1].thread_id == bridge.runs[0].thread_id


async def test_shortcut_on_the_agents_own_message_works(make_bridge, slack):
    bridge = make_bridge()
    own = slack.add_message(
        "C0CHAN",
        {
            "user": AGENT_USER_ID,
            "bot_id": "B0AGENT",
            "text": "Deployed v2",
            "ts": "1700000300.000001",
        },
    )

    await bridge.send_interaction(shortcut(own))

    assert len(bridge.runs) == 1


async def test_shortcut_can_be_disabled(make_bridge, slack):
    bridge = make_bridge(message_shortcut_callback_id=None)
    slack.add_message("C0CHAN", ALERT)

    await bridge.send_interaction(shortcut(ALERT))

    assert bridge.runs == []


@pytest.mark.parametrize(
    "channel, info",
    [("C0OTHER", {"name": "random", "is_channel": True, "is_member": False}), ("D0PEOPLE", None)],
    ids=["channel-app-is-not-in", "dm-between-people"],
)
async def test_shortcut_where_the_app_cannot_reply_explains_privately(
    make_bridge, slack, channel, info
):
    bridge = make_bridge()
    if info:
        slack.channels[channel] = info

    await bridge.send_interaction(shortcut({**ALERT}, channel=channel))

    assert bridge.runs == []
    ((url, text),) = slack.responses
    assert url == "https://hooks.slack.com/app/T0TEAM/1/abc"
    assert "Add me to the channel" in text


async def test_the_subject_is_kept_even_beyond_the_backfill_budget(make_bridge, slack):
    bridge = make_bridge(backfill_max_messages=3)
    slack.add_message("C0CHAN", ALERT)
    target = slack.add_message(
        "C0CHAN",
        {"ts": "1700000110.000001", "user": "U0OMAR", "text": "this one", "thread_ts": THREAD},
    )
    for i in range(5):
        slack.add_message(
            "C0CHAN",
            {
                "ts": f"17000002{i}0.000001",
                "user": "U0OMAR",
                "text": f"later {i}",
                "thread_ts": THREAD,
            },
        )

    await bridge.send_interaction(shortcut(target))

    texts = [m.text for m in bridge.runs[0].backfill]
    assert "this one" in texts
    assert "Disk 92% full on db-1" in texts
    assert len(texts) == 3


async def test_shortcut_on_an_older_message_in_a_joined_thread_includes_it(make_bridge, slack):
    bridge = make_bridge()
    slack.add_message("C0CHAN", ALERT)
    await bridge.send_event(
        bridge.message(
            "C0CHAN",
            f"<@{AGENT_USER_ID}> look",
            channel_type="channel",
            ts="1700000200.000001",
            thread_ts=THREAD,
        )
    )

    await bridge.send_interaction(shortcut(ALERT, trigger_id="t-later"))

    assert [m.text for m in bridge.runs[1].backfill] == ["Disk 92% full on db-1"]


async def test_unreadable_thread_falls_back_to_the_payload_message(make_bridge, slack):
    bridge = make_bridge()
    slack.fail("conversations.replies")

    await bridge.send_interaction(shortcut(ALERT))

    assert [m.text for m in bridge.runs[0].backfill] == ["Disk 92% full on db-1"]


async def test_a_message_asked_about_can_still_trigger_by_edit(make_bridge, slack):
    bridge = make_bridge()
    mine = slack.add_message("C0CHAN", {"ts": "1700000300.000001", "user": HUMAN, "text": "hm"})
    await bridge.send_interaction(shortcut(mine))

    await bridge.send_event(
        {
            "type": "message",
            "subtype": "message_changed",
            "channel": "C0CHAN",
            "channel_type": "channel",
            "message": {
                "type": "message",
                "user": HUMAN,
                "ts": mine["ts"],
                "text": f"<@{AGENT_USER_ID}> hm?",
            },
            "previous_message": {"type": "message", "user": HUMAN, "ts": mine["ts"], "text": "hm"},
        }
    )

    assert [r.slack_config["trigger_kind"] for r in bridge.runs] == ["shortcut", "edit"]
