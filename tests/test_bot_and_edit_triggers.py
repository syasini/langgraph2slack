"""Bot triggers (ADR-0007) and edit triggers, tested through the bridge's public seam."""

import pytest

from .harness import AGENT_BOT_ID, AGENT_USER_ID, TEAM_ID

MENTION = f"<@{AGENT_USER_ID}>"
HUMAN = "U0HUMAN"
THREAD = "1600000000.000001"


def app_bot_message(text, channel="C0CHAN", **fields):
    """A message from a modern Slack app: bot user, bot ID, app ID and bot profile."""
    return {
        "type": "message",
        "channel": channel,
        "channel_type": "channel",
        "user": "U0CIBOT",
        "bot_id": "B0CI",
        "app_id": "A0CI",
        "bot_profile": {"id": "B0CI", "app_id": "A0CI", "name": "ci"},
        "text": text,
        **fields,
    }


def classic_bot_message(text, channel="C0CHAN", **fields):
    """A message from a classic integration: no user, just a bot ID."""
    return {
        "type": "message",
        "subtype": "bot_message",
        "channel": channel,
        "channel_type": "channel",
        "bot_id": "B0LEGACY",
        "username": "alerts",
        "text": text,
        **fields,
    }


def joined_thread(slack, channel="C0CHAN"):
    slack.add_message(channel, {"ts": THREAD, "user": HUMAN, "text": "parent"})
    slack.add_message(
        channel, {"user": AGENT_USER_ID, "bot_id": AGENT_BOT_ID, "text": "hi", "thread_ts": THREAD}
    )


def edit(new_text, previous_text, *, channel="C0CHAN", ts="1600000000.000005", **author):
    author = author or {"user": HUMAN}
    return {
        "type": "message",
        "subtype": "message_changed",
        "hidden": True,
        "channel": channel,
        "channel_type": "channel",
        "message": {
            "type": "message",
            "text": new_text,
            "ts": ts,
            "edited": {"user": author.get("user", "B"), "ts": "1600000099.000000"},
            **author,
        },
        "previous_message": {"type": "message", "text": previous_text, "ts": ts, **author},
    }


@pytest.fixture(params=[True, False], ids=["streaming", "non-streaming"])
def streaming(request):
    return request.param


# ============================================================================
# Bot triggers
# ============================================================================


async def test_bot_mention_is_ignored_by_default(make_bridge):
    bridge = make_bridge()

    await bridge.send_event(app_bot_message(f"{MENTION} build failed"))
    await bridge.send_event(classic_bot_message(f"{MENTION} disk full"))

    assert bridge.runs == []


async def test_app_bot_mention_starts_a_run_with_bot_provenance(make_bridge, streaming):
    bridge = make_bridge(streaming=streaming, allow_bot_triggers=True)

    await bridge.send_event(app_bot_message(f"{MENTION} build 42 failed"))

    (run,) = bridge.runs
    assert run.request == "build 42 failed"
    assert run.slack_config["trigger_kind"] == "mention"
    assert run.slack_config["author_type"] == "bot"
    assert run.slack_config["author_id"] == "U0CIBOT"
    assert run.slack_config["bot_id"] == "B0CI"
    assert run.slack_config["app_id"] == "A0CI"
    assert run.slack_config["requester_id"] is None
    assert run.metadata["slack_author_type"] == "bot"
    assert run.metadata["slack_requester_id"] is None
    assert bridge.agent_messages("C0CHAN"), "the agent did not reply"


async def test_classic_integration_mention_starts_a_run(make_bridge, streaming):
    bridge = make_bridge(streaming=streaming, allow_bot_triggers=True)

    await bridge.send_event(classic_bot_message(f"{MENTION} disk full on db-1"))

    (run,) = bridge.runs
    assert run.request == "disk full on db-1"
    assert run.slack_config["author_type"] == "bot"
    assert run.slack_config["author_id"] is None
    assert run.slack_config["bot_id"] == "B0LEGACY"
    assert bridge.agent_messages("C0CHAN"), "the agent did not reply"


async def test_classic_integration_reply_streams_to_the_bridge_itself(make_bridge):
    bridge = make_bridge(streaming=True, allow_bot_triggers=True)

    await bridge.send_event(classic_bot_message(f"{MENTION} disk full"))

    (start,) = bridge.slack.calls_to("chat.startStream")
    assert start.args["recipient_user_id"] == AGENT_USER_ID
    assert start.args["recipient_team_id"] == TEAM_ID


async def test_app_bot_reply_streams_to_the_bot_user(make_bridge):
    bridge = make_bridge(streaming=True, allow_bot_triggers=True)

    await bridge.send_event(app_bot_message(f"{MENTION} build failed"))

    (start,) = bridge.slack.calls_to("chat.startStream")
    assert start.args["recipient_user_id"] == "U0CIBOT"


@pytest.mark.parametrize("allowlist", [["B0CI"], ["A0CI"]], ids=["bot-id", "app-id"])
async def test_allowlist_matches_bot_or_app_id(make_bridge, allowlist):
    bridge = make_bridge(allow_bot_triggers=True, bot_trigger_allowlist=allowlist)

    await bridge.send_event(app_bot_message(f"{MENTION} build failed"))
    await bridge.send_event(classic_bot_message(f"{MENTION} disk full"))

    assert [r.slack_config["bot_id"] for r in bridge.runs] == ["B0CI"]


async def test_agent_never_triggers_on_its_own_message_even_with_bot_triggers(make_bridge):
    bridge = make_bridge(allow_bot_triggers=True)

    await bridge.send_event(classic_bot_message(f"{MENTION} echo", bot_id=AGENT_BOT_ID))
    await bridge.send_event(app_bot_message(f"{MENTION} echo", user=AGENT_USER_ID))

    assert bridge.runs == []


async def test_bot_needs_a_mention_in_a_joined_thread(make_bridge):
    bridge = make_bridge(allow_bot_triggers=True)
    joined_thread(bridge.slack)

    await bridge.send_event(app_bot_message("status: deploying", thread_ts=THREAD))
    await bridge.send_event(app_bot_message(f"{MENTION} status?", thread_ts=THREAD))

    assert [r.request for r in bridge.runs] == ["status?"]


async def test_bot_needs_a_mention_in_a_direct_message(make_bridge):
    bridge = make_bridge(allow_bot_triggers=True)

    await bridge.send_event(app_bot_message("ping", channel="D0DM", channel_type="im"))

    assert bridge.runs == []


async def test_allowlisted_bot_can_converse_in_a_joined_thread_when_opted_in(make_bridge):
    bridge = make_bridge(
        allow_bot_triggers=True,
        bot_trigger_allowlist=["A0CI"],
        allowlisted_bots_in_joined_threads=True,
    )
    joined_thread(bridge.slack)

    await bridge.send_event(app_bot_message("what next?", thread_ts=THREAD))
    await bridge.send_event(classic_bot_message("not allowlisted", thread_ts=THREAD))

    (run,) = bridge.runs
    assert run.request == "what next?"
    assert run.slack_config["trigger_kind"] == "joined_thread"


def test_joined_thread_opt_in_requires_an_allowlist(make_bridge):
    with pytest.raises(ValueError, match="allowlist"):
        make_bridge(allow_bot_triggers=True, allowlisted_bots_in_joined_threads=True)


# ============================================================================
# Edit triggers
# ============================================================================


async def test_edit_that_adds_a_mention_triggers(make_bridge, streaming):
    bridge = make_bridge(streaming=streaming)

    await bridge.send_event(edit(f"{MENTION} can you look at this?", "can you look at this?"))

    (run,) = bridge.runs
    assert run.request == "can you look at this?"
    assert run.slack_config["trigger_kind"] == "edit"
    assert run.slack_config["message_ts"] == "1600000000.000005"
    assert run.slack_config["requester_id"] == HUMAN
    assert bridge.agent_messages("C0CHAN"), "the agent did not reply"


async def test_edit_in_a_thread_replies_in_that_thread(make_bridge):
    bridge = make_bridge()
    event = edit(f"{MENTION} thoughts?", "thoughts?")
    event["message"]["thread_ts"] = THREAD
    event["previous_message"]["thread_ts"] = THREAD

    await bridge.send_event(event)

    assert bridge.runs[0].slack_config["conversation_ts"] == THREAD
    assert bridge.agent_messages("C0CHAN")[-1]["thread_ts"] == THREAD


async def test_editing_a_message_that_already_triggered_is_ignored(make_bridge):
    bridge = make_bridge()
    await bridge.send_event(
        bridge.message("C0CHAN", f"{MENTION} helo", channel_type="channel", ts="1600000000.000005")
    )

    await bridge.send_event(edit(f"{MENTION} hello", f"{MENTION} helo"))

    assert [r.request for r in bridge.runs] == ["helo"]


async def test_editing_a_joined_thread_message_that_already_triggered_is_ignored(make_bridge):
    bridge = make_bridge()
    joined_thread(bridge.slack)
    await bridge.send_event(
        bridge.message(
            "C0CHAN", "follow up", channel_type="channel", ts="1600000000.000005", thread_ts=THREAD
        )
    )
    event = edit(f"{MENTION} follow up", "follow up")
    event["message"]["thread_ts"] = THREAD

    await bridge.send_event(event)

    assert [r.request for r in bridge.runs] == ["follow up"]


@pytest.mark.parametrize(
    "new, previous",
    [
        ("fixed typo", "fixd typo"),
        (f"{MENTION} again", f"{MENTION} agin"),
        (f"{MENTION} same", f"{MENTION} same"),
    ],
    ids=["no-mention", "already-mentioned", "unfurl"],
)
async def test_edits_that_do_not_add_a_mention_are_ignored(make_bridge, new, previous):
    bridge = make_bridge()

    await bridge.send_event(edit(new, previous))

    assert bridge.runs == []


async def test_a_second_edit_after_an_edit_trigger_is_ignored(make_bridge):
    bridge = make_bridge()
    await bridge.send_event(edit(f"{MENTION} look", "look"))

    await bridge.send_event(edit(f"look {MENTION}", "look"))

    assert len(bridge.runs) == 1


async def test_bot_edits_follow_bot_gating(make_bridge):
    bot_author = {"user": "U0CIBOT", "bot_id": "B0CI", "app_id": "A0CI"}
    blocked = make_bridge()
    await blocked.send_event(edit(f"{MENTION} failed", "failed", **bot_author))
    assert blocked.runs == []

    allowed = make_bridge(allow_bot_triggers=True, bot_trigger_allowlist=["A0CI"])
    await allowed.send_event(
        edit(f"{MENTION} failed", "failed", ts="1600000000.000077", **bot_author)
    )
    assert [r.slack_config["author_type"] for r in allowed.runs] == ["bot"]


async def test_edits_of_the_agents_own_messages_are_ignored(make_bridge):
    bridge = make_bridge(allow_bot_triggers=True)

    await bridge.send_event(
        edit(f"{MENTION} streamed", "streamed", user=AGENT_USER_ID, bot_id=AGENT_BOT_ID)
    )

    assert bridge.runs == []


async def test_edits_of_unhandled_subtypes_are_ignored(make_bridge):
    bridge = make_bridge()
    event = edit(f"{MENTION} waves", "waves")
    event["message"]["subtype"] = "me_message"

    await bridge.send_event(event)

    assert bridge.runs == []


async def test_edit_can_retrigger_a_message_whose_run_was_never_created(make_bridge, deployment):
    bridge = make_bridge(streaming=False)
    create = deployment.runs.create

    async def fail_once(*args, **kwargs):
        deployment.runs.create = create
        raise RuntimeError("deployment unavailable")

    deployment.runs.create = fail_once
    await bridge.send_event(
        bridge.message("C0CHAN", f"{MENTION} helo", channel_type="channel", ts="1600000000.000005")
    )

    await bridge.send_event(edit(f"{MENTION} hello", "hello"))

    assert [r.request for r in bridge.runs] == ["hello"]
