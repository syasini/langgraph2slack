"""Feedback buttons: feedback reaches LangSmith filed under the project the agent's runs are
traced to (LangSmith requires the project from 2027-01-31), and still goes out without it if
the project can't be found.
"""

import logging


async def reply_to_dm(bridge, text="hello"):
    await bridge.send_event(bridge.message("D0DM", text, channel_type="im"))
    replies = bridge.agent_messages("D0DM")
    assert replies, "the agent did not reply"
    return replies[-1]["ts"]


async def test_feedback_names_the_project_the_agent_is_traced_to(
    make_bridge, langsmith, monkeypatch
):
    project_id = langsmith.add_project("plant-bot")
    monkeypatch.setenv("LANGSMITH_PROJECT", "plant-bot")
    bridge = make_bridge(show_feedback_buttons=True)
    reply_ts = await reply_to_dm(bridge)

    await bridge.click("feedback", "positive", channel="D0DM", message_ts=reply_ts)

    assert bridge.feedback_submitted == [
        {
            "run_id": bridge.runs[0].run_id,
            "key": "user_feedback",
            "score": 1.0,
            "comment": "Slack user feedback: positive",
            "session_id": project_id,
        }
    ]


async def test_feedback_without_a_project_setting_goes_to_the_default_project(
    make_bridge, langsmith
):
    bridge = make_bridge(show_feedback_buttons=True)
    reply_ts = await reply_to_dm(bridge)

    await bridge.click("feedback", "negative", channel="D0DM", message_ts=reply_ts)

    assert bridge.feedback_submitted[0]["session_id"] == langsmith.projects["default"]


async def test_langsmith_project_setting_overrides_the_environment(
    make_bridge, langsmith, monkeypatch
):
    langsmith.add_project("from-env")
    project_id = langsmith.add_project("from-setting")
    monkeypatch.setenv("LANGSMITH_PROJECT", "from-env")
    bridge = make_bridge(show_feedback_buttons=True, langsmith_project="from-setting")
    reply_ts = await reply_to_dm(bridge)

    await bridge.click("feedback", "positive", channel="D0DM", message_ts=reply_ts)

    assert bridge.feedback_submitted[0]["session_id"] == project_id


async def test_project_is_looked_up_once(make_bridge, langsmith):
    bridge = make_bridge(show_feedback_buttons=True)
    first = await reply_to_dm(bridge, "one")
    second = await reply_to_dm(bridge, "two")

    await bridge.click("feedback", "positive", channel="D0DM", message_ts=first)
    await bridge.click("feedback", "negative", channel="D0DM", message_ts=second)

    assert langsmith.project_lookups == ["default"]
    assert [f["session_id"] for f in bridge.feedback_submitted] == [
        langsmith.projects["default"]
    ] * 2


async def test_feedback_goes_without_a_project_that_does_not_exist(make_bridge, langsmith, caplog):
    bridge = make_bridge(show_feedback_buttons=True, langsmith_project="no-such-project")
    reply_ts = await reply_to_dm(bridge)

    with caplog.at_level(logging.WARNING, logger="langgraph2slack.bot"):
        await bridge.click("feedback", "positive", channel="D0DM", message_ts=reply_ts)

    assert len(bridge.feedback_submitted) == 1
    assert bridge.feedback_submitted[0]["session_id"] is None
    assert "no-such-project" in caplog.text


async def test_a_failed_project_lookup_is_retried_on_the_next_click(make_bridge, langsmith):
    langsmith.fail_lookups(times=1)
    bridge = make_bridge(show_feedback_buttons=True)
    reply_ts = await reply_to_dm(bridge)

    await bridge.click("feedback", "positive", channel="D0DM", message_ts=reply_ts)
    await bridge.click("feedback", "positive", channel="D0DM", message_ts=reply_ts)

    assert [f["session_id"] for f in bridge.feedback_submitted] == [
        None,
        langsmith.projects["default"],
    ]
