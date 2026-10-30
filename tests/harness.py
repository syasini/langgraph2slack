"""In-memory fakes and a dispatcher for testing the bridge through its public seam.

A test hands the bridge a raw Slack payload. The payload is signed and dispatched through
Bolt exactly as the HTTP endpoint would, against two fakes:

- ``FakeSlackWorkspace`` answers every Web API call the bridge makes (all calls go through
  ``AsyncWebClient.api_call``, which the harness routes to the fake) and records them.
- ``FakeLangGraphDeployment`` records run requests and implements the thread API.

Payloads are signed with the test signing secret, so Bolt's request verification runs as it
does in production.

Tests then assert on outward behavior only: which runs started, with what input, metadata
and config, and what the bridge called in Slack.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from unittest.mock import patch
from urllib.parse import urlencode

from slack_bolt.request.async_request import AsyncBoltRequest
from slack_sdk.signature import SignatureVerifier
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse

SIGNING_SECRET = "test-signing-secret"
BOT_TOKEN = "xoxb-test-token"
ASSISTANT_ID = "test-assistant"

TEAM_ID = "T0TEAM"
AGENT_USER_ID = "U0AGENT"
AGENT_BOT_ID = "B0AGENT"
AGENT_APP_ID = "A0AGENT"


# ============================================================================
# Fake Slack workspace
# ============================================================================


@dataclass
class SlackCall:
    method: str
    args: dict


class FakeSlackWorkspace:
    """Answers the Slack Web API calls the bridge makes and records every call.

    Holds channels of messages so history and thread-replies lookups return what a real
    workspace would. Messages the bridge posts are stored too, authored by the agent.
    """

    def __init__(self) -> None:
        self.calls: list[SlackCall] = []
        # channel_id -> list of message dicts (top-level and replies, in ts order)
        self.messages: dict[str, list[dict]] = {}
        self.users: dict[str, dict] = {}
        self.channels: dict[str, dict] = {}
        self.failing_methods: dict[str, str] = {}
        self._ts = itertools.count(1)

    # -- workspace setup ---------------------------------------------------

    def next_ts(self) -> str:
        return f"1700000000.{next(self._ts):06d}"

    def add_message(self, channel: str, message: dict) -> dict:
        """Store a message in the workspace (as if it had been posted earlier)."""
        message = dict(message)
        message.setdefault("type", "message")
        message.setdefault("ts", self.next_ts())
        self.messages.setdefault(channel, []).append(message)
        return message

    def fail(self, method: str, error: str = "fatal_error") -> None:
        """Make every call to ``method`` fail with a Slack API error."""
        self.failing_methods[method] = error

    # -- inspection --------------------------------------------------------

    def calls_to(self, method: str) -> list[SlackCall]:
        return [c for c in self.calls if c.method == method]

    def thread(self, channel: str, thread_ts: str) -> list[dict]:
        return [
            m
            for m in self.messages.get(channel, [])
            if m.get("ts") == thread_ts or m.get("thread_ts") == thread_ts
        ]

    # -- API dispatch ------------------------------------------------------

    async def handle(self, method: str, args: dict) -> dict:
        self.calls.append(SlackCall(method, args))
        if method in self.failing_methods:
            return {"ok": False, "error": self.failing_methods[method]}
        handler = getattr(self, "_" + method.replace(".", "_"), None)
        if handler is None:
            return {"ok": True}
        return handler(args)

    def _auth_test(self, args: dict) -> dict:
        return {
            "ok": True,
            "url": "https://test.slack.com/",
            "team": "Test",
            "user": "agent",
            "team_id": TEAM_ID,
            "user_id": AGENT_USER_ID,
            "bot_id": AGENT_BOT_ID,
        }

    def _conversations_replies(self, args: dict) -> dict:
        return {"ok": True, "messages": self.thread(args["channel"], args["ts"])}

    def _post_as_agent(self, args: dict) -> dict:
        channel = args["channel"]
        message = self.add_message(
            channel,
            {
                "user": AGENT_USER_ID,
                "bot_id": AGENT_BOT_ID,
                "app_id": AGENT_APP_ID,
                "text": args.get("text", ""),
                **({"thread_ts": args["thread_ts"]} if args.get("thread_ts") else {}),
            },
        )
        return {"ok": True, "channel": channel, "ts": message["ts"], "message": message}

    def _chat_postMessage(self, args: dict) -> dict:
        return self._post_as_agent(args)

    def _chat_startStream(self, args: dict) -> dict:
        return self._post_as_agent(args)

    def _chat_update(self, args: dict) -> dict:
        return {"ok": True, "channel": args["channel"], "ts": args["ts"]}

    def _chat_stopStream(self, args: dict) -> dict:
        return {"ok": True, "channel": args["channel"], "ts": args["ts"]}


# ============================================================================
# Fake LangGraph deployment
# ============================================================================


@dataclass
class Chunk:
    event: str
    data: Any


@dataclass
class RunRequest:
    """One run the bridge asked the deployment to start."""

    thread_id: str
    assistant_id: str
    input: dict
    metadata: dict
    config: Optional[dict]
    multitask_strategy: Optional[str]
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def message(self) -> dict:
        """The single human message the run received."""
        (message,) = self.input["messages"]
        return message

    @property
    def text(self) -> str:
        content = self.message["content"]
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") for b in content if b.get("type") == "text")

    @property
    def slack_config(self) -> dict:
        return (self.config or {}).get("configurable", {}).get("slack", {})


class ThreadNotFound(Exception):
    pass


class _FakeRuns:
    def __init__(self, deployment: "FakeLangGraphDeployment") -> None:
        self._d = deployment

    def _record(self, thread_id, assistant_id, input, metadata, config, multitask_strategy):
        request = RunRequest(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input=input,
            metadata=metadata or {},
            config=config,
            multitask_strategy=multitask_strategy,
        )
        self._d.runs_started.append(request)
        self._d.threads_by_id.setdefault(thread_id, {"thread_id": thread_id, "values": {}})
        return request

    async def stream(
        self,
        thread_id,
        assistant_id,
        *,
        input=None,
        stream_mode=None,
        multitask_strategy=None,
        if_not_exists=None,
        metadata=None,
        config=None,
        **kwargs,
    ):
        request = self._record(thread_id, assistant_id, input, metadata, config, multitask_strategy)
        yield Chunk("metadata", {"run_id": request.run_id})
        reply = self._d.reply_for(request)
        if reply:
            yield Chunk("messages", ({"type": "AIMessageChunk", "content": reply}, {}))

    async def create(
        self,
        thread_id,
        assistant_id,
        *,
        input=None,
        if_not_exists=None,
        multitask_strategy=None,
        metadata=None,
        config=None,
        **kwargs,
    ):
        request = self._record(thread_id, assistant_id, input, metadata, config, multitask_strategy)
        return {"run_id": request.run_id, "thread_id": thread_id}

    async def join(self, thread_id, run_id, **kwargs):
        request = next(r for r in self._d.runs_started if r.run_id == run_id)
        return {
            "messages": [
                *request.input["messages"],
                {"type": "ai", "content": self._d.reply_for(request)},
            ]
        }


class _FakeThreads:
    def __init__(self, deployment: "FakeLangGraphDeployment") -> None:
        self._d = deployment

    async def get(self, thread_id, **kwargs):
        if thread_id not in self._d.threads_by_id:
            raise ThreadNotFound(thread_id)
        return self._d.threads_by_id[thread_id]

    async def create(self, *, thread_id=None, if_exists=None, **kwargs):
        thread_id = thread_id or str(uuid.uuid4())
        return self._d.threads_by_id.setdefault(thread_id, {"thread_id": thread_id, "values": {}})

    async def update_state(self, thread_id, values, **kwargs):
        thread = self._d.threads_by_id.setdefault(thread_id, {"thread_id": thread_id, "values": {}})
        thread["values"].update(values)
        return {}


class FakeLangGraphDeployment:
    """Records run requests and implements the thread API the bridge uses."""

    def __init__(self) -> None:
        self.runs_started: list[RunRequest] = []
        self.threads_by_id: dict[str, dict] = {}
        self.reply: str | Callable[[RunRequest], str] = "Hello from the agent"
        self.runs = _FakeRuns(self)
        self.threads = _FakeThreads(self)

    def reply_for(self, request: RunRequest) -> str:
        return self.reply(request) if callable(self.reply) else self.reply


# ============================================================================
# Bridge harness
# ============================================================================


class BridgeHarness:
    """A SlackBot wired to the fakes, driven by raw Slack payloads."""

    def __init__(self, bot, slack: FakeSlackWorkspace, deployment: FakeLangGraphDeployment):
        self.bot = bot
        self.slack = slack
        self.deployment = deployment
        self._event_ids = itertools.count(1)

    @property
    def runs(self) -> list[RunRequest]:
        return self.deployment.runs_started

    async def send_event(self, event: dict, *, event_id: Optional[str] = None) -> int:
        """Deliver an Events API ``event_callback`` and wait for the bridge to finish."""
        event = dict(event)
        event.setdefault("type", "message")
        event.setdefault("ts", self.slack.next_ts())
        event.setdefault("event_ts", event["ts"])
        envelope = {
            "token": "verification-token",
            "team_id": TEAM_ID,
            "api_app_id": AGENT_APP_ID,
            "event": event,
            "type": "event_callback",
            "event_id": event_id or f"Ev{next(self._event_ids):08d}",
            "event_time": int(time.time()),
            "authorizations": [{"team_id": TEAM_ID, "user_id": AGENT_USER_ID, "is_bot": True}],
        }
        return await self._dispatch(json.dumps(envelope), "application/json")

    async def send_interaction(self, payload: dict) -> int:
        """Deliver an interactivity payload (shortcut, button, view) and wait."""
        body = urlencode({"payload": json.dumps(payload)})
        return await self._dispatch(body, "application/x-www-form-urlencoded")

    async def _dispatch(self, body: str, content_type: str) -> int:
        timestamp = str(int(time.time()))
        signature = SignatureVerifier(SIGNING_SECRET).generate_signature(
            timestamp=timestamp, body=body
        )
        request = AsyncBoltRequest(
            body=body,
            headers={
                "content-type": [content_type],
                "x-slack-request-timestamp": [timestamp],
                "x-slack-signature": [signature],
            },
        )
        response = await self.bot.slack_app.async_dispatch(request)
        await drain()
        return response.status

    def message(self, channel: str, text: str, **fields) -> dict:
        """Build a human message event (not yet delivered)."""
        return {"type": "message", "channel": channel, "user": "U0HUMAN", "text": text, **fields}


async def drain(timeout: float = 5.0) -> None:
    """Wait until every task except the caller has finished (Bolt acks, then runs listeners)."""
    current = asyncio.current_task()
    for _ in range(100):
        pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        if not pending:
            return
        await asyncio.wait(pending, timeout=timeout)
    raise AssertionError("bridge tasks did not settle")


def route_slack_api(workspace: FakeSlackWorkspace):
    """Patch AsyncWebClient.api_call so every Slack call is answered by ``workspace``.

    Returns the started patcher; call ``.stop()`` to undo.
    """

    async def api_call(
        self,
        api_method,
        *,
        http_verb="POST",
        files=None,
        data=None,
        params=None,
        json=None,
        headers=None,
        auth=None,
    ):
        args = {**(params or {}), **(data or {}), **(json or {})}
        response = await workspace.handle(api_method, args)
        return AsyncSlackResponse(
            client=self,
            http_verb=http_verb,
            api_url=f"https://slack.com/api/{api_method}",
            req_args={},
            data=response,
            headers={},
            status_code=200,
        ).validate()

    patcher = patch.object(AsyncWebClient, "api_call", api_call)
    patcher.start()
    return patcher


def build_bridge(
    slack: FakeSlackWorkspace,
    deployment: FakeLangGraphDeployment,
    **slackbot_kwargs,
) -> BridgeHarness:
    """Construct a SlackBot that talks to the fakes (Slack calls must be routed already)."""
    from langgraph2slack import SlackBot

    slackbot_kwargs.setdefault("assistant_id", ASSISTANT_ID)
    slackbot_kwargs.setdefault("slack_bot_token", BOT_TOKEN)
    slackbot_kwargs.setdefault("slack_signing_secret", SIGNING_SECRET)
    with patch("langgraph2slack.bot.get_client", return_value=deployment):
        bot = SlackBot(**slackbot_kwargs)
    return BridgeHarness(bot, slack, deployment)
