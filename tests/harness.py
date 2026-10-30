"""In-memory fakes and a dispatcher for testing the bridge through its public seam.

A test hands the bridge a raw Slack payload. The payload is signed and dispatched through
Bolt exactly as the HTTP endpoint would, against two fakes:

- ``FakeSlackWorkspace`` answers every Web API call the bridge makes (all calls go through
  ``AsyncWebClient.api_call``, which the harness routes to the fake) and records them.
- ``FakeLangGraphDeployment`` records run requests and implements the thread and store APIs.

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
from typing import Any, Callable, Optional, Union
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

import httpx
from langgraph_sdk.errors import InternalServerError, NotFoundError
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
        messages = sorted(self.thread(args["channel"], args["ts"]), key=lambda m: _ts(m["ts"]))
        if args.get("oldest"):
            # Slack's `oldest` is exclusive unless `inclusive` is set
            oldest = _ts(args["oldest"])
            messages = [
                m
                for m in messages
                if _ts(m["ts"]) > oldest or (args.get("inclusive") and _ts(m["ts"]) == oldest)
            ]
        return {"ok": True, "messages": messages, "has_more": False}

    def _users_info(self, args: dict) -> dict:
        user = self.users.get(args["user"])
        if user is None:
            return {"ok": False, "error": "user_not_found"}
        return {"ok": True, "user": {"id": args["user"], **user}}

    def _conversations_info(self, args: dict) -> dict:
        channel = self.channels.get(args["channel"])
        if channel is None:
            return {"ok": False, "error": "channel_not_found"}
        return {"ok": True, "channel": {"id": args["channel"], **channel}}

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
        # Slack requires a recipient user (and team) to open a stream
        if not args.get("recipient_user_id") or not args.get("recipient_team_id"):
            return {"ok": False, "error": "invalid_arguments"}
        return self._post_as_agent(args)

    def _chat_update(self, args: dict) -> dict:
        return {"ok": True, "channel": args["channel"], "ts": args["ts"]}

    def _chat_stopStream(self, args: dict) -> dict:
        return {"ok": True, "channel": args["channel"], "ts": args["ts"]}


def _ts(ts: str) -> tuple[int, int]:
    seconds, _, micros = ts.partition(".")
    return int(seconds), int(micros or 0)


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

    @property
    def request(self) -> str:
        """The request part of the framed input: what follows the "Request from ..." line."""
        _, _, request_section = self.text.rpartition("Request from ")
        _, _, request = request_section.partition("\n")
        return request

    @property
    def backfill(self) -> list["ContextLine"]:
        """The messages in the framed input's untrusted context section."""
        if "<slack_context" not in self.text:
            return []
        section = self.text.split("</slack_context>")[0]
        return [ContextLine.parse(line) for line in section.splitlines() if line.startswith("[")]


@dataclass
class ContextLine:
    """One backfill line: "[time] Name (ID, human|bot): text"."""

    time: str
    display_name: str
    author_id: str
    author_type: str
    text: str

    @classmethod
    def parse(cls, line: str) -> "ContextLine":
        time_part, _, rest = line[1:].partition("] ")
        who, _, text = rest.partition("): ")
        name, _, ids = who.rpartition(" (")
        author_id, _, author_type = ids.partition(", ")
        return cls(time_part, name, author_id, author_type, text)


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
        for message in self._d.reply_messages(request):
            yield Chunk("messages", (message, {}))

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
        reply = self._d.reply_for(request)
        if isinstance(reply, str):
            produced = [{"type": "ai", "content": reply}]
        else:
            # Streamed chunks become complete messages in the final state
            produced = [
                {**m, "type": "ai"} if m.get("type") == "AIMessageChunk" else m for m in reply
            ]
        return {"messages": [*request.input["messages"], *produced]}


class _FakeThreads:
    def __init__(self, deployment: "FakeLangGraphDeployment") -> None:
        self._d = deployment

    async def get(self, thread_id, **kwargs):
        if thread_id not in self._d.threads_by_id:
            request = httpx.Request("GET", f"http://fake/threads/{thread_id}")
            raise NotFoundError(
                "thread not found", response=httpx.Response(404, request=request), body=None
            )
        return self._d.threads_by_id[thread_id]

    async def create(self, *, thread_id=None, if_exists=None, **kwargs):
        thread_id = thread_id or str(uuid.uuid4())
        return self._d.threads_by_id.setdefault(thread_id, {"thread_id": thread_id, "values": {}})

    async def update_state(self, thread_id, values, **kwargs):
        thread = self._d.threads_by_id.setdefault(thread_id, {"thread_id": thread_id, "values": {}})
        thread["values"].update(values)
        return {}


def server_error(message: str = "Internal Server Error") -> InternalServerError:
    """What the SDK raises for an unhandled server error over HTTP: a bare 500."""
    request = httpx.Request("PUT", "http://fake/store/items")
    return InternalServerError(message, response=httpx.Response(500, request=request), body=message)


class _FakeStore:
    """The LangGraph store: namespaced items with TTLs in minutes, on a controllable clock.

    The clock is the deployment's clock: the bridge's own expiry and probe scheduling follow
    it too (see the ``make_bridge`` fixture), so ``advance()`` moves time for both.

    - ``reachable = False``: every call fails with a connection error.
    - ``fail_next(error, times)``: the next calls fail with ``error``, then calls succeed.
    - ``supports_ttl = False``: writes with a TTL fail the way the server reports it, per
      ``ttl_error``: "loopback" (the store's NotImplementedError reaches the client) or
      "http" (a bare 500).
    """

    def __init__(self) -> None:
        self.items: dict[tuple[tuple[str, ...], str], dict] = {}
        self.now_minutes = 0.0
        self.reachable = True
        self.supports_ttl = True
        self.ttl_error = "loopback"
        self._failures: list[Exception] = []

    @property
    def now_seconds(self) -> float:
        return self.now_minutes * 60

    def fail_next(self, error: Exception, times: int = 1) -> None:
        self._failures.extend([error] * times)

    def _check_reachable(self) -> None:
        if self._failures:
            raise self._failures.pop(0)
        if not self.reachable:
            raise httpx.ConnectError("store not reachable")

    def advance(self, *, minutes: float = 0, days: float = 0) -> None:
        self.now_minutes += minutes + days * 24 * 60

    def namespace(self, name: str) -> dict[str, dict]:
        """Live items in one bridge namespace, by key."""
        return {
            key: item
            for (ns, key), item in self.items.items()
            if ns[-1] == name and not self._expired(item)
        }

    def _expired(self, item: dict) -> bool:
        return item["expires_at"] is not None and self.now_minutes >= item["expires_at"]

    async def put_item(self, namespace, /, key, value, index=None, ttl=None, **kwargs):
        self._check_reachable()
        if ttl is not None and not self.supports_ttl:
            if self.ttl_error == "loopback":
                raise NotImplementedError(
                    "TTL is not supported by InMemoryStore. Use a store implementation that "
                    "supports TTL or set ttl=None."
                )
            raise server_error()
        self.items[(tuple(namespace), key)] = {
            "value": dict(value),
            "index": index,
            "ttl": ttl,
            "expires_at": self.now_minutes + ttl if ttl is not None else None,
        }

    async def get_item(self, namespace, /, key, *, refresh_ttl=None, **kwargs):
        self._check_reachable()
        item = self.items.get((tuple(namespace), key))
        if item is None or self._expired(item):
            request = httpx.Request("GET", "http://fake/store/items")
            raise NotFoundError(
                "not found", response=httpx.Response(404, request=request), body=None
            )
        return {"namespace": list(namespace), "key": key, "value": item["value"]}

    async def delete_item(self, namespace, /, key, **kwargs):
        self._check_reachable()
        self.items.pop((tuple(namespace), key), None)


Reply = Union[str, list[dict]]


class FakeLangGraphDeployment:
    """Records run requests and implements the thread and store APIs the bridge uses.

    ``reply`` is what each run streams back: a string (one AI message chunk), a list of
    message dicts (streamed in order, for tool calls), or a callable taking the RunRequest.
    """

    def __init__(self) -> None:
        self.runs_started: list[RunRequest] = []
        self.threads_by_id: dict[str, dict] = {}
        self.reply: Union[Reply, Callable[[RunRequest], Reply]] = "Hello from the agent"
        self.runs = _FakeRuns(self)
        self.threads = _FakeThreads(self)
        self.store = _FakeStore()

    def reply_for(self, request: RunRequest) -> Reply:
        return self.reply(request) if callable(self.reply) else self.reply

    def reply_messages(self, request: RunRequest) -> list[dict]:
        reply = self.reply_for(request)
        if isinstance(reply, str):
            return [{"type": "AIMessageChunk", "content": reply}] if reply else []
        return reply


# ============================================================================
# Bridge harness
# ============================================================================


class BridgeHarness:
    """A SlackBot wired to the fakes, driven by raw Slack payloads."""

    def __init__(self, bot, slack: FakeSlackWorkspace, deployment: FakeLangGraphDeployment):
        self.bot = bot
        self.slack = slack
        self.deployment = deployment

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
            "event_time": int(time.time()),
            "authorizations": [{"team_id": TEAM_ID, "user_id": AGENT_USER_ID, "is_bot": True}],
        }
        # A fresh event ID per delivery unless given; "" sends none (exercises the fallback key)
        if event_id is None:
            event_id = f"Ev{uuid.uuid4().hex[:12]}"
        if event_id:
            envelope["event_id"] = event_id
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

    async def click(
        self, action_id: str, value: str, *, channel: str, message_ts: str, user: str = "U0HUMAN"
    ) -> int:
        """Click a button on a Slack message."""
        return await self.send_interaction(
            {
                "type": "block_actions",
                "team": {"id": TEAM_ID},
                "api_app_id": AGENT_APP_ID,
                "user": {"id": user, "team_id": TEAM_ID},
                "trigger_id": "trigger-1",
                "container": {
                    "type": "message",
                    "channel_id": channel,
                    "message_ts": message_ts,
                },
                "channel": {"id": channel},
                "actions": [
                    {"action_id": action_id, "value": value, "type": "button", "block_id": "b"}
                ],
            }
        )

    @property
    def feedback_submitted(self) -> list[dict]:
        """Feedback the bridge sent to LangSmith (the client is faked)."""
        return [c.kwargs for c in self.bot.langsmith_client.create_feedback.call_args_list]

    def agent_messages(self, channel: str) -> list[dict]:
        """Messages the agent has posted in a channel."""
        return [m for m in self.slack.messages.get(channel, []) if m.get("user") == AGENT_USER_ID]


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
    with (
        patch("langgraph2slack.bot.get_client", return_value=deployment),
        patch("langgraph2slack.bot.Client", MagicMock),
    ):
        bot = SlackBot(**slackbot_kwargs)
    return BridgeHarness(bot, slack, deployment)
