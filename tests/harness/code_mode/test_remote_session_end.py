"""Real SDK checks for session-end statuses exposed by the remote transport."""

from __future__ import annotations

import pytest
from grpclib.const import Status
from grpclib.server import Stream

from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode
from tests.harness.code_mode.grpc_relay import MontyRelay, serve_monty_relay
from tests.harness.code_mode.test_remote import _retries, _snippets_model

# WorkerEvent FunctionCall("ping", call_id=7) and optional session ID, from monty.v1.
_PING_CALL = b'\x12\x08\x0a\x04ping\x20\x07'
_SESSION_ID = b'\xe2\x01\x04park'
_PRIVATE_TEXT = 'private diagnostic text'


async def accept_feed(stream: Stream[bytes, bytes], code: bytes) -> None:
    """Acknowledge an explicit Configure, or accept its embedded first feed."""
    await stream.send_initial_metadata()
    request = await stream.recv_message()
    assert request is not None and request.startswith(b'\x0a')  # Configure
    code_field = b'\x0a' + bytes((len(code),)) + code
    if code_field not in request:
        await stream.send_message(b'\x52\x00' + _SESSION_ID)  # Ok
        request = await stream.recv_message()
        assert request is not None and request.startswith(b'\x1a') and code_field in request


@pytest.mark.parametrize(
    ('cause', 'reason'),
    [
        ('idle_timeout', 'session expired while idle'),
        ('session_timeout', 'session reached its lifetime limit'),
        ('turn_timeout', 'turn reached the relay deadline'),
        (None, 'sandbox connection was lost'),
    ],
)
async def test_remote_session_end_cause(monkeypatch: pytest.MonkeyPatch, cause: str | None, reason: str) -> None:
    """The SDK decodes the stream status without replaying a reply whose receipt is unknown."""
    replies: list[bytes] = []

    async def handle(_relay: MontyRelay, stream: Stream[bytes, bytes]) -> None:
        await accept_feed(stream, b'ping()')
        await stream.send_message(_PING_CALL + _SESSION_ID)
        reply = await stream.recv_message()
        assert reply is not None and reply.startswith(b'\x22')
        replies.append(reply)
        await stream.send_trailing_metadata(
            status=Status.DEADLINE_EXCEEDED,
            status_message=_PRIVATE_TEXT,
            metadata={'monty-cause': cause} if cause is not None else {},
        )

    effects: list[int] = []
    monkeypatch.setattr(MontyRelay, '_bridge', handle)
    async with serve_monty_relay() as relay:
        agent = Agent(_snippets_model('ping()'), capabilities=[CodeMode(monty_sandbox=relay.url)])

        @agent.tool_plain(sequential=True)
        def ping() -> int:
            effects.append(7)
            raise RuntimeError(_PRIVATE_TEXT)

        result = await agent.run('run one feed')

    (retry,) = _retries(result.all_messages())
    assert reason in str(retry.content)
    assert 'session was reset' in str(retry.content)
    assert 'ping({}) started; outcome unknown' in str(retry.content)
    assert _PRIVATE_TEXT not in str(retry.content)
    assert effects == [7]
    assert len(replies) == len(relay.metadata) == 1
