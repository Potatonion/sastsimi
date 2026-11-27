from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from sastsimi.config.user_config import SimpleToolBinding
from sastsimi.simple_runtime.artifacts import SimpleArtifactRepository
from sastsimi.simple_runtime.attempt_owner import AttemptOwner
from sastsimi.simple_runtime.call_queue import RunUsageBudget
from sastsimi.simple_runtime.claude_provider import (
    ClaudeBoundaryError,
    ClaudeCLIResponse,
    ClaudeProvider,
    ClaudeTransportError,
    OfficialClaudeCLITransport,
    _parse_stream,
    _run_child,
    _terminate_process_tree,
)
from sastsimi.simple_runtime.models import CheckpointIdentity, StageFailure
from sastsimi.simple_runtime.provider import SimpleLLMCallResult
from sastsimi.simple_runtime.store import SimpleCheckpointStore


def _binding(tmp_path: Path) -> SimpleToolBinding:
    executable = tmp_path / "claude.exe"
    executable.write_bytes(b"fake executable")
    return SimpleToolBinding(
        executable_path=executable,
        version="2.1.280",
        executable_sha256=hashlib.sha256(b"fake executable").hexdigest(),
    )


def _stream(model: str, *, tools: list[str] | None = None) -> bytes:
    events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session-1",
            "tools": ["StructuredOutput"] if tools is None else tools,
            "mcp_servers": [],
            "plugins": [],
            "slash_commands": [],
            "skills": [],
            "apiKeySource": "none",
            "permissionMode": "dontAsk",
            "model": model,
            "claude_code_version": "2.1.280",
            "agents": ["claude", "Explore", "general-purpose", "Plan"],
        },
        {
            "type": "result",
            "session_id": "session-1",
            "is_error": False,
            "permission_denials": [],
            "structured_output": {"answer": "yes"},
            "total_cost_usd": 0.01,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    ]
    return b"\n".join(json.dumps(event).encode() for event in events) + b"\n"


@pytest.mark.asyncio
async def test_claude_cli_uses_no_tools_and_stdin_only(tmp_path: Path) -> None:
    calls: list[tuple[tuple[str, ...], bytes | None, dict[str, str]]] = []

    async def fake_runner(
        argv: tuple[str, ...],
        *,
        stdin: bytes | None,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes, bytes]:
        calls.append((argv, stdin, dict(env)))
        if "--version" in argv:
            return 0, b"2.1.280 (Claude Code)\n", b""
        if "auth" in argv:
            return (
                0,
                b'{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"pro"}',
                b"",
            )
        return 0, _stream("operator-model"), b""

    transport = OfficialClaudeCLITransport(
        _binding(tmp_path), tmp_path / "config", runner=fake_runner
    )
    response = await transport.invoke(
        prompt=b"secret prompt",
        output_schema={
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        },
        model="operator-model",
        timeout=10,
    )
    assert isinstance(response, ClaudeCLIResponse)
    assert response.value == {"answer": "yes"}
    assert response.cost_minor_units == 1.0
    assert response.input_tokens == 5
    argv, stdin, env = calls[-1]
    assert stdin == b"secret prompt"
    assert "secret prompt" not in " ".join(argv)
    assert argv[argv.index("--tools") + 1] == ""
    assert "--json-schema" in argv
    assert "ANTHROPIC_API_KEY" not in env
    assert "PATH" not in env
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_claude_cli_rejects_tool_in_effective_init(tmp_path: Path) -> None:
    async def fake_runner(
        argv: tuple[str, ...],
        *,
        stdin: bytes | None,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes, bytes]:
        if "--version" in argv:
            return 0, b"2.1.280 (Claude Code)\n", b""
        if "auth" in argv:
            return (
                0,
                b'{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"pro"}',
                b"",
            )
        return 0, _stream("operator-model", tools=["Read", "StructuredOutput"]), b""

    transport = OfficialClaudeCLITransport(
        _binding(tmp_path), tmp_path / "config", runner=fake_runner
    )
    with pytest.raises(ClaudeBoundaryError):
        await transport.invoke(
            prompt=b"hello",
            output_schema={"type": "object"},
            model="operator-model",
            timeout=10,
        )


@pytest.mark.asyncio
async def test_nonzero_exit_after_success_is_not_retried(tmp_path: Path) -> None:
    async def fake_runner(
        argv: tuple[str, ...],
        *,
        stdin: bytes | None,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes, bytes]:
        if "--version" in argv:
            return 0, b"2.1.280 (Claude Code)\n", b""
        if "auth" in argv:
            return (
                0,
                b'{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"pro"}',
                b"",
            )
        return 1, _stream("operator-model"), b""

    transport = OfficialClaudeCLITransport(
        _binding(tmp_path), tmp_path / "config", runner=fake_runner
    )
    with pytest.raises(ClaudeTransportError) as failure:
        await transport.invoke(
            prompt=b"hello",
            output_schema={"type": "object"},
            model="operator-model",
            timeout=10,
        )
    assert failure.value.retryable is False


@pytest.mark.asyncio
async def test_nonzero_unknown_model_result_has_terminal_safe_error(
    tmp_path: Path,
) -> None:
    model = "unavailable-model"
    events = [json.loads(line) for line in _stream(model).splitlines()]
    events[-1].update(
        {
            "is_error": True,
            "subtype": "success",
            "errors": [],
            "api_error_status": 404,
            "result": (
                f"There's an issue with the selected model ({model}). "
                "It may not exist or you may not have access to it."
            ),
        }
    )
    del events[-1]["structured_output"]
    raw = b"\n".join(json.dumps(event).encode() for event in events) + b"\n"

    async def fake_runner(
        argv: tuple[str, ...],
        *,
        stdin: bytes | None,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes, bytes]:
        if "--version" in argv:
            return 0, b"2.1.280 (Claude Code)\n", b""
        if "auth" in argv:
            return (
                0,
                b'{"loggedIn":true,"authMethod":"claude.ai","apiProvider":"firstParty","subscriptionType":"pro"}',
                b"",
            )
        return 1, raw, b"secret from stderr"

    transport = OfficialClaudeCLITransport(
        _binding(tmp_path), tmp_path / "config", runner=fake_runner
    )
    with pytest.raises(ClaudeTransportError) as failure:
        await transport.invoke(
            prompt=b"secret prompt",
            output_schema={"type": "object"},
            model=model,
            timeout=10,
        )
    assert failure.value.code == "CLAUDE_MODEL_UNSUPPORTED"
    assert failure.value.retryable is False
    assert str(failure.value) == "CLAUDE_MODEL_UNSUPPORTED"


@pytest.mark.asyncio
async def test_broken_stdin_terminates_child_and_becomes_typed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenStdin:
        def write(self, _value: bytes) -> None:
            pass

        async def drain(self) -> None:
            raise BrokenPipeError("child exited")

        def close(self) -> None:
            pass

    class FakeProcess:
        pid = None

        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.stdin = BrokenStdin()
            self.returncode: int | None = None
            self.killed = False

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()

    async def fake_spawn(*_args: object, **_kwargs: object) -> FakeProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    with pytest.raises(ClaudeTransportError):
        await _run_child(("fake",), stdin=b"prompt", cwd=tmp_path, env={}, timeout=1)
    assert process.killed


class _DelayedChild:
    pid = None
    stdin = None

    def __init__(self, delay: float = 0.8) -> None:
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.returncode: int | None = None
        self.done = asyncio.Event()
        asyncio.get_running_loop().call_later(delay, self._finish)

    def _finish(self) -> None:
        if self.returncode is None:
            self.returncode = 0
            self.done.set()

    def kill(self) -> None:
        self.returncode = -9
        self.done.set()

    async def wait(self) -> int:
        await self.done.wait()
        assert self.returncode is not None
        return self.returncode


@pytest.mark.asyncio
async def test_concurrent_claude_children_stagger_starts_but_overlap_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    starts: list[float] = []
    finishes: list[float] = []

    class TimedChild(_DelayedChild):
        async def wait(self) -> int:
            code = await super().wait()
            finishes.append(loop.time())
            return code

    async def fake_spawn(*_args: object, **_kwargs: object) -> TimedChild:
        starts.append(loop.time())
        return TimedChild()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    await asyncio.gather(
        _run_child(("fake",), stdin=None, cwd=tmp_path, env={}, timeout=3),
        _run_child(("fake",), stdin=None, cwd=tmp_path, env={}, timeout=3),
    )

    assert len(starts) == 2
    assert starts[1] - starts[0] >= 0.45
    assert starts[1] < min(finishes)


@pytest.mark.asyncio
async def test_cancelled_queued_claude_child_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = asyncio.Event()
    starts: list[float] = []

    async def fake_spawn(*_args: object, **_kwargs: object) -> _DelayedChild:
        starts.append(asyncio.get_running_loop().time())
        spawned.set()
        return _DelayedChild()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    first = asyncio.create_task(
        _run_child(("fake",), stdin=None, cwd=tmp_path, env={}, timeout=3)
    )
    await spawned.wait()
    second = asyncio.create_task(
        _run_child(("fake",), stdin=None, cwd=tmp_path, env={}, timeout=3)
    )
    await asyncio.sleep(0.05)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    await first

    assert len(starts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancel_terminates_claude_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    class HangingProcess:
        pid = None
        stdin = None

        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.returncode: int | None = None
            self.killed = False
            self.done = asyncio.Event()

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9
            self.done.set()

        async def wait(self) -> int:
            await self.done.wait()
            assert self.returncode is not None
            return self.returncode

    process = HangingProcess()

    async def fake_spawn(*_args: object, **_kwargs: object) -> HangingProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    task = asyncio.create_task(
        _run_child(
            ("fake",),
            stdin=None,
            cwd=tmp_path,
            env={},
            timeout=0.05 if not cancel else 10,
        )
    )
    if cancel:
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(TimeoutError):
            await task
    assert process.killed


@pytest.mark.asyncio
async def test_windows_termination_uses_recursive_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows process-tree path")

    class Target:
        pid = 123
        returncode: int | None = None

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    target = Target()
    launched: list[tuple[object, ...]] = []

    class Killer:
        async def wait(self) -> int:
            target.returncode = -9
            return 0

    async def fake_spawn(*argv: object, **_kwargs: object) -> Killer:
        launched.append(argv)
        return Killer()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    await _terminate_process_tree(target)  # type: ignore[arg-type]
    assert launched == [("taskkill", "/PID", "123", "/T", "/F")]


class FakeTransport:
    def __init__(self, outcomes: list[ClaudeCLIResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, bytes]] = []

    async def invoke(
        self,
        *,
        prompt: bytes,
        output_schema: Mapping[str, Any],
        model: str,
        timeout: float,
    ) -> ClaudeCLIResponse:
        self.calls.append((model, prompt))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _provider(
    tmp_path: Path,
    fake: FakeTransport,
    *,
    retries: int = 2,
    budget_check: Callable[[], StageFailure | None] | None = None,
) -> ClaudeProvider:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    return ClaudeProvider(
        artifacts=SimpleArtifactRepository(tmp_path, identity),
        default_model="base-model",
        agent_models={"verification_result": "verification-model"},
        timeout_seconds=5,
        max_retries=retries,
        semaphore=asyncio.Semaphore(1),
        transport=fake,
        budget_check=budget_check,
    )


@pytest.mark.asyncio
async def test_claude_attempt_records_logical_owner(tmp_path: Path) -> None:
    fake = FakeTransport(
        [
            ClaudeCLIResponse(
                raw_output=b"raw stream",
                value={"verdict": "TRUE"},
                input_tokens=4,
                output_tokens=2,
            )
        ]
    )
    result = await _provider(tmp_path, fake).call(
        prompt=b"prompt",
        output_schema={
            "type": "object",
            "required": ["verdict"],
            "properties": {"verdict": {"type": "string"}},
        },
        timeout_ms=5_000,
        agent_name="verification_result",
        owner=AttemptOwner(
            analysis_id="analysis-1",
            stage="VERIFICATION_FINAL_DONE",
            hypothesis_id="hypothesis-1",
        ),
    )
    assert isinstance(result, SimpleLLMCallResult)
    with sqlite3.connect(tmp_path / "db" / "sastsimi.sqlite3") as connection:
        rows = connection.execute(
            "SELECT m.stage, m.hypothesis_id FROM simple_llm_attempts AS a "
            "JOIN simple_llm_attempt_metadata AS m ON m.attempt_id = a.attempt_id "
            "WHERE a.analysis_id = ?",
            ("analysis-1",),
        ).fetchall()
    assert rows == [("VERIFICATION_FINAL_DONE", "hypothesis-1")]


@pytest.mark.asyncio
async def test_claude_adapter_preserves_schema_model_and_distinct_artifacts(
    tmp_path: Path,
) -> None:
    fake = FakeTransport(
        [
            ClaudeCLIResponse(
                raw_output=b"raw stream",
                value={"verdict": "TRUE"},
                input_tokens=4,
                output_tokens=2,
            )
        ]
    )
    result = await _provider(tmp_path, fake).call(
        prompt=b"Verification Agent prompt",
        output_schema={
            "type": "object",
            "required": ["verdict"],
            "properties": {"verdict": {"type": "string", "enum": ["TRUE", "FALSE"]}},
            "additionalProperties": False,
        },
        timeout_ms=5_000,
        agent_name="verification_result",
    )
    assert isinstance(result, SimpleLLMCallResult)
    assert result.value == {"verdict": "TRUE"}
    assert result.model == "verification-model"
    assert result.raw_output_ref != result.parsed_output_ref
    assert fake.calls == [("verification-model", b"Verification Agent prompt")]
    assert result.input_tokens == 4


@pytest.mark.asyncio
async def test_claude_budget_blocks_completion_before_charging(tmp_path: Path) -> None:
    fake = FakeTransport([])
    blocked = StageFailure(
        code="LLM_TOKEN_BUDGET_EXHAUSTED",
        retryable=False,
        safe_message="limit",
    )
    result = await _provider(tmp_path, fake, budget_check=lambda: blocked).call(
        prompt=b"safe",
        output_schema={},
        timeout_ms=1000,
    )
    assert isinstance(result, StageFailure)
    assert result.code == "LLM_TOKEN_BUDGET_EXHAUSTED"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_claude_schema_mismatch_is_retried_then_fails(tmp_path: Path) -> None:
    fake = FakeTransport(
        [
            ClaudeCLIResponse(raw_output=b"first", value={"wrong": True}),
            ClaudeCLIResponse(raw_output=b"second", value={"wrong": True}),
        ]
    )
    result = await _provider(tmp_path, fake, retries=1).call(
        prompt=b"agent prompt",
        output_schema={
            "type": "object",
            "required": ["verdict"],
            "properties": {"verdict": {"type": "string"}},
        },
        timeout_ms=5_000,
        agent_name="verification_result",
    )
    assert isinstance(result, StageFailure)
    assert result.code == "CLAUDE_INVALID_OUTPUT"
    assert len(fake.calls) == 2
    assert b"$.verdict" in fake.calls[1][1]


@pytest.mark.asyncio
async def test_claude_auth_failure_is_not_retried(tmp_path: Path) -> None:
    fake = FakeTransport([ClaudeTransportError("CLAUDE_AUTH_REQUIRED")])
    result = await _provider(tmp_path, fake).call(
        prompt=b"hello", output_schema={"type": "object"}, timeout_ms=5_000
    )
    assert isinstance(result, StageFailure)
    assert result.code == "CLAUDE_AUTH_REQUIRED"
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_claude_rate_limit_retries_once_then_succeeds(tmp_path: Path) -> None:
    fake = FakeTransport(
        [
            ClaudeTransportError("CLAUDE_RATE_LIMITED", retryable=True),
            ClaudeCLIResponse(raw_output=b"response", value={"ok": True}),
        ]
    )
    budget = RunUsageBudget(
        store=SimpleCheckpointStore(tmp_path / "db" / "sastsimi.sqlite3"),
        analysis_id="analysis-1",
        max_tokens=1000,
        max_cost_minor_units=1000,
        max_elapsed_seconds="unlimited",
    )
    result = await _provider(tmp_path, fake, retries=1, budget_check=budget.check).call(
        prompt=b"agent prompt",
        output_schema={
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        timeout_ms=5_000,
    )
    assert isinstance(result, SimpleLLMCallResult)
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_claude_fractional_cost_is_recorded_without_invalid_artifact(
    tmp_path: Path,
) -> None:
    fake = FakeTransport(
        [
            ClaudeCLIResponse(
                raw_output=b"response",
                value={"ok": True},
                input_tokens=4,
                output_tokens=2,
                cost_minor_units=1.5,
            )
        ]
    )
    result = await _provider(tmp_path, fake).call(
        prompt=b"agent prompt",
        output_schema={
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        timeout_ms=5_000,
    )

    assert isinstance(result, SimpleLLMCallResult)
    assert result.cost_minor_units == 1.5


_MODEL = "operator-model"


def _init_event() -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "init",
        "session_id": "session-1",
        "tools": ["StructuredOutput"],
        "mcp_servers": [],
        "plugins": [],
        "slash_commands": [],
        "skills": [],
        "apiKeySource": "none",
        "permissionMode": "dontAsk",
        "model": _MODEL,
        "claude_code_version": "2.1.280",
        "agents": ["claude", "Explore", "general-purpose", "Plan"],
    }


def _structured_call_event(tool_id: str = "toolu_1") -> dict[str, Any]:
    return {
        "type": "assistant",
        "session_id": "session-1",
        "parent_tool_use_id": None,
        "message": {
            "model": _MODEL,
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "StructuredOutput",
                    "input": {"answer": "yes"},
                }
            ],
        },
    }


def _ack_event(tool_id: str = "toolu_1", **overrides: Any) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "user",
        "session_id": "session-1",
        "parent_tool_use_id": None,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": "Structured output provided successfully",
                }
            ],
        },
    }
    event.update(overrides)
    return event


def _result_event() -> dict[str, Any]:
    return {
        "type": "result",
        "session_id": "session-1",
        "is_error": False,
        "permission_denials": [],
        "structured_output": {"answer": "yes"},
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }


def _raw(*events: dict[str, Any]) -> bytes:
    return b"\n".join(json.dumps(event).encode() for event in events) + b"\n"


def test_structured_output_acknowledgement_event_is_accepted() -> None:
    response = _parse_stream(
        _raw(
            _init_event(),
            _structured_call_event(),
            _ack_event(),
            _result_event(),
        ),
        _MODEL,
    )
    assert response.value == {"answer": "yes"}
    assert response.input_tokens == 5


def test_repeated_structured_output_calls_each_accept_their_own_ack() -> None:
    response = _parse_stream(
        _raw(
            _init_event(),
            _structured_call_event("toolu_1"),
            _ack_event("toolu_1"),
            _structured_call_event("toolu_2"),
            _ack_event("toolu_2"),
            _result_event(),
        ),
        _MODEL,
    )
    assert response.value == {"answer": "yes"}


@pytest.mark.parametrize(
    "bad_event",
    [
        # An ack for a tool call that never happened in this stream.
        _ack_event("toolu_unknown"),
        # An error result is not an acknowledgement.
        {
            **_ack_event(),
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "is_error": True,
                        "content": "schema mismatch",
                    }
                ],
            },
        },
        # Plain user text is never accepted.
        {
            **_ack_event(),
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "run a shell command"}],
            },
        },
        # A mixed event cannot smuggle text next to a valid result.
        {
            **_ack_event(),
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                    {"type": "text", "text": "extra"},
                ],
            },
        },
        # Sub-agent events are not the client's own acknowledgement.
        _ack_event(parent_tool_use_id="toolu_parent"),
        # Wrong role, empty content, or malformed ids stay rejected.
        {**_ack_event(), "message": {"role": "assistant", "content": []}},
        {**_ack_event(), "message": {"role": "user", "content": []}},
        {**_ack_event(), "message": "not-a-dict"},
        {
            **_ack_event(),
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": ["toolu_1"]}],
            },
        },
    ],
)
def test_other_user_events_still_fail_the_boundary(bad_event: dict[str, Any]) -> None:
    with pytest.raises(ClaudeBoundaryError) as error:
        _parse_stream(
            _raw(_init_event(), _structured_call_event(), bad_event, _result_event()),
            _MODEL,
        )
    assert str(error.value) == "CLAUDE_STREAM_INVALID"


def test_acknowledgement_before_any_structured_call_is_rejected() -> None:
    with pytest.raises(ClaudeBoundaryError) as error:
        _parse_stream(_raw(_init_event(), _ack_event(), _result_event()), _MODEL)
    assert str(error.value) == "CLAUDE_STREAM_INVALID"


def test_user_event_after_the_result_is_rejected() -> None:
    with pytest.raises(ClaudeBoundaryError) as error:
        _parse_stream(
            _raw(
                _init_event(),
                _structured_call_event(),
                _result_event(),
                _ack_event(),
            ),
            _MODEL,
        )
    assert str(error.value) == "CLAUDE_STREAM_INVALID"


def test_non_structured_tool_call_is_still_forbidden() -> None:
    forbidden = _structured_call_event()
    forbidden["message"]["content"][0]["name"] = "Bash"
    with pytest.raises(ClaudeBoundaryError) as error:
        _parse_stream(
            _raw(_init_event(), forbidden, _ack_event(), _result_event()), _MODEL
        )
    assert str(error.value) == "CLAUDE_TOOL_FORBIDDEN"


def _result_with_usage(usage: dict[str, Any]) -> dict[str, Any]:
    return {**_result_event(), "usage": usage}


def _parse_with_usage(usage: dict[str, Any]) -> ClaudeCLIResponse:
    return _parse_stream(
        _raw(
            _init_event(),
            _structured_call_event(),
            _ack_event(),
            _result_with_usage(usage),
        ),
        _MODEL,
    )


def test_input_tokens_include_prompt_cache_traffic() -> None:
    response = _parse_with_usage(
        {
            "input_tokens": 5,
            "cache_creation_input_tokens": 1_000,
            "cache_read_input_tokens": 20_000,
            "output_tokens": 2,
        }
    )
    assert response.input_tokens == 21_005
    assert response.output_tokens == 2


def test_input_tokens_without_cache_fields_are_unchanged() -> None:
    response = _parse_with_usage({"input_tokens": 5, "output_tokens": 2})
    assert response.input_tokens == 5


@pytest.mark.parametrize("bad", [-1, "10", None, True, 1.5])
def test_invalid_cache_field_makes_input_tokens_unknown(bad: object) -> None:
    response = _parse_with_usage(
        {"input_tokens": 5, "cache_read_input_tokens": bad, "output_tokens": 2}
    )
    assert response.input_tokens is None
    assert response.output_tokens == 2
