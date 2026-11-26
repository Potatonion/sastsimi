from __future__ import annotations

import asyncio
import io
import json
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from sastsimi.simple_runtime.artifacts import SimpleArtifactRepository
from sastsimi.simple_runtime.call_queue import RunUsageBudget
from sastsimi.simple_runtime.cursor_provider import (
    CursorCLIExecutionError,
    CursorCLIPromptTooLargeError,
    CursorCLIRateLimitError,
    CursorCLISafetyBlockError,
    CursorModelCatalog,
    CursorProvider,
    OfficialCursorCLITransport,
    _error_code,
    _invalid_output_correction,
)
from sastsimi.simple_runtime.models import CheckpointIdentity, StageFailure
from sastsimi.simple_runtime.provider import SimpleLLMCallResult
from sastsimi.simple_runtime.store import SimpleCheckpointStore


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class FakeTransport:
    def __init__(
        self, outcomes: list[str | BaseException], models: set[str] | None = None
    ):
        self.outcomes = outcomes
        self.models = models or {"available-model", "verification-model"}
        self.calls: list[tuple[str, str]] = []
        self.list_calls = 0

    async def list_models(self, api_key: str) -> set[str]:
        assert api_key in {"", "test-key"}
        self.list_calls += 1
        return self.models

    async def complete(
        self, *, api_key: str, model: str, prompt: str, timeout: float
    ) -> tuple[str, dict[str, int | float | None]]:
        assert api_key in {"", "test-key"}
        self.calls.append((model, prompt))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome, {"input_tokens": 12, "output_tokens": 3}


def provider(
    tmp_path: Path,
    transport: FakeTransport,
    *,
    max_retries: int = 2,
    allow_on_demand: bool = True,
    agent_models: Mapping[str, str] | None = None,
    use_cli_login: bool = False,
    model_catalog: CursorModelCatalog | None = None,
    budget_check: Callable[[], StageFailure | None] | None = None,
    poc_transport: FakeTransport | None = None,
) -> CursorProvider:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    return CursorProvider(
        artifacts=SimpleArtifactRepository(tmp_path, identity),
        default_model="available-model",
        agent_models=agent_models or {},
        timeout_seconds=5,
        max_retries=max_retries,
        semaphore=asyncio.Semaphore(1),
        allow_on_demand=allow_on_demand,
        transport=transport,
        poc_transport=poc_transport,
        use_cli_login=use_cli_login,
        model_catalog=model_catalog,
        budget_check=budget_check,
    )


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["TRUE", "FALSE"]}},
    "required": ["verdict"],
    "additionalProperties": False,
}


async def call(client: CursorProvider) -> SimpleLLMCallResult | StageFailure:
    return await client.call(
        prompt=b"You are the Verification Agent.",
        output_schema=SCHEMA,
        timeout_ms=5_000,
        agent_name="verification_result",
    )


@pytest.mark.asyncio
async def test_valid_json_and_separate_raw_parsed_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(['{"verdict":"TRUE"}'])
    client = provider(
        tmp_path, fake, agent_models={"verification_result": "verification-model"}
    )
    result = await call(client)
    assert isinstance(result, SimpleLLMCallResult)
    assert result.value == {"verdict": "TRUE"}
    assert result.model == "verification-model"
    assert result.raw_output_ref != result.parsed_output_ref
    assert result.input_tokens == 12
    assert result.on_demand_possible
    with sqlite3.connect(tmp_path / "db" / "sastsimi.sqlite3") as connection:
        attempt = connection.execute(
            "SELECT input_tokens, output_tokens, status, artifact_ref_json "
            "FROM simple_llm_attempts WHERE analysis_id = ?",
            ("analysis-1",),
        ).fetchone()
    assert attempt is not None
    assert attempt[:3] == (12, 3, "SUCCEEDED")
    assert "content_hash" in attempt[3]


@pytest.mark.asyncio
async def test_code_fence_is_unwrapped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    result = await call(
        provider(tmp_path, FakeTransport(['```json\n{"verdict":"FALSE"}\n```']))
    )
    assert isinstance(result, SimpleLLMCallResult)
    assert result.value["verdict"] == "FALSE"


@pytest.mark.asyncio
async def test_one_prefaced_json_object_is_schema_validated_and_raw_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    raw = 'I checked the evidence. {"verdict":"TRUE"}'
    client = provider(tmp_path, FakeTransport([raw]))

    result = await call(client)

    assert isinstance(result, SimpleLLMCallResult)
    assert result.value == {"verdict": "TRUE"}
    assert result.raw_output_ref is not None
    assert result.parsed_output_ref is not None
    artifacts = client._artifacts
    assert json.loads(artifacts.read(result.raw_output_ref))["text"] == raw
    assert json.loads(artifacts.read(result.parsed_output_ref)) == result.value


@pytest.mark.asyncio
async def test_multiple_prefaced_json_objects_are_not_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    client = provider(
        tmp_path,
        FakeTransport(['Here: {"verdict":"TRUE"} {"verdict":"FALSE"}']),
        max_retries=0,
    )

    result = await call(client)

    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_INVALID_OUTPUT"


@pytest.mark.asyncio
async def test_invalid_json_and_schema_get_bounded_reprompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(["not json", '{"verdict":"MAYBE"}', '{"verdict":"TRUE"}'])
    result = await call(provider(tmp_path, fake))
    assert isinstance(result, SimpleLLMCallResult)
    assert len(fake.calls) == 3
    schema_text = json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True)
    assert fake.calls[0][1].count(schema_text) == 1
    assert fake.calls[1][1].count(schema_text) == 1
    assert "validation" in fake.calls[1][1]
    assert "already given above" in fake.calls[1][1]


@pytest.mark.asyncio
async def test_invalid_output_exhaustion_is_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(["not json", "still not json"])
    result = await call(provider(tmp_path, fake, max_retries=1))
    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_INVALID_OUTPUT"
    assert len(fake.calls) == 2
    with sqlite3.connect(tmp_path / "db" / "sastsimi.sqlite3") as connection:
        attempts = connection.execute(
            "SELECT COUNT(*) FROM simple_llm_attempts WHERE analysis_id = ?",
            ("analysis-1",),
        ).fetchone()
    assert attempts == (2,)


@pytest.mark.asyncio
async def test_poc_refusal_is_preserved_without_schema_reprompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    refusal = "I can’t provide a PoC script for cross-user access."
    fake = FakeTransport([refusal])
    client = provider(tmp_path, fake, max_retries=2)

    result = await client.call(
        prompt=b"Plan an isolated local test.",
        output_schema=SCHEMA,
        timeout_ms=5_000,
        agent_name="poc_candidate",
    )

    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_REFUSED"
    assert not result.retryable
    assert len(fake.calls) == 1
    assert len(result.evidence_refs) == 1
    artifacts = SimpleArtifactRepository(
        tmp_path,
        CheckpointIdentity(
            analysis_id="analysis-1",
            workspace_id="workspace-1",
            commit_id="a" * 40,
            hypothesis_id="hypothesis-1",
        ),
    )
    assert json.loads(artifacts.read(result.evidence_refs[0]))["text"] == refusal
    with sqlite3.connect(tmp_path / "db" / "sastsimi.sqlite3") as connection:
        statuses = connection.execute(
            "SELECT status FROM simple_llm_attempts WHERE analysis_id = ?",
            ("analysis-1",),
        ).fetchall()
    assert statuses == [("CURSOR_REFUSED",)]


@pytest.mark.asyncio
async def test_valid_poc_json_after_refusal_like_commentary_is_not_mislabeled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    raw = (
        "I can't provide a PoC script as prose; returning the required JSON.\n"
        '{"verdict":"TRUE"}'
    )
    fake = FakeTransport([raw])

    result = await provider(tmp_path, fake, max_retries=0).call(
        prompt=b"Plan an isolated local test.",
        output_schema=SCHEMA,
        timeout_ms=5_000,
        agent_name="poc_candidate",
    )

    assert isinstance(result, SimpleLLMCallResult)
    assert result.value == {"verdict": "TRUE"}
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_poc_quality_failure_is_invalid_output_not_model_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(
        ["I can't write a PoC script because the source context is missing."]
    )

    result = await provider(tmp_path, fake, max_retries=0).call(
        prompt=b"Plan an isolated local test.",
        output_schema=SCHEMA,
        timeout_ms=5_000,
        agent_name="poc_candidate",
    )

    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_INVALID_OUTPUT"
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_text_poc_refusal_falls_back_once_and_remains_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(
        [
            "I can't provide a PoC script for cross-user access.",
            '{"verdict":"TRUE"}',
        ],
        models={"available-model", "poc-model"},
    )
    client = provider(
        tmp_path,
        fake,
        agent_models={"poc_candidate": "poc-model"},
        max_retries=0,
    )

    result = await client.call(
        prompt=b"Plan an isolated local test.",
        output_schema=SCHEMA,
        timeout_ms=5_000,
        agent_name="poc_candidate",
    )

    assert isinstance(result, SimpleLLMCallResult)
    assert result.model == "available-model"
    assert [model for model, _ in fake.calls] == ["poc-model", "available-model"]
    counts = SimpleCheckpointStore(
        tmp_path / "db" / "sastsimi.sqlite3"
    ).llm_attempt_status_counts("analysis-1", agent="poc_candidate")
    assert counts == {"CURSOR_REFUSED": 1, "SUCCEEDED": 1}


@pytest.mark.asyncio
async def test_korean_policy_refusal_is_not_treated_as_parse_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(
        ["못 쓰겠다. 안전 정책상 PoC 스크립트를 제공할 수 없다."]
    )

    result = await provider(tmp_path, fake, max_retries=0).call(
        prompt=b"Plan an isolated local test.",
        output_schema=SCHEMA,
        timeout_ms=5_000,
        agent_name="poc_candidate",
    )

    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_REFUSED"


@pytest.mark.asyncio
async def test_poc_calls_use_dedicated_transport_without_changing_other_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    default = FakeTransport(['{"verdict":"TRUE"}'])
    poc = FakeTransport(['{"verdict":"FALSE"}'])
    client = provider(tmp_path, default, poc_transport=poc)

    normal = await call(client)
    candidate = await client.call(
        prompt=b"Use the pinned source context.",
        output_schema=SCHEMA,
        timeout_ms=5_000,
        agent_name="poc_candidate",
    )

    assert isinstance(normal, SimpleLLMCallResult)
    assert isinstance(candidate, SimpleLLMCallResult)
    assert normal.value["verdict"] == "TRUE"
    assert candidate.value["verdict"] == "FALSE"
    assert len(default.calls) == 1
    assert len(poc.calls) == 1


@pytest.mark.asyncio
async def test_rate_limit_retries_but_auth_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    limited = FakeTransport([RateLimitError(), '{"verdict":"TRUE"}'])
    budget = RunUsageBudget(
        store=SimpleCheckpointStore(tmp_path / "db" / "sastsimi.sqlite3"),
        analysis_id="analysis-1",
        max_tokens=1000,
        max_cost_minor_units=1000,
        max_elapsed_seconds="unlimited",
    )
    result = await call(provider(tmp_path, limited, budget_check=budget.check))
    assert isinstance(result, SimpleLLMCallResult)
    assert len(limited.calls) == 2
    auth = FakeTransport([AuthenticationError()])
    result = await call(provider(tmp_path, auth))
    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_AUTH_FAILED"
    assert len(auth.calls) == 1


@pytest.mark.asyncio
async def test_invalid_negative_cursor_usage_cannot_reduce_run_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class InvalidUsageTransport(FakeTransport):
        async def complete(
            self, *, api_key: str, model: str, prompt: str, timeout: float
        ) -> tuple[str, dict[str, int | float | None]]:
            raw, _ = await super().complete(
                api_key=api_key, model=model, prompt=prompt, timeout=timeout
            )
            return raw, {
                "input_tokens": -100,
                "output_tokens": 3,
                "cost_minor_units": -4,
            }

    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    budget = RunUsageBudget(
        store=SimpleCheckpointStore(tmp_path / "db" / "sastsimi.sqlite3"),
        analysis_id="analysis-1",
        max_tokens=1000,
        max_cost_minor_units=1000,
        max_elapsed_seconds="unlimited",
    )
    result = await call(
        provider(
            tmp_path,
            InvalidUsageTransport(['{"verdict":"TRUE"}']),
            budget_check=budget.check,
        )
    )

    assert isinstance(result, SimpleLLMCallResult)
    assert result.input_tokens is None
    assert result.cost_minor_units is None
    failure = budget.check()
    assert isinstance(failure, StageFailure)
    assert failure.code == "LLM_TOKEN_USAGE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_cursor_fractional_cost_is_recorded_without_invalid_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class CostTransport(FakeTransport):
        async def complete(
            self, *, api_key: str, model: str, prompt: str, timeout: float
        ) -> tuple[str, dict[str, int | float | None]]:
            raw, usage = await super().complete(
                api_key=api_key, model=model, prompt=prompt, timeout=timeout
            )
            usage["cost_minor_units"] = 1.5
            return raw, usage

    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    result = await call(provider(tmp_path, CostTransport(['{"verdict":"TRUE"}'])))

    assert isinstance(result, SimpleLLMCallResult)
    assert result.cost_minor_units == 1.5
    with sqlite3.connect(tmp_path / "db" / "sastsimi.sqlite3") as connection:
        stored = connection.execute(
            "SELECT cost_cents FROM simple_llm_attempts WHERE analysis_id = ?",
            ("analysis-1",),
        ).fetchone()
    assert stored == (1.5,)


@pytest.mark.asyncio
async def test_missing_model_and_on_demand_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport([], models={"other-model"})
    result = await call(provider(tmp_path, fake))
    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_MODEL_UNAVAILABLE"
    assert not fake.calls
    result = await call(provider(tmp_path, fake, allow_on_demand=False))
    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_ON_DEMAND_CONTROL_UNAVAILABLE"


@pytest.mark.asyncio
async def test_cursor_budget_blocks_completion_before_charging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(['{"verdict":"TRUE"}'])
    blocked = StageFailure(
        code="LLM_COST_BUDGET_EXHAUSTED",
        retryable=False,
        safe_message="limit",
    )
    result = await call(provider(tmp_path, fake, budget_check=lambda: blocked))
    assert isinstance(result, StageFailure)
    assert result.code == "LLM_COST_BUDGET_EXHAUSTED"
    assert fake.calls == []


@pytest.mark.asyncio
async def test_timeout_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport([TimeoutError()])
    result = await call(provider(tmp_path, fake, max_retries=0))
    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_TIMED_OUT"


@pytest.mark.asyncio
async def test_active_call_is_visible_then_cleared_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    entered = asyncio.Event()
    release = asyncio.Event()

    class WaitingTransport(FakeTransport):
        async def complete(self, *, api_key, model, prompt, timeout):
            entered.set()
            await release.wait()
            return '{"verdict":"TRUE"}', {}

    task = asyncio.create_task(call(provider(tmp_path, WaitingTransport([]))))
    await entered.wait()
    database = tmp_path / "db" / "sastsimi.sqlite3"
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT agent, model, attempt_number FROM simple_llm_inflight"
        ).fetchone()
    assert row == ("verification_result", "available-model", 1)
    release.set()
    assert isinstance(await task, SimpleLLMCallResult)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM simple_llm_inflight"
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_cancelled_call_clears_active_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    entered = asyncio.Event()

    class WaitingTransport(FakeTransport):
        async def complete(self, *, api_key, model, prompt, timeout):
            entered.set()
            await asyncio.Event().wait()
            return '{"verdict":"TRUE"}', {}

    task = asyncio.create_task(call(provider(tmp_path, WaitingTransport([]))))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with sqlite3.connect(tmp_path / "db" / "sastsimi.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM simple_llm_inflight"
        ).fetchone() == (0,)


@pytest.mark.asyncio
async def test_timeout_budget_includes_retries_and_backoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")

    class SlowTransport(FakeTransport):
        async def complete(self, *, api_key, model, prompt, timeout):
            self.calls.append((model, prompt))
            await asyncio.sleep(60)
            return '{"verdict":"TRUE"}', {}

    fake = SlowTransport([])
    client = provider(tmp_path, fake, max_retries=2)
    started = asyncio.get_running_loop().time()
    result = await client.call(
        prompt=b"Return one harmless JSON object.",
        output_schema=SCHEMA,
        timeout_ms=50,
        agent_name="test_agent",
    )
    elapsed = asyncio.get_running_loop().time() - started
    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_TIMED_OUT"
    assert len(fake.calls) == 1
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_backoff_cannot_start_a_retry_after_call_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport([RateLimitError(), '{"verdict":"TRUE"}'])
    result = await provider(tmp_path, fake).call(
        prompt=b"Return one harmless JSON object.",
        output_schema=SCHEMA,
        timeout_ms=50,
    )
    assert isinstance(result, StageFailure)
    assert result.code == "CURSOR_RATE_LIMITED"
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_cli_login_needs_no_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    result = await call(
        provider(
            tmp_path,
            FakeTransport(['{"verdict":"TRUE"}']),
            use_cli_login=True,
        )
    )
    assert isinstance(result, SimpleLLMCallResult)
    assert result.provider == "cursor-cli"


@pytest.mark.asyncio
async def test_model_catalog_is_shared_between_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(['{"verdict":"TRUE"}', '{"verdict":"FALSE"}'])
    catalog = CursorModelCatalog()
    assert isinstance(
        await call(provider(tmp_path, fake, model_catalog=catalog)),
        SimpleLLMCallResult,
    )
    assert isinstance(
        await call(provider(tmp_path, fake, model_catalog=catalog)),
        SimpleLLMCallResult,
    )
    assert fake.list_calls == 1


@pytest.mark.asyncio
async def test_cli_transport_parses_only_verified_catalog_and_json_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = tmp_path / "node.exe"
    native.write_bytes(b"test")
    (tmp_path / "index.js").write_bytes(b"test")
    calls: list[tuple[str, ...]] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            if len(calls) == 1:
                return b"Available models\nauto - Auto\ncomposer-2.5 - Composer\n", b""
            return (
                b'{"type":"result","subtype":"success","is_error":false,'
                b'"result":"{\\"verdict\\":\\"TRUE\\"}"}',
                b"",
            )

    async def fake_subprocess(*args: str, **kwargs: Any) -> Process:
        assert kwargs["cwd"] != str(tmp_path)
        assert "CURSOR_API_KEY" not in kwargs["env"]
        calls.append(args)
        return Process()

    monkeypatch.setenv("CURSOR_API_KEY", "must-not-leak")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    transport = OfficialCursorCLITransport(str(native))
    assert await transport.list_models("") == {"auto", "composer-2.5"}
    raw, usage = await transport.complete(
        api_key="", model="composer-2.5", prompt="test prompt", timeout=5
    )
    assert raw == '{"verdict":"TRUE"}'
    assert usage == {}
    assert calls[1][:2] == (str(native), str(tmp_path / "index.js"))
    assert "--mode" in calls[1] and "ask" in calls[1]
    assert "--trust" in calls[1]
    assert calls[1][-2:] == ("--", "test prompt")

    large_prompt = (
        '{"facts":['
        + ",".join(f'"fact-{i:05d}"' for i in range(18000))
        + "]}"
    )
    await transport.complete(
        api_key="", model="composer-2.5", prompt=large_prompt, timeout=5
    )
    split_args = calls[2][calls[2].index("--") + 1 :]
    assert len(split_args) > 1
    assert json.loads(" ".join(split_args)) == json.loads(large_prompt)


@pytest.mark.asyncio
async def test_cli_stream_json_records_event_metadata_without_prompt_or_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = tmp_path / "node.exe"
    native.write_bytes(b"test")
    (tmp_path / "index.js").write_bytes(b"test")
    observed: dict[str, Any] = {}

    class Process:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            for event in (
                {"type": "system", "subtype": "init", "model": "test-model"},
                {"type": "assistant", "message": "PRIVATE_OUTPUT"},
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": '{"verdict":"TRUE","detail":"PRIVATE_OUTPUT"}',
                },
            ):
                self.stdout.feed_data(json.dumps(event).encode() + b"\n")
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        async def wait(self) -> int:
            return self.returncode

    async def fake_subprocess(*args: str, **kwargs: Any) -> Process:
        observed["args"] = args
        assert kwargs["limit"] >= 2 * 1024 * 1024
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    trace_dir = tmp_path / "logs"
    transport = OfficialCursorCLITransport(
        str(native), stream_json=True, trace_dir=trace_dir, trace_label="analysis-1"
    )
    raw, usage = await transport.complete(
        api_key="", model="test-model", prompt="PRIVATE_PROMPT", timeout=5
    )

    assert raw == '{"verdict":"TRUE","detail":"PRIVATE_OUTPUT"}'
    assert usage == {}
    format_index = observed["args"].index("--output-format") + 1
    assert observed["args"][format_index] == "stream-json"
    traces = list(trace_dir.glob("cursor-poc-analysis-1-*.jsonl"))
    assert len(traces) == 1
    trace_text = traces[0].read_text()
    assert "PRIVATE_PROMPT" not in trace_text
    assert "PRIVATE_OUTPUT" not in trace_text
    events = [json.loads(line) for line in trace_text.splitlines()]
    assert [event["event"] for event in events] == [
        "started",
        "event",
        "event",
        "event",
        "exited",
    ]
    assert [event["type"] for event in events[1:4]] == [
        "system",
        "assistant",
        "result",
    ]
    assert events[-1]["result_seen"] is True


@pytest.mark.asyncio
async def test_cli_stream_json_keeps_progress_after_initial_trace_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            for _ in range(6):
                self.stdout.feed_data(b'{"type":"stream_delta","message":"PRIVATE"}\n')
            self.stdout.feed_data(
                b'{"type":"result","subtype":"success","is_error":false,'
                b'"result":"{}"}\n'
            )
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        async def wait(self) -> int:
            return self.returncode

    monkeypatch.setattr(OfficialCursorCLITransport, "_MAX_STREAM_EVENTS", 2)
    monkeypatch.setattr(OfficialCursorCLITransport, "_TRACE_PROGRESS_INTERVAL", 2)
    trace = io.StringIO()

    stdout, stderr = await OfficialCursorCLITransport._collect_stream_json(
        Process(), trace
    )

    assert stderr == b""
    assert b'"type": "result"' in stdout
    events = [json.loads(line) for line in trace.getvalue().splitlines()]
    assert [event["event"] for event in events] == [
        "event",
        "event",
        "event_limit_reached",
        "event_progress",
        "event_progress",
        "exited",
    ]
    assert [
        event["event_count"]
        for event in events
        if event["event"] == "event_progress"
    ] == [4, 6]
    assert events[0]["type"] == "other"
    assert len(events[0]["type_hash"]) == 12
    assert events[-2]["stream_bytes"] > 0
    assert events[-2]["last_type_hash"] == events[0]["type_hash"]
    assert "PRIVATE" not in trace.getvalue()


@pytest.mark.asyncio
async def test_cli_stream_json_timeout_preserves_partial_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = tmp_path / "node.exe"
    native.write_bytes(b"test")
    (tmp_path / "index.js").write_bytes(b"test")

    class Process:
        returncode: int | None = None

        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_data(b'{"type":"system","subtype":"init"}\n')
            self.stopped = asyncio.Event()

        def kill(self) -> None:
            self.returncode = -9
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.stopped.set()

        async def wait(self) -> int:
            await self.stopped.wait()
            assert self.returncode is not None
            return self.returncode

    async def fake_subprocess(*args: str, **kwargs: Any) -> Process:
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    trace_dir = tmp_path / "logs"
    transport = OfficialCursorCLITransport(
        str(native), stream_json=True, trace_dir=trace_dir
    )
    with pytest.raises(TimeoutError):
        await transport.complete(
            api_key="", model="test-model", prompt="PRIVATE_PROMPT", timeout=0.05
        )

    traces = list(trace_dir.glob("cursor-poc-cursor-*.jsonl"))
    assert len(traces) == 1
    events = [json.loads(line) for line in traces[0].read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "started",
        "event",
        "timed_out",
    ]
    assert "PRIVATE_PROMPT" not in traces[0].read_text()


@pytest.mark.asyncio
async def test_cli_stream_json_requires_terminal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = tmp_path / "node.exe"
    native.write_bytes(b"test")
    (tmp_path / "index.js").write_bytes(b"test")

    class Process:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.stderr = asyncio.StreamReader()
            self.stdout.feed_data(b'{"type":"system","subtype":"init"}\n')
            self.stdout.feed_eof()
            self.stderr.feed_eof()

        async def wait(self) -> int:
            return self.returncode

    async def fake_subprocess(*args: str, **kwargs: Any) -> Process:
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    transport = OfficialCursorCLITransport(str(native), stream_json=True)
    with pytest.raises(CursorCLIExecutionError):
        await transport.complete(
            api_key="", model="test-model", prompt="test", timeout=5
        )


def test_openai_cyber_safety_stderr_is_not_retried_on_the_same_model() -> None:
    with pytest.raises(CursorCLISafetyBlockError):
        OfficialCursorCLITransport._raise_cli_error(
            b"ActionRequiredError: OpenAI flagged this request for potential "
            b"high-risk cybersecurity activity."
        )
    code, retryable = _error_code(CursorCLISafetyBlockError())
    assert code == "CURSOR_MODEL_SAFETY_BLOCK"
    assert retryable is False


@pytest.mark.asyncio
async def test_safety_block_continues_on_the_default_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    fake = FakeTransport(
        [CursorCLISafetyBlockError(), '{"verdict":"TRUE"}'],
        models={"available-model", "gpt-blocked"},
    )
    client = provider(
        tmp_path,
        fake,
        agent_models={"poc_candidate": "gpt-blocked"},
        max_retries=0,
    )

    result = await client.call(
        prompt=b"You are the Verification Agent.",
        output_schema=SCHEMA,
        timeout_ms=5_000,
        agent_name="poc_candidate",
    )

    assert isinstance(result, SimpleLLMCallResult)
    assert result.model == "available-model"
    assert result.value == {"verdict": "TRUE"}
    assert [model for model, _prompt in fake.calls] == [
        "gpt-blocked",
        "available-model",
    ]


def test_cli_execution_without_result_is_retryable() -> None:
    code, retryable = _error_code(CursorCLIExecutionError("no result"))

    assert code == "CURSOR_REQUEST_FAILED"
    assert retryable is True


@pytest.mark.asyncio
async def test_cli_poc_transport_stages_only_prompt_sources_in_sandbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = tmp_path / "node.exe"
    native.write_bytes(b"test")
    (tmp_path / "index.js").write_bytes(b"test")
    observed: dict[str, Any] = {}

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (
                b'{"type":"result","subtype":"success","is_error":false,'
                b'"result":"{\\"verdict\\":\\"TRUE\\"}"}',
                b"",
            )

    async def fake_subprocess(*args: str, **kwargs: Any) -> Process:
        observed["args"] = args
        observed["cwd"] = kwargs["cwd"]
        staged = Path(kwargs["cwd"]) / "pkg" / "module.py"
        observed["staged_content"] = staged.read_text()
        observed["other_files"] = sorted(
            str(path.relative_to(kwargs["cwd"]))
            for path in Path(kwargs["cwd"]).rglob("*")
            if path.is_file()
        )
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    transport = OfficialCursorCLITransport(
        str(native), seed_prompt_sources=True
    )
    prompt = (
        "<UNTRUSTED_EXACT_INPUTS>\n"
        + json.dumps(
            {
                "exact_inputs": [
                    {
                        "data": {
                            "kind": "simple_requested_sources",
                            "served": [
                                {"path": "pkg/module.py", "content": "x = 1\n"}
                            ],
                        }
                    }
                ]
            }
        )
        + "\n</UNTRUSTED_EXACT_INPUTS>"
    )
    raw, _ = await transport.complete(
        api_key="", model="auto", prompt=prompt, timeout=5
    )

    assert raw == '{"verdict":"TRUE"}'
    assert observed["cwd"] != str(tmp_path)
    assert observed["staged_content"] == "x = 1\n"
    assert observed["other_files"] == ["pkg/module.py"]
    args = observed["args"]
    assert args[args.index("--sandbox") + 1] == "enabled"
    assert args[args.index("--mode") + 1] == "ask"


def test_cli_poc_transport_rejects_path_escape_from_prompt(tmp_path: Path) -> None:
    prompt = (
        "<UNTRUSTED_EXACT_INPUTS>\n"
        + json.dumps(
            {
                "exact_inputs": [
                    {
                        "data": {
                            "kind": "simple_requested_sources",
                            "served": [
                                {"path": "../outside.py", "content": "x = 1\n"}
                            ],
                        }
                    }
                ]
            }
        )
        + "\n</UNTRUSTED_EXACT_INPUTS>"
    )
    with pytest.raises(ValueError, match="CURSOR_PROMPT_SOURCE_PATH_INVALID"):
        OfficialCursorCLITransport._stage_prompt_sources(tmp_path, prompt)
    assert not (tmp_path.parent / "outside.py").exists()


def test_cli_transport_splits_large_json_prompt_without_dropping_facts() -> None:
    prompt = '{"facts":[' + ",".join(f'"fact-{i:05d}"' for i in range(18000)) + "]}"

    parts = OfficialCursorCLITransport._prompt_args(prompt)

    assert len(parts) > 1
    assert all(len(part.encode("utf-8")) <= 96 * 1024 for part in parts)
    assert json.loads(" ".join(parts)) == json.loads(prompt)


def test_invalid_output_correction_stays_under_the_argument_limit() -> None:
    base = '"' + ("x" * (95 * 1024))
    assert OfficialCursorCLITransport._prompt_args(base) == (base,)
    oversized = base + "\n" + ("y" * (20 * 1024))
    with pytest.raises(CursorCLIPromptTooLargeError):
        OfficialCursorCLITransport._prompt_args(oversized)

    correction = _invalid_output_correction("$.citations", base)
    parts = OfficialCursorCLITransport._prompt_args(base + correction)
    assert all(len(part.encode("utf-8")) <= 96 * 1024 for part in parts)
    assert "already given above" in "".join(parts)
    assert correction.count("{") == 0


def test_cli_transport_rejects_unsplittable_oversized_prompt() -> None:
    with pytest.raises(CursorCLIPromptTooLargeError):
        OfficialCursorCLITransport._prompt_args("x" * (96 * 1024 + 1))


@pytest.mark.asyncio
async def test_cli_transport_classifies_rate_limit_without_leaking_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    native = tmp_path / "node.exe"
    native.write_bytes(b"test")
    (tmp_path / "index.js").write_bytes(b"test")

    class Process:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b"429 rate limit for account@example.test"

    async def fake_subprocess(*_args: str, **_kwargs: Any) -> Process:
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
    with pytest.raises(CursorCLIRateLimitError):
        await OfficialCursorCLITransport(str(native)).list_models("")
