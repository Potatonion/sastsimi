from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sastsimi.contracts.refs import StoredDataRef
from sastsimi.simple_runtime.artifacts import SimpleArtifactRepository
from sastsimi.simple_runtime.models import (
    CheckpointIdentity,
    SimpleStage,
    StageCheckpoint,
    StageFailure,
    StageStatus,
    input_reference_hash,
)
from sastsimi.simple_runtime.provider import SimpleLLMCallResult
from sastsimi.simple_runtime.runner import StageBlocked
from sastsimi.simple_runtime.stages import PoCCandidateStage
from sastsimi.simple_runtime.store import SimpleCheckpointStore


class _RepairClient:
    def __init__(self) -> None:
        self.prompts: list[bytes] = []

    async def call(self, **kwargs: Any) -> SimpleLLMCallResult:
        self.prompts.append(kwargs["prompt"])
        if "plan" in kwargs["output_schema"]["properties"]:
            return SimpleLLMCallResult(
                value={"plan": "Use the local entry point and /tmp fixtures."},
                prompt_digest="a" * 64,
                output_digest="b" * 64,
            )
        content = (
            "#!/bin/sh\ncookie=fixture_value\nprintf x\n"
            if len(self.prompts) == 2
            else "#!/bin/sh\nfixture_value=x\nprintf '%s' \"$fixture_value\"\n"
        )
        return SimpleLLMCallResult(
            value={"content": content},
            prompt_digest="a" * 64,
            output_digest="b" * 64,
        )


class _HostPathRepairClient(_RepairClient):
    async def call(self, **kwargs: Any) -> SimpleLLMCallResult:
        self.prompts.append(kwargs["prompt"])
        if "plan" in kwargs["output_schema"]["properties"]:
            return SimpleLLMCallResult(
                value={"plan": "Use the local entry point and /tmp fixtures."},
                prompt_digest="a" * 64,
                output_digest="b" * 64,
            )
        content = (
            "#!/bin/sh\nprintf '%s\\n' '\\\\attacker.example'\n"
            if len(self.prompts) == 2
            else (
                "#!/bin/sh\npython - <<'PY'\n"
                "print(chr(92) * 2 + 'attacker.example')\nPY\n"
            )
        )
        return SimpleLLMCallResult(
            value={"content": content},
            prompt_digest="a" * 64,
            output_digest="b" * 64,
        )


class _HomePathRepairClient(_RepairClient):
    async def call(self, **kwargs: Any) -> SimpleLLMCallResult:
        self.prompts.append(kwargs["prompt"])
        if "plan" in kwargs["output_schema"]["properties"]:
            return SimpleLLMCallResult(
                value={"plan": "Use the local entry point and /tmp fixtures."},
                prompt_digest="a" * 64,
                output_digest="b" * 64,
            )
        content = (
            "#!/bin/sh\nfor directory in /home /root /tmp; do\n"
            "  printf '%s\\n' \"$directory\"\ndone\n"
            if len(self.prompts) == 2
            else "#!/bin/sh\nfor directory in /workspace /tmp; do\n"
            "  printf '%s\\n' \"$directory\"\ndone\n"
        )
        return SimpleLLMCallResult(
            value={"content": content},
            prompt_digest="a" * 64,
            output_digest="b" * 64,
        )


class _ExternalUrlRepairClient(_RepairClient):
    async def call(self, **kwargs: Any) -> SimpleLLMCallResult:
        self.prompts.append(kwargs["prompt"])
        if "plan" in kwargs["output_schema"]["properties"]:
            return SimpleLLMCallResult(
                value={"plan": "Use the local entry point and /tmp fixtures."},
                prompt_digest="a" * 64,
                output_digest="b" * 64,
            )
        content = (
            "#!/bin/sh\nprintf '%s\\n' 'https://attacker.example/path'\n"
            if len(self.prompts) == 2
            else (
                "#!/bin/sh\npython - <<'PY'\n"
                "print('https:' + chr(47) * 2 + 'attacker.example/path')\nPY\n"
            )
        )
        return SimpleLLMCallResult(
            value={"content": content},
            prompt_digest="a" * 64,
            output_digest="b" * 64,
        )


class _SourceRecordingClient:
    def __init__(self) -> None:
        self.prompt = b""
        self.prompts: list[bytes] = []

    async def call(self, **kwargs: Any) -> SimpleLLMCallResult:
        self.prompt = kwargs["prompt"]
        self.prompts.append(self.prompt)
        value = (
            {"plan": "Use the real local entry point and isolated /tmp fixtures."}
            if "plan" in kwargs["output_schema"]["properties"]
            else {"content": "#!/bin/sh\nprintf 'reproduced\\n'\n"}
        )
        return SimpleLLMCallResult(
            value=value,
            prompt_digest="a" * 64,
            output_digest="b" * 64,
        )


def _complete_prior(
    artifacts: SimpleArtifactRepository,
    identity: CheckpointIdentity,
    *,
    include_source: bool = True,
) -> dict[SimpleStage, StageCheckpoint]:
    proposal_ref = artifacts.put_json(
        {
            "kind": "simple_hypothesis_proposal",
            "hypothesis_id": identity.hypothesis_id,
            "proposal": {
                "title": "Example hypothesis",
                "vulnerability_type": "test",
                "summary": "Exercise the real local handler",
                "code_locations": ["pkg/watch.py:1"],
                "source": "local fixture",
                "sink": "pkg/watch.py",
                "rationale": "The handler needs a control and case",
            },
        }
    )
    source_ref = artifacts.put_json(
        {
            "kind": "simple_requested_sources",
            "served": [{"path": "pkg/watch.py", "content": "def handler(): pass\n"}],
            "refused": [],
        }
    )
    source_refs = [source_ref.model_dump(mode="json")] if include_source else []
    pro_ref = artifacts.put_json(
        {
            "kind": "simple_pro_evidence",
            "result": {
                "claims": ["A case may reach the handler"],
                "limitations": ["A control is required"],
            },
            "source_refs": source_refs,
        }
    )
    con_ref = artifacts.put_json(
        {
            "kind": "simple_con_evidence",
            "result": {
                "claims": ["Validation may reject the case"],
                "limitations": ["Check the real handler"],
            },
            "source_refs": source_refs,
        }
    )
    verification_ref = artifacts.put_json(
        {
            "kind": "simple_initial_verification",
            "result": {
                "initial_assessment": "HOLD",
                "rationale": "Needs a local reproduction",
                "reproduction_goal": "Compare control and case in the handler",
                "environment_requirements": ["local Python"],
                "limitations": [],
            },
        }
    )
    return {
        SimpleStage.PRO_CON_DONE: StageCheckpoint(
            identity=identity,
            stage=SimpleStage.PRO_CON_DONE,
            status=StageStatus.SUCCEEDED,
            input_refs=(proposal_ref,),
            input_hash=input_reference_hash((proposal_ref,)),
            output_refs=(pro_ref, con_ref),
        ),
        SimpleStage.VERIFICATION_INITIAL_DONE: StageCheckpoint(
            identity=identity,
            stage=SimpleStage.VERIFICATION_INITIAL_DONE,
            status=StageStatus.SUCCEEDED,
            input_refs=(pro_ref, con_ref),
            input_hash=input_reference_hash((pro_ref, con_ref)),
            output_refs=(verification_ref,),
        ),
    }


@pytest.mark.asyncio
async def test_poc_model_refusal_becomes_inconclusive_stage_failure(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    refusal_ref = artifacts.put_json(
        {"kind": "cursor_raw_output", "text": "I can't provide a PoC script."}
    )

    class RefusingClient:
        async def call(self, **kwargs: Any) -> StageFailure:
            return StageFailure(
                code="CURSOR_REFUSED",
                retryable=False,
                safe_message="Model declined PoC generation",
                evidence_refs=(refusal_ref,),
            )

    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
    )
    stage = PoCCandidateStage(client=RefusingClient(), artifacts=artifacts)

    with pytest.raises(StageBlocked) as blocked:
        await stage(checkpoint, _complete_prior(artifacts, identity))

    assert blocked.value.failure.code == "POC_MODEL_REFUSED"
    assert blocked.value.failure.evidence_refs == (refusal_ref,)


@pytest.mark.asyncio
async def test_poc_without_served_source_blocks_before_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    source_ref = artifacts.put_json(
        {
            "kind": "simple_requested_sources",
            "served": [],
            "refused": [
                {"path": "pkg/large.py", "reason": "TOTAL_BUDGET_EXHAUSTED"}
            ],
        }
    )

    class UnexpectedClient:
        async def call(self, **kwargs: Any) -> SimpleLLMCallResult:
            raise AssertionError("model should not be called without source")

    stage = PoCCandidateStage(client=UnexpectedClient(), artifacts=artifacts)
    monkeypatch.setattr(
        stage, "_requested_source_ref", lambda *args, **kwargs: source_ref
    )
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
    )

    with pytest.raises(StageBlocked) as blocked:
        await stage(checkpoint, {})

    assert blocked.value.failure.code == "POC_SOURCE_CONTEXT_UNAVAILABLE"
    assert blocked.value.failure.evidence_refs == (source_ref,)


@pytest.mark.asyncio
async def test_poc_plan_is_reused_on_candidate_restart(tmp_path: Path) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    store = SimpleCheckpointStore(artifacts.paths.database)
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-1",
    )
    client = _SourceRecordingClient()
    calls: list[str] = []
    original_call = client.call

    async def recording_call(**kwargs: Any) -> SimpleLLMCallResult:
        calls.append(next(iter(kwargs["output_schema"]["properties"])))
        return await original_call(**kwargs)

    client.call = recording_call  # type: ignore[method-assign]
    stage = PoCCandidateStage(client=client, artifacts=artifacts, store=store)
    prior = _complete_prior(artifacts, identity)
    await stage(checkpoint, prior)
    await stage(checkpoint, prior)
    assert calls == ["plan", "content", "content"]


@pytest.mark.asyncio
async def test_execution_recovery_reuses_the_original_plan(tmp_path: Path) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    store = SimpleCheckpointStore(artifacts.paths.database)
    client = _SourceRecordingClient()
    calls: list[str] = []
    original_call = client.call

    async def recording_call(**kwargs: Any) -> SimpleLLMCallResult:
        calls.append(next(iter(kwargs["output_schema"]["properties"])))
        return await original_call(**kwargs)

    client.call = recording_call  # type: ignore[method-assign]
    stage = PoCCandidateStage(client=client, artifacts=artifacts, store=store)
    prior = _complete_prior(artifacts, identity)
    first = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-1",
    )
    await stage(first, prior)

    previous_content_ref = artifacts.put_bytes(
        b"#!/bin/sh\necho previous-script-marker\n", "text/x-shellscript"
    )
    candidate_ref = artifacts.put_json(
        {
            "kind": "simple_poc_candidate",
            "content_ref": previous_content_ref.model_dump(mode="json"),
            "content_digest": previous_content_ref.content_hash,
        }
    )
    stderr_ref = artifacts.put_bytes(
        b"recovery-failure-marker\n", "text/plain"
    )
    execution_ref = artifacts.put_json(
        {
            "kind": "simple_poc_execution",
            "exit_code": 2,
            "timed_out": False,
            "stderr_ref": stderr_ref.model_dump(mode="json"),
        }
    )
    repair_inputs = (candidate_ref, execution_ref)
    repaired = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=repair_inputs,
        input_hash=input_reference_hash(repair_inputs),
        attempt_id="attempt-2",
    )
    await stage(repaired, prior)

    assert calls == ["plan", "content", "content"]
    assert b"recovery-failure-marker" in client.prompt
    assert b"previous-script-marker" in client.prompt
    assert b"real local entry point" in client.prompt


@pytest.mark.asyncio
async def test_poc_plan_small_length_overrun_does_not_consume_recovery(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
    )

    class SlightlyLongPlanClient(_SourceRecordingClient):
        async def call(self, **kwargs: Any) -> SimpleLLMCallResult:
            if "plan" in kwargs["output_schema"]["properties"]:
                return SimpleLLMCallResult(
                    value={"plan": "P" * 1212},
                    prompt_digest="a" * 64,
                    output_digest="b" * 64,
                )
            return await super().call(**kwargs)

    stage = PoCCandidateStage(
        client=SlightlyLongPlanClient(), artifacts=artifacts
    )
    result = await stage(checkpoint, _complete_prior(artifacts, identity))

    assert result.output_refs


@pytest.mark.asyncio
async def test_poc_generation_missing_required_evidence_never_calls_script_model(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
    )
    client = _SourceRecordingClient()

    with pytest.raises(StageBlocked) as blocked:
        await PoCCandidateStage(client=client, artifacts=artifacts)(checkpoint, {})

    assert blocked.value.failure.code == "POC_GENERATION_CONTEXT_INCOMPLETE"
    assert len(client.prompts) == 0


@pytest.mark.asyncio
async def test_poc_candidate_receives_requested_tracked_source_with_provenance(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg" / "watch.py").write_text(
        "class Watch:\n    def __init__(self, **kwargs):\n"
        "        self.datastore = kwargs['datastore']\n",
        encoding="utf-8",
    )
    (workspace / "pkg" / "oversize.py").write_text("x" * 140_000, encoding="utf-8")
    (workspace / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (workspace / "private.txt").write_text("private-marker", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "add",
            "pkg/watch.py",
            "pkg/oversize.py",
            "Dockerfile",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "user.name=SASTSIMI Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "pinned fixture",
        ],
        check=True,
    )
    commit = (
        subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"])
        .decode("ascii")
        .strip()
    )
    (workspace / "pkg" / "watch.py").write_text(
        "class Watch:\n    # dirty-workspace-marker\n",
        encoding="utf-8",
    )
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id=commit,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path / "data", identity)
    manifest_ref = artifacts.put_json(
        {
            "kind": "simple_tracked_sources",
            "paths": ["pkg/watch.py", "pkg/oversize.py"],
        }
    )
    bundle_ref = artifacts.put_json(
        {
            "kind": "simple_static_fact_bundle",
            "source_manifest_ref": manifest_ref.model_dump(mode="json"),
            "poc_source_manifest_ref": artifacts.put_json(
                {
                    "kind": "simple_tracked_sources",
                    "paths": ["pkg/watch.py", "pkg/oversize.py", "Dockerfile"],
                }
            ).model_dump(mode="json"),
        }
    )
    oversized_prior_ref = artifacts.put_json(
        {"kind": "oversized_prior_context", "content": "x" * 300_000}
    )
    irrelevant_prior_ref = artifacts.put_json(
        {"kind": "unrelated_prior_context", "marker": "irrelevant-prior-marker"}
    )
    proposal_ref = artifacts.put_json(
        {
            "kind": "simple_hypothesis_proposal",
            "hypothesis_id": identity.hypothesis_id,
            "proposal": {
                "title": "Watch handler",
                "summary": "Exercise the pinned handler",
                "rationale": "The handler may accept unsafe input",
                "code_locations": ["pkg/watch.py:1"],
            },
        }
    )
    pro_ref = artifacts.put_json(
        {
            "kind": "simple_pro_evidence",
            "result": {
                "claims": ["The handler accepts a fixture"],
                "limitations": [],
                "requested_paths": [
                    "pkg/watch.py",
                    "Dockerfile",
                    "pkg/oversize.py",
                    "private.txt",
                    "../outside",
                ]
            },
        }
    )
    con_ref = artifacts.put_json(
        {
            "kind": "simple_con_evidence",
            "result": {
                "claims": ["The handler may reject the fixture"],
                "limitations": [],
                "requested_paths": [],
            },
        }
    )
    verification_ref = artifacts.put_json(
        {
            "kind": "simple_initial_verification",
            "result": {
                "reproduction_goal": "core-verification-marker",
                "environment_requirements": ["local Python"],
            },
        }
    )
    prior = {
        SimpleStage.HYPOTHESIS_DONE: StageCheckpoint(
            identity=identity,
            stage=SimpleStage.HYPOTHESIS_DONE,
            status=StageStatus.SUCCEEDED,
            input_refs=(bundle_ref,),
            input_hash=input_reference_hash((bundle_ref,)),
            output_refs=(irrelevant_prior_ref, oversized_prior_ref),
        ),
        SimpleStage.PRO_CON_DONE: StageCheckpoint(
            identity=identity,
            stage=SimpleStage.PRO_CON_DONE,
            status=StageStatus.SUCCEEDED,
            input_refs=(proposal_ref, bundle_ref),
            input_hash=input_reference_hash((proposal_ref, bundle_ref)),
            output_refs=(pro_ref, con_ref),
        ),
        SimpleStage.VERIFICATION_INITIAL_DONE: StageCheckpoint(
            identity=identity,
            stage=SimpleStage.VERIFICATION_INITIAL_DONE,
            status=StageStatus.SUCCEEDED,
            input_refs=(bundle_ref,),
            input_hash=input_reference_hash((bundle_ref,)),
            output_refs=(verification_ref,),
        ),
    }
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(bundle_ref,),
        input_hash=input_reference_hash((bundle_ref,)),
        attempt_id="attempt-1",
    )
    client = _SourceRecordingClient()
    stage = PoCCandidateStage(
        client=client,
        artifacts=artifacts,
        workspace_path=workspace,
        static_bundle_ref=bundle_ref,
    )

    result = await stage(checkpoint, prior)

    assert b"def __init__(self, **kwargs)" in client.prompt
    assert b"FROM python:3.12-slim" in client.prompt
    assert b"dirty-workspace-marker" not in client.prompt
    assert b"core-verification-marker" in client.prompt
    assert b"simple_pro_evidence" in client.prompt
    assert b"irrelevant-prior-marker" not in client.prompts[0]
    assert b"irrelevant-prior-marker" not in client.prompt
    assert b"simple_poc_generation_input" in client.prompt
    assert b"operator has authorized testing" in client.prompts[0]
    assert b"operator-controlled" in client.prompt
    assert b"No live service, real user" in client.prompt
    assert b"synthetic test identities and fixture records" in client.prompt
    assert b"status codes," in client.prompt
    assert b"Do not print or return record contents" in client.prompt
    assert b"Do not invent a positive" in client.prompt
    assert client.prompt.split(b"<UNTRUSTED_EXACT_INPUTS>", 1)[1] == (
        client.prompts[0].split(b"<UNTRUSTED_EXACT_INPUTS>", 1)[1]
    )
    assert b"private-marker" not in client.prompt
    assert b"Existing files under /workspace are immutable" in client.prompt
    assert b"Target at most 12 KiB" in client.prompt
    assert b"runtime storage" in client.prompt
    candidate = json.loads(artifacts.read(result.output_refs[0]))
    generation_input = json.loads(
        artifacts.read(StoredDataRef.model_validate(candidate["generation_input_ref"]))
    )
    assert generation_input["hypothesis"]["summary"] == "Exercise the pinned handler"
    assert generation_input["initial_verification"]["reproduction_goal"] == (
        "core-verification-marker"
    )
    assert generation_input["selected_source_hashes"]
    assert "source_refs" not in generation_input
    source_records = [
        json.loads(artifacts.read(StoredDataRef.model_validate(raw_ref)))
        for raw_ref in candidate["source_refs"]
    ]
    retrieved = next(
        record
        for record in source_records
        if record.get("kind") == "simple_requested_sources"
    )
    source_ref = next(
        StoredDataRef.model_validate(raw_ref)
        for raw_ref in candidate["source_refs"]
        if json.loads(artifacts.read(StoredDataRef.model_validate(raw_ref))).get("kind")
        == "simple_requested_sources"
    )
    assert source_ref in result.output_refs[2:]
    assert [item["path"] for item in retrieved["served"]] == [
        "pkg/watch.py",
        "Dockerfile",
        "pkg/oversize.py",
    ]
    oversize = next(
        item for item in retrieved["served"] if item["path"] == "pkg/oversize.py"
    )
    assert oversize["excerpt"] is True
    assert len(oversize["content"]) <= 8_000
    assert {item["reason"] for item in retrieved["refused"]} == {
        "NOT_TRACKED",
        "PATH_OUTSIDE_REPOSITORY",
    }


@pytest.mark.asyncio
async def test_poc_generation_restores_entry_source_and_outlines_large_json(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    prior = _complete_prior(artifacts, identity)
    entry_source_ref = artifacts.put_json(
        {
            "kind": "simple_requested_sources",
            "served": [
                {
                    "path": "apps/_migrations/0007_seed_reference_data.py",
                    "content": "def seed_reference_data(apps, schema_editor): pass\n",
                }
            ],
            "refused": [],
        }
    )
    pro_record = json.loads(
        artifacts.read(prior[SimpleStage.PRO_CON_DONE].output_refs[0])
    )
    pro_record["source_refs"] = [entry_source_ref.model_dump(mode="json")]
    pro_ref = artifacts.put_json(pro_record)
    con_ref = prior[SimpleStage.PRO_CON_DONE].output_refs[1]
    prior[SimpleStage.PRO_CON_DONE] = prior[SimpleStage.PRO_CON_DONE].model_copy(
        update={"output_refs": (pro_ref, con_ref)}
    )
    large_marker = "large-fixture-marker-" + "x" * 50_000
    requested_source_ref = artifacts.put_json(
        {
            "kind": "simple_requested_sources",
            "served": [
                {
                    "path": "apps/_migrations/seed_data/reference_data.json",
                    "content": json.dumps(
                        {
                            "integrations": [{"code": "sample", "name": "sample"}],
                            "padding": large_marker,
                        }
                    ),
                },
                {"path": "apps/models.py", "content": "class Model: pass\n"},
            ],
            "refused": [],
        }
    )
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
    )
    client = _SourceRecordingClient()
    stage = PoCCandidateStage(client=client, artifacts=artifacts)
    stage._requested_source_ref = lambda *args, **kwargs: requested_source_ref

    result = await stage(checkpoint, prior)

    assert result.output_refs
    assert len(client.prompts) == 2
    assert b"def seed_reference_data" in client.prompt
    assert b"class Model: pass" in client.prompt
    assert large_marker.encode() not in client.prompt
    assert b"large_source_outlines" in client.prompt
    assert b"reference_data.json" in client.prompt
    assert len(client.prompt) < 20_000
    candidate = json.loads(artifacts.read(result.output_refs[0]))
    assert requested_source_ref.model_dump(mode="json") in candidate["source_refs"]
    assert entry_source_ref.model_dump(mode="json") in candidate["source_refs"]


@pytest.mark.asyncio
async def test_gate_feedback_precedes_bulk_context_and_is_forwarded(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    bulk_ref = artifacts.put_json({"kind": "bulk", "content": "x" * 300_000})
    feedback_ref = artifacts.put_json(
        {
            "kind": "simple_technical_gate",
            "result": {
                "status": "REVISE",
                "revision_requests": ["production-route-marker"],
            },
        }
    )
    previous_content_ref = artifacts.put_bytes(
        b"#!/bin/sh\necho previous-script-marker\n", "text/x-shellscript"
    )
    previous_candidate_ref = artifacts.put_json(
        {
            "kind": "simple_poc_candidate",
            "content_ref": previous_content_ref.model_dump(mode="json"),
            "content_digest": previous_content_ref.content_hash,
        }
    )
    previous_stdout_ref = artifacts.put_bytes(b"control case\n", "text/plain")
    previous_stderr_ref = artifacts.put_bytes(b"previous-error-marker\n", "text/plain")
    previous_execution_ref = artifacts.put_json(
        {
            "kind": "simple_poc_execution",
            "exit_code": 2,
            "timed_out": False,
            "stdout_ref": previous_stdout_ref.model_dump(mode="json"),
            "stderr_ref": previous_stderr_ref.model_dump(mode="json"),
        }
    )
    prior = _complete_prior(artifacts, identity)
    prior[SimpleStage.HYPOTHESIS_DONE] = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.HYPOTHESIS_DONE,
        status=StageStatus.SUCCEEDED,
        input_refs=(),
        input_hash=input_reference_hash(()),
        output_refs=(bulk_ref,),
    )
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(
            feedback_ref,
            previous_candidate_ref,
            previous_execution_ref,
        ),
        input_hash=input_reference_hash(
            (feedback_ref, previous_candidate_ref, previous_execution_ref)
        ),
        attempt_id="revised-poc",
    ).model_copy(update={"gate_revision_count": 1})
    client = _SourceRecordingClient()

    result = await PoCCandidateStage(client=client, artifacts=artifacts)(
        checkpoint, prior
    )

    assert b"production-route-marker" in client.prompt
    assert b"previous-script-marker" in client.prompt
    assert b"previous-error-marker" in client.prompt
    assert b"\"kind\":\"bulk\"" not in client.prompt
    assert feedback_ref in result.output_refs[2:]


@pytest.mark.asyncio
async def test_execution_failure_feedback_reaches_next_candidate_without_gate(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    previous_content_ref = artifacts.put_bytes(
        b"#!/bin/sh\necho previous-script-marker\n", "text/x-shellscript"
    )
    candidate_ref = artifacts.put_json(
        {
            "kind": "simple_poc_candidate",
            "content_ref": previous_content_ref.model_dump(mode="json"),
            "content_digest": previous_content_ref.content_hash,
        }
    )
    stderr_ref = artifacts.put_bytes(
        b"sample-placeholder-marker stopped Django startup\n", "text/plain"
    )
    execution_ref = artifacts.put_json(
        {
            "kind": "simple_poc_execution",
            "exit_code": 2,
            "timed_out": False,
            "stderr_ref": stderr_ref.model_dump(mode="json"),
        }
    )
    inputs = (candidate_ref, execution_ref)
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=inputs,
        input_hash=input_reference_hash(inputs),
        attempt_id="repair-after-execution",
    )
    client = _SourceRecordingClient()

    await PoCCandidateStage(client=client, artifacts=artifacts)(
        checkpoint, _complete_prior(artifacts, identity)
    )

    assert b"previous-script-marker" in client.prompt
    assert b"sample-placeholder-marker" in client.prompt
    assert b'"exit_code":2' in client.prompt
    assert b"Do not invoke `docker`" in client.prompt
    assert b'"pro_con":' not in client.prompt
    assert b'"initial_assessment":' not in client.prompt
    assert b"dependency closure" in client.prompts[0]
    assert b"Do not copy\na numeric range." in client.prompt
    assert b"extra writable field" in client.prompt


@pytest.mark.asyncio
async def test_missing_migration_parent_is_named_in_repair_context(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    previous_content_ref = artifacts.put_bytes(
        b"#!/bin/sh\necho previous-script-marker\n", "text/x-shellscript"
    )
    candidate_ref = artifacts.put_json(
        {
            "kind": "simple_poc_candidate",
            "content_ref": previous_content_ref.model_dump(mode="json"),
            "content_digest": previous_content_ref.content_hash,
        }
    )
    stderr_ref = artifacts.put_bytes(
        b"NodeNotFoundError: Migration demo_app.0002_example dependencies "
        b"reference nonexistent parent node "
        b"('third_party_app', '0018_example')\n",
        "text/plain",
    )
    execution_ref = artifacts.put_json(
        {
            "kind": "simple_poc_execution",
            "exit_code": 2,
            "timed_out": False,
            "stderr_ref": stderr_ref.model_dump(mode="json"),
        }
    )
    inputs = (candidate_ref, execution_ref)
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=inputs,
        input_hash=input_reference_hash(inputs),
        attempt_id="repair-migration-graph",
    )
    client = _SourceRecordingClient()

    await PoCCandidateStage(client=client, artifacts=artifacts)(
        checkpoint, _complete_prior(artifacts, identity)
    )

    assert b'"migration_graph"' not in client.prompts[0]
    assert b'"migration_graph"' in client.prompt
    assert b"demo_app.0002_example" in client.prompt
    assert b"third_party_app" in client.prompt
    assert b"0018_example" in client.prompt
    assert b"packaged migration" in client.prompt


@pytest.mark.asyncio
async def test_swappable_dependency_literal_eval_is_named_in_repair_context(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    previous_content_ref = artifacts.put_bytes(
        b"#!/bin/sh\necho previous-script-marker\n", "text/x-shellscript"
    )
    candidate_ref = artifacts.put_json(
        {
            "kind": "simple_poc_candidate",
            "content_ref": previous_content_ref.model_dump(mode="json"),
            "content_digest": previous_content_ref.content_hash,
        }
    )
    stderr_ref = artifacts.put_bytes(
        b"ValueError: malformed node or string: Call("
        b"func=Attribute(attr='swappable_dependency'))\n"
        b"ast.literal_eval\n",
        "text/plain",
    )
    execution_ref = artifacts.put_json(
        {
            "kind": "simple_poc_execution",
            "exit_code": 2,
            "timed_out": False,
            "stderr_ref": stderr_ref.model_dump(mode="json"),
        }
    )
    inputs = (candidate_ref, execution_ref)
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=inputs,
        input_hash=input_reference_hash(inputs),
        attempt_id="repair-swappable-dependency",
    )
    client = _SourceRecordingClient()

    await PoCCandidateStage(client=client, artifacts=artifacts)(
        checkpoint, _complete_prior(artifacts, identity)
    )

    assert b"swappable_dependency(settings.AUTH_USER_MODEL)" in client.prompt
    assert b"concrete (app, name)" in client.prompt


@pytest.mark.asyncio
async def test_completed_control_does_not_remain_a_missing_field_search(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    previous_content_ref = artifacts.put_bytes(
        b"#!/bin/sh\necho previous-script-marker\n", "text/x-shellscript"
    )
    candidate_ref = artifacts.put_json(
        {
            "kind": "simple_poc_candidate",
            "content_ref": previous_content_ref.model_dump(mode="json"),
            "content_digest": previous_content_ref.content_hash,
        }
    )
    stdout_ref = artifacts.put_bytes(b"control_catalog=complete\n", "text/plain")
    stderr_ref = artifacts.put_bytes(
        b"MissingPrerequisite\nwritable non-lookup field unavailable\n",
        "text/plain",
    )
    execution_ref = artifacts.put_json(
        {
            "kind": "simple_poc_execution",
            "exit_code": 2,
            "timed_out": False,
            "stdout_ref": stdout_ref.model_dump(mode="json"),
            "stderr_ref": stderr_ref.model_dump(mode="json"),
        }
    )
    inputs = (candidate_ref, execution_ref)
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=inputs,
        input_hash=input_reference_hash(inputs),
        attempt_id="repair-fixture-case",
    )
    client = _SourceRecordingClient()

    await PoCCandidateStage(client=client, artifacts=artifacts)(
        checkpoint, _complete_prior(artifacts, identity)
    )

    assert b'"fixture_case"' in client.prompt
    assert b"absent from the model" in client.prompt


@pytest.mark.asyncio
async def test_sensitive_candidate_repair_explains_secret_shaped_names(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-1",
    )
    client = _RepairClient()
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    stage = PoCCandidateStage(
        client=client,
        artifacts=artifacts,
    )

    result = await stage(checkpoint, _complete_prior(artifacts, identity))

    assert result.output_refs
    assert len(client.prompts) == 3
    assert b"concise error type and traceback to stderr" in client.prompts[1]
    assert b"secret-shaped identifiers" in client.prompts[2]
    assert b"cookie, session, token" in client.prompts[2]


@pytest.mark.asyncio
async def test_sensitive_candidate_repair_explains_host_home_paths(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-1",
    )
    client = _HomePathRepairClient()
    artifacts = SimpleArtifactRepository(tmp_path, identity)

    result = await PoCCandidateStage(client=client, artifacts=artifacts)(
        checkpoint, _complete_prior(artifacts, identity)
    )

    assert result.output_refs
    assert len(client.prompts) == 3
    assert b"Do not mention or probe `/home` or `/root`" in client.prompts[1]
    assert b"scratch names like $BASE/home" in client.prompts[2]


@pytest.mark.asyncio
async def test_host_path_repair_explains_how_to_build_backslash_url_fixtures(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-1",
    )
    client = _HostPathRepairClient()
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    stage = PoCCandidateStage(
        client=client,
        artifacts=artifacts,
    )

    result = await stage(checkpoint, _complete_prior(artifacts, identity))

    assert result.output_refs
    assert len(client.prompts) == 3
    assert b"external-looking and backslash-confused URL fixtures" in client.prompts[1]
    assert b"construct them at runtime" in client.prompts[1]
    assert b"construct backslash-confused URL fixtures at runtime" in client.prompts[2]
    assert b"chr(92)" in client.prompts[2]


@pytest.mark.asyncio
async def test_external_url_repair_keeps_network_off_and_builds_fixture_at_runtime(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.RUNNING,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-1",
    )
    client = _ExternalUrlRepairClient()
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    stage = PoCCandidateStage(
        client=client,
        artifacts=artifacts,
    )

    result = await stage(checkpoint, _complete_prior(artifacts, identity))

    assert result.output_refs
    assert len(client.prompts) == 3
    assert b"external-looking and backslash-confused URL fixtures" in client.prompts[1]
    assert b"construct them at runtime" in client.prompts[1]
    assert b"must not make an external network request" in client.prompts[2]
    assert b"construct the URL fixture at runtime" in client.prompts[2]
