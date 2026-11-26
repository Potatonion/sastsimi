from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

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
from sastsimi.simple_runtime.stages import ProConStage
from sastsimi.simple_runtime.store import SimpleCheckpointStore


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes]] = []
        self.fail_con_once = True

    async def call(self, **kwargs: Any) -> SimpleLLMCallResult | StageFailure:
        agent = kwargs["agent_name"]
        self.calls.append((agent, kwargs["prompt"]))
        if agent == "con_evidence" and self.fail_con_once:
            self.fail_con_once = False
            return StageFailure(
                code="CURSOR_TIMED_OUT",
                retryable=True,
                safe_message="Test timeout",
            )
        return SimpleLLMCallResult(
            value={
                "claims": [],
                "evidence_refs": [],
                "limitations": [],
                "requested_paths": [],
            },
            prompt_digest="a" * 64,
            output_digest="b" * 64,
        )


@pytest.mark.asyncio
async def test_pro_con_uses_selected_bounded_pinned_evidence_and_reuses_pro(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "aso").mkdir(parents=True)
    source = workspace / "aso" / "back_link.py"
    source.write_text("def back_link():\n    return 'pinned-source-marker'\n")
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "aso/back_link.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True
    ).strip()
    source.write_text("def back_link():\n    return 'dirty-source-marker'\n")
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id=commit,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path / "data", identity)
    store = SimpleCheckpointStore(artifacts.paths.database)
    manifest_ref = artifacts.put_json(
        {
            "kind": "simple_tracked_sources",
            "paths": ["aso/back_link.py"],
        }
    )
    bundle_ref = artifacts.put_json(
        {
            "kind": "simple_static_fact_bundle",
            "ast_summary": {
                "facts": [
                    {
                        "path": "aso/other.py",
                        "line": i,
                        "name": "unrelated-static-marker",
                    }
                    for i in range(7000)
                ]
                + [{"path": "aso/back_link.py", "line": 2, "name": "back_link"}],
            },
            "opengrep_findings": [
                {"path": "aso/back_link.py", "line": 2, "snippet": "relevant-finding"}
            ],
            "source_manifest_ref": manifest_ref.model_dump(mode="json"),
        }
    )
    hypothesis_ref = artifacts.put_json(
        {
            "kind": "simple_hypothesis_proposal",
            "proposal": {
                "code_locations": ["aso/back_link.py:2"],
                "source": "aso/back_link.py input",
                "sink": "aso/back_link.py return",
                "rationale": "selected-hypothesis-marker",
            },
        }
    )
    unrelated_ref = artifacts.put_json(
        {
            "kind": "simple_hypothesis_proposal",
            "proposal": {"rationale": "other-hypothesis-marker"},
        }
    )
    checkpoint = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.PRO_CON_DONE,
        status=StageStatus.RUNNING,
        input_refs=(hypothesis_ref, bundle_ref),
        input_hash=input_reference_hash((hypothesis_ref, bundle_ref)),
        attempt_id="attempt-1",
    )
    prior = {
        SimpleStage.HYPOTHESIS_DONE: StageCheckpoint(
            identity=identity,
            stage=SimpleStage.HYPOTHESIS_DONE,
            status=StageStatus.SUCCEEDED,
            input_refs=(),
            input_hash=input_reference_hash(()),
            output_refs=(hypothesis_ref, unrelated_ref),
        )
    }
    client = _RecordingClient()
    stage = ProConStage(
        client,
        artifacts,
        store=store,
        workspace_path=workspace,
    )
    with pytest.raises(StageBlocked) as error:
        await stage(checkpoint, prior)
    assert error.value.failure.code == "CURSOR_TIMED_OUT"
    result = await stage(checkpoint, prior)
    assert len(result.output_refs) == 2
    assert [agent for agent, _ in client.calls] == [
        "pro_evidence",
        "con_evidence",
        "con_evidence",
    ]
    for _, prompt in client.calls:
        assert len(prompt) < 60_000
        assert b"selected-hypothesis-marker" in prompt
        assert b"pinned-source-marker" in prompt
        assert b"relevant-finding" in prompt
        assert b"other-hypothesis-marker" not in prompt
        assert b"unrelated-static-marker" not in prompt
        assert b"dirty-source-marker" not in prompt
    pro = json.loads(artifacts.read(result.output_refs[0]))
    assert pro["kind"] == "simple_pro_evidence"
