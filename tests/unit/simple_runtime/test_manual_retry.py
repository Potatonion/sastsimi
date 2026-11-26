from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from sastsimi.composition.simple_runtime_composition import (
    PublicSimpleRuntimeApplication,
)
from sastsimi.simple_runtime.artifacts import SimpleArtifactRepository
from sastsimi.simple_runtime.models import (
    MAX_RECOVERY_ATTEMPTS,
    STAGE_VERSION,
    CheckpointIdentity,
    SimpleAnalysisRun,
    SimpleStage,
    StageCheckpoint,
    StageStatus,
    input_reference_hash,
)
from sastsimi.simple_runtime.store import SimpleCheckpointStore


def _exhausted(identity: CheckpointIdentity) -> StageCheckpoint:
    return StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.BLOCKED,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-3",
        attempt_number=MAX_RECOVERY_ATTEMPTS,
        error_code="RECOVERY_EXHAUSTED",
        retryable=False,
    )


def _retryable_blocked(identity: CheckpointIdentity) -> StageCheckpoint:
    return StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.BLOCKED,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-2",
        attempt_number=2,
        error_code="CURSOR_TIMED_OUT",
        retryable=True,
    )


def test_reopen_exhausted_grants_exactly_one_attempt_and_drops_downstream(
    tmp_path: Path,
) -> None:
    store = SimpleCheckpointStore(tmp_path / "sastsimi.sqlite3")
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    exhausted = _exhausted(identity)
    store.save_checkpoint(exhausted)
    store.save_checkpoint(
        StageCheckpoint(
            identity=identity,
            stage=SimpleStage.POC_EXECUTION_DONE,
            status=StageStatus.SUCCEEDED,
            input_refs=(),
            input_hash=input_reference_hash(()),
        )
    )

    pending = store.reopen_for_manual_retry(exhausted)

    assert pending.status is StageStatus.PENDING
    assert pending.attempt_number == MAX_RECOVERY_ATTEMPTS - 1
    assert pending.error_code is None
    assert store.get(identity, SimpleStage.POC_EXECUTION_DONE) is None
    with pytest.raises(ValueError, match="MANUAL_RETRY_CHECKPOINT_STALE"):
        store.reopen_for_manual_retry(exhausted)


def test_reopen_retryable_blocked_bypasses_recovery_stop(
    tmp_path: Path,
) -> None:
    store = SimpleCheckpointStore(tmp_path / "sastsimi.sqlite3")
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    blocked = _retryable_blocked(identity)
    store.save_checkpoint(blocked)

    pending = store.reopen_for_manual_retry(blocked)

    assert pending.status is StageStatus.PENDING
    assert pending.attempt_number == 2
    assert pending.error_code is None
    assert pending.retryable is False


def test_reopen_failed_cursor_output_grants_one_manual_attempt(
    tmp_path: Path,
) -> None:
    store = SimpleCheckpointStore(tmp_path / "sastsimi.sqlite3")
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id=None,
    )
    failed = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.HYPOTHESIS_DONE,
        status=StageStatus.FAILED,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-3",
        attempt_number=MAX_RECOVERY_ATTEMPTS,
        error_code="CURSOR_INVALID_OUTPUT",
        retryable=False,
    )
    store.save_checkpoint(failed)

    pending = store.reopen_for_manual_retry(failed)

    assert pending.status is StageStatus.PENDING
    assert pending.attempt_number == MAX_RECOVERY_ATTEMPTS - 1
    assert pending.error_code is None


def test_reopen_elapsed_budget_failure_grants_one_manual_attempt(
    tmp_path: Path,
) -> None:
    store = SimpleCheckpointStore(tmp_path / "sastsimi.sqlite3")
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    failed = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.VERIFICATION_INITIAL_DONE,
        status=StageStatus.FAILED,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_id="attempt-2",
        attempt_number=2,
        error_code="LLM_ELAPSED_BUDGET_EXHAUSTED",
        retryable=False,
    )
    store.save_checkpoint(failed)

    pending = store.reopen_for_manual_retry(failed)

    assert pending.status is StageStatus.PENDING
    assert pending.attempt_number == 2
    assert pending.error_code is None


@pytest.mark.parametrize(
    ("stage", "status", "error_code"),
    [
        (SimpleStage.POC_CANDIDATE_DONE, StageStatus.BLOCKED, "POC_MODEL_REFUSED"),
        (
            SimpleStage.POC_CANDIDATE_DONE,
            StageStatus.FAILED,
            "CURSOR_MODEL_SAFETY_BLOCK",
        ),
        (SimpleStage.REPORT_DONE, StageStatus.FAILED, "CURSOR_PROMPT_TOO_LARGE"),
    ],
)
def test_terminal_retry_requires_explicit_opt_in(
    tmp_path: Path,
    stage: SimpleStage,
    status: StageStatus,
    error_code: str,
) -> None:
    store = SimpleCheckpointStore(tmp_path / "sastsimi.sqlite3")
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    failed = StageCheckpoint(
        identity=identity,
        stage=stage,
        status=status,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_number=1,
        error_code=error_code,
        retryable=False,
    )
    store.save_checkpoint(failed)

    with pytest.raises(ValueError, match="MANUAL_RETRY_REQUIRES"):
        store.reopen_for_manual_retry(failed)
    pending = store.reopen_for_manual_retry(failed, allow_terminal_retry=True)

    assert pending.status is StageStatus.PENDING
    assert pending.attempt_number == 1
    assert pending.error_code is None


def test_terminal_safety_retry_grants_one_more_attempt(tmp_path: Path) -> None:
    store = SimpleCheckpointStore(tmp_path / "sastsimi.sqlite3")
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    failed = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.FAILED,
        input_refs=(),
        input_hash=input_reference_hash(()),
        attempt_number=5,
        error_code="CURSOR_MODEL_SAFETY_BLOCK",
        retryable=False,
    )
    store.save_checkpoint(failed)

    pending = store.reopen_for_manual_retry(failed, allow_terminal_retry=True)

    assert pending.status is StageStatus.PENDING
    assert pending.attempt_number == MAX_RECOVERY_ATTEMPTS - 1
    assert pending.error_code is None


def test_terminal_retry_does_not_reopen_other_safety_failures(tmp_path: Path) -> None:
    store = SimpleCheckpointStore(tmp_path / "sastsimi.sqlite3")
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    blocked = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        status=StageStatus.BLOCKED,
        input_refs=(),
        input_hash=input_reference_hash(()),
        error_code="POC_SENSITIVE_CONTENT",
        retryable=False,
    )
    store.save_checkpoint(blocked)

    with pytest.raises(ValueError, match="MANUAL_RETRY_REQUIRES"):
        store.reopen_for_manual_retry(blocked, allow_terminal_retry=True)
    assert store.get(identity, SimpleStage.POC_CANDIDATE_DONE) == blocked


def test_public_terminal_retry_selects_only_named_hypotheses(tmp_path: Path) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    hypothesis_ids = (
        "hypothesis-refused-1",
        "hypothesis-refused-2",
        "hypothesis-report",
        "hypothesis-safety",
        "hypothesis-execution",
    )
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
            hypothesis_ids=hypothesis_ids,
        )
    )
    identities = [
        CheckpointIdentity(
            analysis_id=exact,
            workspace_id="workspace-1",
            commit_id="commit-1",
            hypothesis_id=hypothesis_id,
        )
        for hypothesis_id in hypothesis_ids
    ]
    for identity, stage, status, code in zip(
        identities,
        (
            SimpleStage.POC_CANDIDATE_DONE,
            SimpleStage.POC_CANDIDATE_DONE,
            SimpleStage.REPORT_DONE,
            SimpleStage.POC_CANDIDATE_DONE,
            SimpleStage.POC_EXECUTION_DONE,
        ),
        (
            StageStatus.BLOCKED,
            StageStatus.BLOCKED,
            StageStatus.FAILED,
            StageStatus.FAILED,
            StageStatus.BLOCKED,
        ),
        (
            "POC_MODEL_REFUSED",
            "POC_MODEL_REFUSED",
            "CURSOR_PROMPT_TOO_LARGE",
            "CURSOR_MODEL_SAFETY_BLOCK",
            "RECOVERY_EXHAUSTED",
        ),
        strict=True,
    ):
        application._store.save_checkpoint(
            StageCheckpoint(
                identity=identity,
                stage=stage,
                status=status,
                input_refs=(),
                input_hash=input_reference_hash(()),
                attempt_number=1,
                error_code=code,
                retryable=False,
            )
        )

    result = application.retry(
        display,
        hypothesis_ids=hypothesis_ids[:4],
        allow_terminal_retry=True,
    )

    assert result["status"] == "READY_TO_RESUME"
    assert result["hypothesis_ids"] == list(hypothesis_ids[:4])
    assert (tmp_path / str(result["backup_path"])).is_file()
    for identity in identities[:4]:
        stage = (
            SimpleStage.REPORT_DONE
            if identity == identities[2]
            else SimpleStage.POC_CANDIDATE_DONE
        )
        assert application._store.require(identity, stage).status is StageStatus.PENDING
    remaining = application._store.require(
        identities[4], SimpleStage.POC_EXECUTION_DONE
    )
    assert remaining.status is StageStatus.BLOCKED
    assert remaining.error_code == "RECOVERY_EXHAUSTED"


def test_public_retry_backs_up_database_and_reopens_display_analysis(
    tmp_path: Path,
) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
            hypothesis_ids=("hypothesis-1",),
        )
    )
    identity = CheckpointIdentity(
        analysis_id=exact,
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    application._store.save_checkpoint(_exhausted(identity))

    result = application.retry(display)

    backup = tmp_path / str(result["backup_path"])
    assert result["status"] == "READY_TO_RESUME"
    assert result["hypothesis_id"] == "hypothesis-1"
    assert backup.is_file()
    assert backup.stat().st_size > 0


def test_public_retry_repairs_failed_poc_with_previous_execution_feedback(
    tmp_path: Path,
) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
            hypothesis_ids=("hypothesis-1",),
        )
    )
    identity = CheckpointIdentity(
        analysis_id=exact,
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    content_ref = artifacts.put_bytes(b"#!/bin/sh\nexit 2\n", "text/x-shellscript")
    candidate_ref = artifacts.put_json(
        {
            "kind": "simple_poc_candidate",
            "content_ref": content_ref.model_dump(mode="json"),
        }
    )
    candidate_outputs = (candidate_ref, content_ref)
    candidate = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_CANDIDATE_DONE,
        stage_version=STAGE_VERSION[SimpleStage.POC_CANDIDATE_DONE],
        status=StageStatus.SUCCEEDED,
        input_refs=(),
        input_hash=input_reference_hash(()),
        output_refs=candidate_outputs,
        attempt_id="old-attempt",
    )
    application._store.save_checkpoint(candidate)
    stderr_ref = artifacts.put_bytes(b"sample config rejected\n", "text/plain")
    execution_ref = artifacts.put_json(
        {
            "kind": "simple_poc_execution",
            "exit_code": 2,
            "timed_out": False,
            "stderr_ref": stderr_ref.model_dump(mode="json"),
        }
    )
    blocked = StageCheckpoint(
        identity=identity,
        stage=SimpleStage.POC_EXECUTION_DONE,
        stage_version=STAGE_VERSION[SimpleStage.POC_EXECUTION_DONE],
        status=StageStatus.BLOCKED,
        input_refs=candidate_outputs,
        input_hash=input_reference_hash(candidate_outputs),
        output_refs=(execution_ref, stderr_ref),
        attempt_number=MAX_RECOVERY_ATTEMPTS,
        error_code="RECOVERY_EXHAUSTED",
    )
    application._store.save_checkpoint(blocked)

    result = application.retry(display)

    pending = application._store.require(identity, SimpleStage.POC_CANDIDATE_DONE)
    assert result["current_stage"] == SimpleStage.POC_CANDIDATE_DONE.value
    assert pending.status is StageStatus.PENDING
    assert pending.attempt_number == MAX_RECOVERY_ATTEMPTS - 1
    assert pending.recovery_lineage_id is not None
    assert pending.input_refs == (candidate_ref, execution_ref)
    assert application._store.get(identity, SimpleStage.POC_EXECUTION_DONE) is None

    obsolete_ref = artifacts.put_json({"kind": "obsolete_retry_context"})
    accumulated = (*pending.input_refs, obsolete_ref)
    application._store.save_checkpoint(
        pending.model_copy(
            update={
                "status": StageStatus.FAILED,
                "input_refs": accumulated,
                "input_hash": input_reference_hash(accumulated),
                "attempt_number": MAX_RECOVERY_ATTEMPTS,
                "error_code": "CURSOR_REQUEST_FAILED",
            }
        )
    )

    again = application.retry(display)
    compact = application._store.require(identity, SimpleStage.POC_CANDIDATE_DONE)
    assert again["status"] == "READY_TO_RESUME"
    assert compact.input_refs == (candidate_ref, execution_ref)


def test_public_retry_reopens_current_retryable_blocked_checkpoint(
    tmp_path: Path,
) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
            hypothesis_ids=("hypothesis-1",),
        )
    )
    identity = CheckpointIdentity(
        analysis_id=exact,
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    application._store.save_checkpoint(_retryable_blocked(identity))

    result = application.retry(display)

    assert result["status"] == "READY_TO_RESUME"
    assert result["retry_source_error"] == "CURSOR_TIMED_OUT"
    assert (
        application._store.require(identity, SimpleStage.POC_CANDIDATE_DONE).status
        is StageStatus.PENDING
    )


def test_public_retry_rebuilds_only_degraded_environment_chain(
    tmp_path: Path,
) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-rebuild"
    display = application._display.get_or_allocate(exact)
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-rebuild",
            commit_id="a" * 40,
            repository="https://example.invalid/repository.git",
            hypothesis_ids=("hypothesis-rebuild",),
        )
    )
    identity = CheckpointIdentity(
        analysis_id=exact,
        workspace_id="workspace-rebuild",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-rebuild",
    )
    recipe_ref = SimpleArtifactRepository(tmp_path, identity).put_json(
        {
            "kind": "simple_environment_recipe",
            "degraded": True,
            "dockerfile_source": "GENERATED_NO_INSTALL",
        }
    )
    for stage, status in (
        (SimpleStage.PRO_CON_DONE, StageStatus.SUCCEEDED),
        (SimpleStage.VERIFICATION_INITIAL_DONE, StageStatus.SUCCEEDED),
        (SimpleStage.POC_CANDIDATE_DONE, StageStatus.SUCCEEDED),
        (SimpleStage.POC_EXECUTION_DONE, StageStatus.PENDING),
    ):
        application._store.save_checkpoint(
            StageCheckpoint(
                identity=identity,
                stage=stage,
                status=status,
                input_refs=(),
                input_hash=input_reference_hash(()),
                recipe_ref=(
                    recipe_ref if stage is not SimpleStage.PRO_CON_DONE else None
                ),
                image_digest=(
                    f"sha256:{'a' * 64}"
                    if stage is not SimpleStage.PRO_CON_DONE
                    else None
                ),
            )
        )

    result = application.retry(display, rebuild_environment=True)

    assert result["status"] == "READY_TO_RESUME"
    assert result["current_stage"] == SimpleStage.VERIFICATION_INITIAL_DONE.value
    assert (tmp_path / str(result["backup_path"])).is_file()
    assert application._store.get(identity, SimpleStage.PRO_CON_DONE) is not None
    assert (
        application._store.get(identity, SimpleStage.VERIFICATION_INITIAL_DONE) is None
    )
    assert application._store.get(identity, SimpleStage.POC_CANDIDATE_DONE) is None
    assert application._store.get(identity, SimpleStage.POC_EXECUTION_DONE) is None


def test_public_retry_reopens_failed_cursor_output(tmp_path: Path) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
        )
    )
    identity = CheckpointIdentity(
        analysis_id=exact,
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id=None,
    )
    application._store.save_checkpoint(
        StageCheckpoint(
            identity=identity,
            stage=SimpleStage.HYPOTHESIS_DONE,
            status=StageStatus.FAILED,
            input_refs=(),
            input_hash=input_reference_hash(()),
            attempt_id="attempt-3",
            attempt_number=MAX_RECOVERY_ATTEMPTS,
            error_code="CURSOR_INVALID_OUTPUT",
            retryable=False,
        )
    )

    result = application.retry(display)

    assert result["status"] == "READY_TO_RESUME"
    assert result["retry_source_error"] == "CURSOR_INVALID_OUTPUT"
    assert (
        application._store.require(identity, SimpleStage.HYPOTHESIS_DONE).status
        is StageStatus.PENDING
    )


def test_public_retry_reopens_elapsed_budget_failure(tmp_path: Path) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
            hypothesis_ids=("hypothesis-1",),
        )
    )
    identity = CheckpointIdentity(
        analysis_id=exact,
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    application._store.save_checkpoint(
        StageCheckpoint(
            identity=identity,
            stage=SimpleStage.VERIFICATION_INITIAL_DONE,
            status=StageStatus.FAILED,
            input_refs=(),
            input_hash=input_reference_hash(()),
            attempt_id="attempt-2",
            attempt_number=2,
            error_code="LLM_ELAPSED_BUDGET_EXHAUSTED",
            retryable=False,
        )
    )

    result = application.retry(display)

    assert result["status"] == "READY_TO_RESUME"
    assert result["retry_source_error"] == "LLM_ELAPSED_BUDGET_EXHAUSTED"
    checkpoint = application._store.require(
        identity, SimpleStage.VERIFICATION_INITIAL_DONE
    )
    assert checkpoint.status is StageStatus.PENDING


def test_public_retry_reopens_all_poc_timeouts_without_refused_candidate(
    tmp_path: Path,
) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    hypothesis_ids = ("hypothesis-1", "hypothesis-2", "hypothesis-3")
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
            hypothesis_ids=hypothesis_ids,
        )
    )
    identities = [
        CheckpointIdentity(
            analysis_id=exact,
            workspace_id="workspace-1",
            commit_id="commit-1",
            hypothesis_id=hypothesis_id,
        )
        for hypothesis_id in hypothesis_ids
    ]
    for identity, error_code in zip(
        identities,
        ("CURSOR_TIMED_OUT", "CURSOR_TIMED_OUT", "POC_MODEL_REFUSED"),
        strict=True,
    ):
        application._store.save_checkpoint(
            StageCheckpoint(
                identity=identity,
                stage=SimpleStage.POC_CANDIDATE_DONE,
                status=StageStatus.BLOCKED,
                input_refs=(),
                input_hash=input_reference_hash(()),
                attempt_number=1,
                error_code=error_code,
                retryable=error_code == "CURSOR_TIMED_OUT",
            )
        )

    result = application.retry(display, all_poc_timeouts=True)

    assert result["status"] == "READY_TO_RESUME"
    assert result["retry_count"] == 2
    assert result["hypothesis_ids"] == list(hypothesis_ids[:2])
    assert (tmp_path / str(result["backup_path"])).is_file()
    for identity in identities[:2]:
        checkpoint = application._store.require(
            identity, SimpleStage.POC_CANDIDATE_DONE
        )
        assert checkpoint.status is StageStatus.PENDING
    refused = application._store.require(
        identities[2], SimpleStage.POC_CANDIDATE_DONE
    )
    assert refused.status is StageStatus.BLOCKED
    assert refused.error_code == "POC_MODEL_REFUSED"


def test_public_retry_reopens_every_failed_poc_candidate(tmp_path: Path) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    hypothesis_ids = (
        "hypothesis-cursor",
        "hypothesis-source",
        "hypothesis-exhausted",
        "hypothesis-refused",
    )
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
            hypothesis_ids=hypothesis_ids,
        )
    )
    identities = [
        CheckpointIdentity(
            analysis_id=exact,
            workspace_id="workspace-1",
            commit_id="commit-1",
            hypothesis_id=hypothesis_id,
        )
        for hypothesis_id in hypothesis_ids
    ]
    for identity, error_code, status in zip(
        identities,
        (
            "CURSOR_REQUEST_FAILED",
            "POC_SOURCE_CONTEXT_UNAVAILABLE",
            "RECOVERY_EXHAUSTED",
            "POC_MODEL_REFUSED",
        ),
        (
            StageStatus.FAILED,
            StageStatus.BLOCKED,
            StageStatus.BLOCKED,
            StageStatus.BLOCKED,
        ),
        strict=True,
    ):
        application._store.save_checkpoint(
            StageCheckpoint(
                identity=identity,
                stage=SimpleStage.POC_CANDIDATE_DONE,
                status=status,
                input_refs=(),
                input_hash=input_reference_hash(()),
                attempt_number=(
                    MAX_RECOVERY_ATTEMPTS
                    if error_code == "RECOVERY_EXHAUSTED"
                    else 1
                ),
                error_code=error_code,
                retryable=False,
            )
        )

    result = application.retry(display)

    assert result["status"] == "READY_TO_RESUME"
    assert result["retry_count"] == 3
    assert result["hypothesis_ids"] == list(hypothesis_ids[:3])
    for identity in identities[:2]:
        checkpoint = application._store.require(
            identity, SimpleStage.POC_CANDIDATE_DONE
        )
        assert checkpoint.status is StageStatus.PENDING
        assert checkpoint.attempt_number == 1
    exhausted = application._store.require(
        identities[2], SimpleStage.POC_CANDIDATE_DONE
    )
    assert exhausted.status is StageStatus.PENDING
    assert exhausted.attempt_number == MAX_RECOVERY_ATTEMPTS - 1
    refused = application._store.require(
        identities[3], SimpleStage.POC_CANDIDATE_DONE
    )
    assert refused.status is StageStatus.BLOCKED
    assert refused.error_code == "POC_MODEL_REFUSED"


def test_public_retry_skips_non_exhausted_analysis_without_backup(
    tmp_path: Path,
) -> None:
    config = cast(Any, SimpleNamespace(data_dir=tmp_path))
    application = PublicSimpleRuntimeApplication(config, cast(Any, None))
    exact = "analysis-1"
    display = application._display.get_or_allocate(exact)
    application._store.save_analysis_run(
        SimpleAnalysisRun(
            analysis_id=exact,
            display_analysis_id=display,
            workspace_id="workspace-1",
            commit_id="commit-1",
            repository="https://example.invalid/repository.git",
        )
    )
    identity = CheckpointIdentity(
        analysis_id=exact,
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id=None,
    )
    application._store.save_checkpoint(
        StageCheckpoint(
            identity=identity,
            stage=SimpleStage.STATIC_DONE,
            status=StageStatus.SUCCEEDED,
            input_refs=(),
            input_hash=input_reference_hash(()),
        )
    )

    result = application.retry(display)

    assert result["retry_skipped_reason"] == "ANALYSIS_NOT_MANUALLY_RETRYABLE"
    assert not (tmp_path / "db" / "backups").exists()
