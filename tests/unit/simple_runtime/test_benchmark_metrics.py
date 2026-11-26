from __future__ import annotations

import hashlib

import pytest

from sastsimi.contracts.refs import StoredDataRef
from sastsimi.simple_runtime.benchmark import (
    StaticGroundTruthComparison,
    build_benchmark_metrics,
)
from sastsimi.simple_runtime.models import (
    CheckpointIdentity,
    SimpleStage,
    StageCheckpoint,
    StageStatus,
    input_reference_hash,
)


def _identity(name: str) -> CheckpointIdentity:
    return CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id=name,
    )


def _ref(name: str) -> StoredDataRef:
    digest = hashlib.sha256(name.encode()).hexdigest()
    return StoredDataRef(
        stored_data_id=digest,
        data_kind="artifact",
        content_hash=digest,
        workspace_id="workspace-1",
        commit_id="a" * 40,
        record_id=None,
    )


def _checkpoint(
    hypothesis: str,
    stage: SimpleStage,
    status: StageStatus,
    *,
    error_code: str | None = None,
    validated: bool = False,
) -> StageCheckpoint:
    return StageCheckpoint(
        identity=_identity(hypothesis),
        stage=stage,
        status=status,
        input_refs=(),
        input_hash=input_reference_hash(()),
        error_code=error_code,
        validated_poc_ref=_ref(f"validated-{hypothesis}") if validated else None,
    )


def test_benchmark_metrics_keep_static_dynamic_and_refusal_rates_separate() -> None:
    checkpoints = (
        _checkpoint(
            "hypothesis-1", SimpleStage.POC_CANDIDATE_DONE, StageStatus.SUCCEEDED
        ),
        _checkpoint(
            "hypothesis-2",
            SimpleStage.POC_CANDIDATE_DONE,
            StageStatus.BLOCKED,
            error_code="POC_MODEL_REFUSED",
        ),
        _checkpoint(
            "hypothesis-1",
            SimpleStage.POC_EXECUTION_DONE,
            StageStatus.SUCCEEDED,
            validated=True,
        ),
    )
    comparison = StaticGroundTruthComparison(
        true_positive_ids=("GT-1", "GT-2"),
        false_positive_ids=("PRED-3",),
        false_negative_ids=("GT-4",),
    )

    metrics = build_benchmark_metrics(
        checkpoints=checkpoints,
        static_detection_count=3,
        poc_attempt_statuses={
            "SUCCEEDED": 2,
            "CURSOR_REFUSED": 1,
            "CURSOR_MODEL_SAFETY_BLOCK": 1,
        },
        static_comparison=comparison,
    )

    assert metrics["static_detection"] == {
        "status": "MEASURED",
        "detected_count": 3,
        "ground_truth_count": 3,
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 1,
        "precision": 0.666667,
        "recall": 0.666667,
        "f1": 0.666667,
    }
    assert metrics["poc_generation"] == {
        "attempted_hypotheses": 2,
        "succeeded_hypotheses": 1,
        "refused_hypotheses": 1,
        "success_rate": 0.5,
    }
    assert metrics["dynamic_validation"] == {
        "attempted_hypotheses": 1,
        "succeeded_hypotheses": 1,
        "success_rate": 1.0,
    }
    assert metrics["model_refusal"] == {
        "poc_llm_calls": 4,
        "refused_calls": 2,
        "refusal_rate": 0.5,
    }


def test_metrics_without_ground_truth_preserve_static_detection_count() -> None:
    metrics = build_benchmark_metrics(
        checkpoints=(),
        static_detection_count=4,
        poc_attempt_statuses={},
    )

    assert metrics["static_detection"] == {
        "status": "GROUND_TRUTH_NOT_ATTACHED",
        "detected_count": 4,
        "ground_truth_count": None,
        "true_positive": None,
        "false_positive": None,
        "false_negative": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    assert metrics["poc_generation"]["success_rate"] is None
    assert metrics["dynamic_validation"]["success_rate"] is None
    assert metrics["model_refusal"]["refusal_rate"] is None


def test_static_comparison_cannot_omit_a_detection() -> None:
    comparison = StaticGroundTruthComparison(
        true_positive_ids=("GT-1",),
        false_positive_ids=(),
        false_negative_ids=(),
    )

    with pytest.raises(ValueError, match="BENCHMARK_STATIC_DETECTION_COUNT_MISMATCH"):
        build_benchmark_metrics(
            checkpoints=(),
            static_detection_count=2,
            poc_attempt_statuses={},
            static_comparison=comparison,
        )
