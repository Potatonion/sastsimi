"""Stage-separated benchmark metrics for the small local runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .models import SimpleStage, StageCheckpoint, StageStatus


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


@dataclass(frozen=True, slots=True)
class StaticGroundTruthComparison:
    """A grader-owned mapping between static detections and ground truth.

    The runtime intentionally does not infer these matches from PoC outcomes.
    That would hide static true positives whenever dynamic reproduction fails.
    """

    true_positive_ids: tuple[str, ...]
    false_positive_ids: tuple[str, ...]
    false_negative_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (
            self.true_positive_ids,
            self.false_positive_ids,
            self.false_negative_ids,
        )
        if any(not item or item != item.strip() for group in groups for item in group):
            raise ValueError("BENCHMARK_COMPARISON_ID_INVALID")
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("BENCHMARK_COMPARISON_ID_DUPLICATE")
        if set(self.true_positive_ids) & set(self.false_negative_ids):
            raise ValueError("BENCHMARK_GROUND_TRUTH_CLASS_CONFLICT")


def build_benchmark_metrics(
    *,
    checkpoints: Sequence[StageCheckpoint],
    static_detection_count: int,
    poc_attempt_statuses: Mapping[str, int],
    static_comparison: StaticGroundTruthComparison | None = None,
) -> dict[str, object]:
    """Summarize static, generation, execution, and refusal outcomes separately."""

    invalid_attempt_count = any(value < 0 for value in poc_attempt_statuses.values())
    if static_detection_count < 0 or invalid_attempt_count:
        raise ValueError("BENCHMARK_COUNT_INVALID")

    candidates = tuple(
        item
        for item in checkpoints
        if item.identity.hypothesis_id is not None
        and item.stage is SimpleStage.POC_CANDIDATE_DONE
        and item.status is not StageStatus.PENDING
    )
    executions = tuple(
        item
        for item in checkpoints
        if item.identity.hypothesis_id is not None
        and item.stage is SimpleStage.POC_EXECUTION_DONE
        and item.status is not StageStatus.PENDING
    )
    generated = sum(item.status is StageStatus.SUCCEEDED for item in candidates)
    validated = sum(item.validated_poc_ref is not None for item in executions)
    refused_hypotheses = sum(
        item.error_code == "POC_MODEL_REFUSED" for item in candidates
    )

    poc_calls = sum(poc_attempt_statuses.values())
    refused_calls = sum(
        poc_attempt_statuses.get(status, 0)
        for status in ("CURSOR_REFUSED", "CURSOR_MODEL_SAFETY_BLOCK")
    )

    if static_comparison is None:
        static_metrics: dict[str, object] = {
            "status": "GROUND_TRUTH_NOT_ATTACHED",
            "detected_count": static_detection_count,
            "ground_truth_count": None,
            "true_positive": None,
            "false_positive": None,
            "false_negative": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    else:
        true_positive = len(static_comparison.true_positive_ids)
        false_positive = len(static_comparison.false_positive_ids)
        false_negative = len(static_comparison.false_negative_ids)
        if true_positive + false_positive != static_detection_count:
            raise ValueError("BENCHMARK_STATIC_DETECTION_COUNT_MISMATCH")
        precision = _rate(true_positive, true_positive + false_positive)
        recall = _rate(true_positive, true_positive + false_negative)
        f1 = _rate(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        )
        static_metrics = {
            "status": "MEASURED",
            "detected_count": static_detection_count,
            "ground_truth_count": true_positive + false_negative,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "static_detection": static_metrics,
        "poc_generation": {
            "attempted_hypotheses": len(candidates),
            "succeeded_hypotheses": generated,
            "refused_hypotheses": refused_hypotheses,
            "success_rate": _rate(generated, len(candidates)),
        },
        "dynamic_validation": {
            "attempted_hypotheses": len(executions),
            "succeeded_hypotheses": validated,
            "success_rate": _rate(validated, len(executions)),
        },
        "model_refusal": {
            "poc_llm_calls": poc_calls,
            "refused_calls": refused_calls,
            "refusal_rate": _rate(refused_calls, poc_calls),
        },
    }


__all__ = ["StaticGroundTruthComparison", "build_benchmark_metrics"]
