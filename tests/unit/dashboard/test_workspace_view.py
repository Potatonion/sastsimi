from sastsimi.dashboard.models import (
    AnalysisDetailView,
    HypothesisProgressView,
    StageProgressView,
)
from sastsimi.dashboard.workspace import analysis_summary, analysis_tab


class _Query:
    def __init__(self, detail: AnalysisDetailView) -> None:
        self._detail = detail

    def get_analysis(self, analysis_id: str) -> AnalysisDetailView:
        assert analysis_id == "analysis-1"
        return self._detail

    def list_events(self, analysis_id: str, after_event_id: str | None = None):
        del analysis_id, after_event_id
        return ()


def _detail() -> AnalysisDetailView:
    return AnalysisDetailView(
        analysis_id="analysis-1",
        display_analysis_id="A-004",
        current_stage="VERIFICATION_FINAL_DONE",
        status="PARTIAL",
        completed_count=7,
        stage_count=14,
        hypothesis_count=1,
        finding_count=0,
        progress_percent=99,
        repository="https://github.com/example/backupsheep.git",
        logs_url="/api/analyses/A-004/logs/download",
        bundle_url="/api/analyses/A-004/bundle.zip",
        hypotheses=(
            HypothesisProgressView(
                analysis_id="analysis-1",
                hypothesis_id="hypothesis-1",
                current_stage="VERIFICATION_FINAL_DONE",
                status="SUCCEEDED",
                completed_count=5,
                stage_count=12,
                verdict="FALSE",
            ),
        ),
        pipeline=(
            StageProgressView(
                stage="VERIFICATION_FINAL_DONE",
                label_ko="최종 검증",
                status="SUCCEEDED",
                agent_role="Verification Agent",
            ),
        ),
    )


def test_summary_keeps_local_progress_and_skips_missing_pipeline_counts() -> None:
    summary = analysis_summary(_Query(_detail()), "analysis-1")

    assert summary["display_analysis_id"] == "A-004"
    assert summary["progress_percent"] == 99
    assert summary["kpis"]["verification_done"] == 7
    assert summary["candidate_total_count"] is None
    assert summary["finding_group_count"] == 0
    assert summary["presentation_bundle_url"] is None
    assert "hypotheses" not in summary


def test_missing_group_tabs_stay_empty_while_local_hypotheses_remain() -> None:
    query = _Query(_detail())

    findings = analysis_tab(query, "analysis-1", "findings", offset=0, limit=50)
    outputs = analysis_tab(query, "analysis-1", "outputs", offset=0, limit=50)

    assert findings["items"][0]["hypothesis_id"] == "hypothesis-1"
    assert findings["finding_traces"] == []
    assert outputs["finding_groups"] == []
    assert outputs["finding_traces"] == []
    assert outputs["reports"] == []
