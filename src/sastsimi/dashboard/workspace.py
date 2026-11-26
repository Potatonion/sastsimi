"""Read-only tab payloads for the main dashboard layout.

The stored analysis stays on this branch's checkpoint shape. Tabs that need
finding groups, attack surfaces, or candidate batches stay empty.
"""

from __future__ import annotations

from typing import Any

from sastsimi.dashboard.models import AnalysisDetailView
from sastsimi.dashboard.query import DashboardNotFound, DashboardQuery

_TABS = frozenset(
    {
        "overview",
        "progress",
        "findings",
        "coverage",
        "artifacts",
        "llm",
        "outputs",
        "logs",
    }
)
_DETAIL_KEYS = (
    "hypotheses",
    "reports",
    "pipeline",
    "static_tools",
    "artifacts",
    "llm_invocations",
    "poc_artifact_ids",
    "evidence_artifact_ids",
    "static_coverage_gap_preview",
)


def analysis_summary(query: DashboardQuery, analysis_id: str) -> dict[str, Any]:
    detail = query.get_analysis(analysis_id)
    payload = detail.model_dump(mode="json")
    for key in _DETAIL_KEYS:
        payload.pop(key, None)
    payload["kpis"] = {
        "discovery_done": None,
        "discovery_total": None,
        "verification_done": detail.completed_count,
        "verification_total": detail.stage_count,
        "confirmed_findings": detail.finding_count,
    }
    payload["validated_poc_count"] = sum(
        1 for item in detail.hypotheses if item.validated_poc
    )
    payload["llm_token_usage_known"] = detail.llm_attempt_count > 0 and (
        detail.llm_unknown_cost_calls == 0
    )
    payload["logs_url"] = detail.logs_url
    payload["bundle_url"] = detail.bundle_url
    payload["presentation_bundle_url"] = None
    payload["error_code"] = next(
        (item.error_code for item in reversed(detail.pipeline) if item.error_code),
        None,
    )
    payload["finding_group_count"] = 0
    payload["candidate_total_count"] = None
    return payload


def analysis_tab(
    query: DashboardQuery,
    analysis_id: str,
    tab: str,
    *,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    if tab not in _TABS:
        raise DashboardNotFound("DASHBOARD_ROUTE_NOT_FOUND")
    detail = query.get_analysis(analysis_id)
    if tab == "overview":
        return {"readiness": []}
    if tab == "progress":
        return {
            "pipeline": [_dump(item) for item in detail.pipeline],
            "history": [_dump(item) for item in query.list_events(analysis_id)],
        }
    if tab == "findings":
        page = _page([_dump(item) for item in detail.hypotheses], offset, limit)
        page["finding_traces"] = []
        return page
    if tab == "coverage":
        payload = detail.model_dump(mode="json")
        payload["static_tool_findings"] = []
        return payload
    if tab == "artifacts":
        page = _page([_dump(item) for item in detail.artifacts], offset, limit)
        page["omitted_count"] = detail.artifact_omitted_count
        page["relations"] = []
        return page
    if tab == "llm":
        return _page([_dump(item) for item in detail.llm_invocations], offset, limit)
    if tab == "outputs":
        return {
            "artifacts": [_dump(item) for item in detail.artifacts],
            "poc_artifact_ids": list(detail.poc_artifact_ids),
            "evidence_artifact_ids": list(detail.evidence_artifact_ids),
            "reports": [_dump(item) for item in detail.reports],
            "finding_groups": [],
            "finding_traces": [],
        }
    return {"items": [], "offset": 0, "total": 0, "limit": limit}


def status_cells(
    query: DashboardQuery,
    analysis_id: str,
    *,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    detail = query.get_analysis(analysis_id)
    items = [
        {
            "id": item.hypothesis_id,
            "status": item.verdict or item.status,
            "label_ko": item.hypothesis_id,
        }
        for item in detail.hypotheses
    ]
    return _page(items, offset, limit)


def event_page(
    query: DashboardQuery,
    analysis_id: str,
    *,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    events = [_dump(item) for item in query.list_events(analysis_id)]
    return _page(events, offset, limit)


def llm_invocation(
    query: DashboardQuery, analysis_id: str, invocation_id: str
) -> dict[str, Any]:
    detail = query.get_analysis(analysis_id)
    match = next(
        (
            item
            for item in detail.llm_invocations
            if item.invocation_id == invocation_id
        ),
        None,
    )
    if match is None:
        raise DashboardNotFound("DASHBOARD_LLM_NOT_FOUND")
    return {
        "invocation": _dump(match),
        "system_prompt": None,
        "user_prompt": None,
        "response_result": None,
        "stored_request_json": None,
        "stored_response_json": None,
    }


def _page(items: list[dict[str, Any]], offset: int, limit: int) -> dict[str, Any]:
    start = max(0, offset)
    size = min(max(1, limit), 200)
    return {
        "items": items[start : start + size],
        "offset": start,
        "limit": size,
        "total": len(items),
    }


def _dump(value: AnalysisDetailView | object) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    if dump is None:
        raise TypeError("DASHBOARD_VIEW_INVALID")
    return dump(mode="json")
