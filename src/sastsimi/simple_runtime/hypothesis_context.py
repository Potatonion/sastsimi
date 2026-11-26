"""Bounded static-fact view for hypothesis prompts.

The stored bundle keeps every fact. A hypothesis prompt only accepts 256 KiB,
and a raw prefix of a large bundle is the AST summary, so later tool findings
never reach the model. This projection spends that budget on findings first
and spreads both findings and AST calls across files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sastsimi.contracts.canonical_json import canonical_bytes

_PROJECTION_BUDGET = 180_000
_SNIPPET_CHARS = 480
_MESSAGE_CHARS = 240
_FACT_TEXT_CHARS = 200


def project_hypothesis_facts(bundle: object) -> dict[str, Any]:
    """Return a prompt-sized projection that keeps tool findings reachable."""

    source = bundle if isinstance(bundle, dict) else {}
    opengrep = _compact_findings(source.get("opengrep_findings"))
    codeql = _compact_findings(source.get("codeql_findings"))
    facts, ast_meta = _ast_facts(source.get("ast_summary"))
    projection: dict[str, Any] = {
        "kind": "simple_hypothesis_fact_projection",
        "codeql_executed": source.get("codeql_executed") is True,
        "ast_summary": ast_meta,
        "opengrep_findings": [],
        "codeql_findings": [],
        "ast_facts": [],
        "omitted_opengrep_findings": len(opengrep),
        "omitted_codeql_findings": len(codeql),
        "omitted_ast_facts": len(facts),
    }
    opengrep_snippets = _fit(
        projection,
        "opengrep_findings",
        _diverse(opengrep),
        "omitted_opengrep_findings",
    )
    codeql_snippets = _fit(
        projection,
        "codeql_findings",
        _diverse(codeql),
        "omitted_codeql_findings",
    )
    _attach_snippets(projection, opengrep_snippets, codeql_snippets)
    _fit(projection, "ast_facts", _diverse(facts), "omitted_ast_facts")
    _enforce_budget(projection, len(opengrep), len(codeql), len(facts))
    return projection


def _ast_facts(value: object) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(value, dict):
        return [], {}
    meta = {
        key: value[key]
        for key in (
            "parsed_file_count",
            "truncated",
            "parse_error_count",
            "oversize_count",
        )
        if key in value
    }
    raw_facts = value.get("facts")
    if not isinstance(raw_facts, list):
        return [], meta
    facts = [_compact_fact(item) for item in raw_facts if isinstance(item, dict)]
    return facts, meta


def _compact_findings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_compact_finding(item) for item in value if isinstance(item, dict)]


def _compact_finding(item: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("rule_id", "engine", "path", "line"):
        if key not in item:
            continue
        field = item[key]
        if key == "line" and not isinstance(field, int):
            continue
        if isinstance(field, str):
            field = field[:_FACT_TEXT_CHARS]
        compact[key] = field
    message = item.get("message")
    if isinstance(message, str) and message.strip():
        compact["message"] = message[:_MESSAGE_CHARS]
    snippet = item.get("snippet")
    if isinstance(snippet, str) and snippet.strip():
        compact["snippet"] = snippet[:_SNIPPET_CHARS]
    return compact


def _compact_fact(item: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("kind", "line", "name", "path"):
        if key not in item:
            continue
        field = item[key]
        if key == "line" and not isinstance(field, int):
            continue
        if isinstance(field, str):
            field = field[:_FACT_TEXT_CHARS]
        compact[key] = field
    return compact


def _diverse(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in items:
        path = item.get("path")
        key = path if isinstance(path, str) else ""
        bucket = buckets.get(key)
        if bucket is None:
            bucket = []
            buckets[key] = bucket
            order.append(key)
        bucket.append(item)
    selected: list[dict[str, Any]] = []
    index = 0
    while True:
        progressed = False
        for key in order:
            bucket = buckets[key]
            if index < len(bucket):
                selected.append(bucket[index])
                progressed = True
        if not progressed:
            return selected
        index += 1


def _fit(
    projection: dict[str, Any],
    field: str,
    candidates: Sequence[dict[str, Any]],
    omitted_key: str,
) -> list[str | None]:
    selected: list[dict[str, Any]] = projection[field]
    snippets: list[str | None] = []
    used = len(canonical_bytes(projection))
    stalled = 0
    for item in candidates:
        snippet = item.get("snippet")
        payload = (
            {key: value for key, value in item.items() if key != "snippet"}
            if "snippet" in item
            else item
        )
        extra = len(canonical_bytes(payload)) + 1
        if used + extra > _PROJECTION_BUDGET:
            stalled += 1
            if stalled >= 8:
                break
            continue
        selected.append(payload)
        snippets.append(snippet if isinstance(snippet, str) else None)
        used += extra
        stalled = 0
    projection[omitted_key] = len(candidates) - len(selected)
    return snippets


def _attach_snippets(
    projection: dict[str, Any],
    opengrep_snippets: Sequence[str | None],
    codeql_snippets: Sequence[str | None],
) -> None:
    pairs = zip(projection["opengrep_findings"], opengrep_snippets, strict=True)
    codeql_pairs = zip(projection["codeql_findings"], codeql_snippets, strict=True)
    used = len(canonical_bytes(projection))
    for item, snippet in (*pairs, *codeql_pairs):
        if not snippet:
            continue
        addition = len(snippet.encode("utf-8")) + 14
        if used + addition > _PROJECTION_BUDGET:
            return
        item["snippet"] = snippet
        used += addition


def _enforce_budget(
    projection: dict[str, Any],
    opengrep_total: int,
    codeql_total: int,
    ast_total: int,
) -> None:
    while len(canonical_bytes(projection)) > _PROJECTION_BUDGET:
        if projection["ast_facts"]:
            projection["ast_facts"].pop()
            continue
        if _drop_snippet(projection["opengrep_findings"]) or _drop_snippet(
            projection["codeql_findings"]
        ):
            continue
        if projection["opengrep_findings"]:
            projection["opengrep_findings"].pop()
            continue
        if projection["codeql_findings"]:
            projection["codeql_findings"].pop()
            continue
        break
    projection["omitted_opengrep_findings"] = opengrep_total - len(
        projection["opengrep_findings"]
    )
    projection["omitted_codeql_findings"] = codeql_total - len(
        projection["codeql_findings"]
    )
    projection["omitted_ast_facts"] = ast_total - len(projection["ast_facts"])


def _drop_snippet(findings: list[dict[str, Any]]) -> bool:
    for item in reversed(findings):
        if "snippet" in item:
            item.pop("snippet")
            return True
    return False
