from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sastsimi.contracts.canonical_json import canonical_bytes
from sastsimi.simple_runtime.application import StaticBootstrapResult
from sastsimi.simple_runtime.artifacts import SimpleArtifactRepository
from sastsimi.simple_runtime.bootstrap_stages import DirectHypothesisBootstrap
from sastsimi.simple_runtime.hypothesis_context import project_hypothesis_facts
from sastsimi.simple_runtime.models import CheckpointIdentity, StageFailure
from sastsimi.simple_runtime.provider import SimpleLLMCallResult


def test_projection_keeps_a_later_file_when_one_file_would_fill_the_budget() -> None:
    facts = [
        {
            "kind": "Call",
            "line": index,
            "name": "row.items",
            "path": "apps/_migrations/0001_initial.py",
        }
        for index in range(3000)
    ]
    facts.append(
        {
            "kind": "Call",
            "line": 12,
            "name": "cursor.execute",
            "path": "apps/api/views.py",
        }
    )
    findings = [
        {
            "rule_id": "rule-a",
            "path": "apps/old.py",
            "line": index,
            "message": "m" * 240,
            "snippet": "s" * 2000,
        }
        for index in range(800)
    ]
    findings.append(
        {
            "rule_id": "python.django.security.injection.sql.sql-injection",
            "path": "apps/api/views.py",
            "line": 40,
            "message": "query uses request data",
            "snippet": "cursor.execute(request.GET['q'])\n" * 40,
        }
    )
    ast_only = project_hypothesis_facts(
        {
            "kind": "simple_static_fact_bundle",
            "ast_summary": {"facts": facts, "parsed_file_count": 2},
            "opengrep_findings": [],
            "codeql_findings": [],
        }
    )
    with_findings = project_hypothesis_facts(
        {
            "kind": "simple_static_fact_bundle",
            "codeql_executed": False,
            "ast_summary": {"facts": [], "parsed_file_count": 2},
            "opengrep_findings": findings,
            "codeql_findings": [],
        }
    )

    assert len(canonical_bytes(ast_only)) <= 180_000
    assert len(canonical_bytes(with_findings)) <= 180_000
    assert any(item["path"] == "apps/api/views.py" for item in ast_only["ast_facts"])
    assert ast_only["omitted_ast_facts"] > 0
    assert any(
        item["path"] == "apps/api/views.py"
        for item in with_findings["opengrep_findings"]
    )
    assert with_findings["omitted_opengrep_findings"] > 0
    assert all(
        len(item.get("snippet", "")) <= 480
        for item in with_findings["opengrep_findings"]
    )


def test_projection_of_unknown_bundle_is_empty_and_bounded() -> None:
    projected = project_hypothesis_facts(["not", "a", "bundle"])

    assert projected["opengrep_findings"] == []
    assert projected["omitted_opengrep_findings"] == 0
    assert len(canonical_bytes(projected)) <= 180_000


@pytest.mark.asyncio
async def test_hypothesis_prompt_contains_finding_beyond_the_old_byte_prefix(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id=None,
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    facts = [
        {
            "kind": "Call",
            "line": index,
            "name": "row.items",
            "path": "apps/_migrations/0001_initial.py",
        }
        for index in range(4000)
    ]
    bundle_ref = artifacts.put_json(
        {
            "kind": "simple_static_fact_bundle",
            "ast_summary": {"facts": facts, "parsed_file_count": 933},
            "opengrep_findings": [
                {
                    "rule_id": "python.django.security.injection.sql.sql-injection",
                    "path": "apps/api/views.py",
                    "line": 40,
                    "snippet": "cursor.execute(request.GET['q'])",
                }
            ],
            "codeql_findings": [],
            "codeql_executed": False,
        }
    )
    raw_bundle = artifacts.read(bundle_ref)
    assert b"apps/api/views.py" not in raw_bundle[: 256 * 1024]

    class Client:
        prompt = b""

        async def call(
            self,
            *,
            prompt: bytes,
            output_schema: Mapping[str, Any],
            timeout_ms: int,
            agent_name: str = "agent",
        ) -> StageFailure:
            del output_schema, timeout_ms
            assert agent_name == "hypothesis"
            self.prompt = prompt
            return StageFailure(
                code="TEST_STOP", retryable=False, safe_message="test only"
            )

    client = Client()
    result = await DirectHypothesisBootstrap(
        data_dir=tmp_path,
        client_factory=lambda *_args: client,
    ).propose(
        identity,
        StaticBootstrapResult(
            repository_profile_ref=artifacts.put_json({"kind": "profile"}),
            static_bundle_ref=bundle_ref,
            workspace_path=tmp_path,
        ),
    )

    assert isinstance(result, StageFailure)
    assert result.code == "TEST_STOP"
    context = client.prompt.split(b"<UNTRUSTED_EXACT_INPUTS>\n", 1)[1]
    context = context.split(b"\n</UNTRUSTED_EXACT_INPUTS>", 1)[0]
    data = json.loads(context)["exact_inputs"][0]["data"]
    assert data["kind"] == "simple_hypothesis_fact_projection"
    assert data["opengrep_findings"][0]["path"] == "apps/api/views.py"
    assert b"omitted_* counts" in client.prompt
    assert b"at most" not in client.prompt
    assert b"each distinct issue" in client.prompt
    assert b"facts make plausible" in client.prompt
    assert b"issue supported by the facts" not in client.prompt


def _proposal(title: str) -> dict[str, object]:
    return {
        "title": title,
        "vulnerability_type": "TEST",
        "summary": title,
        "code_locations": ["pkg.py:1"],
        "source": "input",
        "sink": "output",
        "rationale": title,
    }


@pytest.mark.asyncio
async def test_hypothesis_count_keeps_distinct_issues_past_the_old_quota(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-count",
        workspace_id="workspace-count",
        commit_id="b" * 40,
        hypothesis_id=None,
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    bundle_ref = artifacts.put_json(
        {
            "kind": "simple_static_fact_bundle",
            "ast_summary": {"facts": [], "parsed_file_count": 1},
            "opengrep_findings": [],
            "codeql_findings": [],
            "codeql_executed": False,
        }
    )
    proposals = [_proposal(f"issue-{index}") for index in range(15)]
    proposals.append(_proposal("issue-0"))

    class Client:
        async def call(self, **_kwargs: object) -> SimpleLLMCallResult:
            return SimpleLLMCallResult(
                value={"hypotheses": proposals},
                prompt_digest="prompt",
                output_digest="output",
            )

    seeds = await DirectHypothesisBootstrap(
        data_dir=tmp_path,
        client_factory=lambda *_args: Client(),
    ).propose(
        identity,
        StaticBootstrapResult(
            repository_profile_ref=artifacts.put_json({"kind": "profile"}),
            static_bundle_ref=bundle_ref,
            workspace_path=tmp_path,
        ),
    )

    assert not isinstance(seeds, StageFailure)
    assert len(seeds) == 15


@pytest.mark.asyncio
async def test_hypothesis_runaway_list_stops_after_distinct_issues(
    tmp_path: Path,
) -> None:
    identity = CheckpointIdentity(
        analysis_id="analysis-runaway",
        workspace_id="workspace-runaway",
        commit_id="c" * 40,
        hypothesis_id=None,
    )
    artifacts = SimpleArtifactRepository(tmp_path, identity)
    bundle_ref = artifacts.put_json(
        {
            "kind": "simple_static_fact_bundle",
            "ast_summary": {"facts": [], "parsed_file_count": 1},
            "opengrep_findings": [],
            "codeql_findings": [],
            "codeql_executed": False,
        }
    )

    class Client:
        async def call(self, **_kwargs: object) -> SimpleLLMCallResult:
            hypotheses = [_proposal(f"issue-{index}") for index in range(8)]
            return SimpleLLMCallResult(
                value={"hypotheses": hypotheses},
                prompt_digest="prompt",
                output_digest="output",
            )

    seeds = await DirectHypothesisBootstrap(
        data_dir=tmp_path,
        client_factory=lambda *_args: Client(),
        max_hypotheses=3,
    ).propose(
        identity,
        StaticBootstrapResult(
            repository_profile_ref=artifacts.put_json({"kind": "profile"}),
            static_bundle_ref=bundle_ref,
            workspace_path=tmp_path,
        ),
    )

    assert not isinstance(seeds, StageFailure)
    assert len(seeds) == 3
