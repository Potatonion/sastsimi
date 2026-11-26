from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from sastsimi.simple_runtime.artifacts import SimpleArtifactRepository
from sastsimi.simple_runtime.models import CheckpointIdentity, SimpleStage, StageFailure
from sastsimi.simple_runtime.stages import build_stage_handlers


@pytest.mark.asyncio
async def test_stage_client_uses_profile_timeout_for_local_llm_calls(
    tmp_path: Path,
) -> None:
    class RecordingClient:
        timeouts: list[int]

        def __init__(self) -> None:
            self.timeouts = []

        async def call(
            self,
            *,
            prompt: bytes,
            output_schema: Mapping[str, Any],
            timeout_ms: int,
            agent_name: str = "agent",
        ) -> StageFailure:
            del prompt, output_schema, agent_name
            self.timeouts.append(timeout_ms)
            return StageFailure(code="TEST_STOP", retryable=False, safe_message="test")

    client = RecordingClient()
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="commit-1",
        hypothesis_id="hypothesis-1",
    )
    handlers = build_stage_handlers(
        client=client,
        artifacts=SimpleArtifactRepository(tmp_path, identity),
        docker=cast(Any, None),
        containers=cast(Any, None),
        llm_timeout_ms=600_000,
    )
    stage_client = cast(Any, handlers[SimpleStage.PRO_CON_DONE])._con._client

    await stage_client.call(prompt=b"test", output_schema={}, timeout_ms=180_000)
    await stage_client.call(prompt=b"test", output_schema={}, timeout_ms=120_000)

    assert client.timeouts == [600_000, 120_000]
