from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sastsimi.simple_runtime.artifacts import SimpleArtifactRepository
from sastsimi.simple_runtime.models import CheckpointIdentity
from sastsimi.simple_runtime.retrieval import collect_requested_sources


def test_host_paths_and_bad_spans_are_refused(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("first\nsecond\n", encoding="utf-8")
    result = collect_requested_sources(
        (
            "../secret",
            "C:\\Users\\secret",
            "\\\\host\\share",
            "app.py:0-1",
            "app.py:5-6",
        ),
        workspace=tmp_path,
        tracked=("app.py",),
    )

    assert result["served"] == []
    assert len(result["refused"]) == 5


def test_only_tracked_in_root_lines_are_served(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("first\nsecond\n", encoding="utf-8")
    (tmp_path / "local.env").write_text("SECRET=hidden\n", encoding="utf-8")
    result = collect_requested_sources(
        ("app.py:2-2", "local.env"), workspace=tmp_path, tracked=("app.py",)
    )

    assert len(result["served"]) == 1
    assert result["served"][0]["content"] == "2|second"
    assert result["refused"][0]["reason"] == "NOT_TRACKED"


def test_outside_symlink_is_refused(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-source.txt"
    outside.write_text("private", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        return
    result = collect_requested_sources(
        ("link.txt",), workspace=tmp_path, tracked=("link.txt",)
    )
    assert result["served"] == []
    assert result["refused"][0]["reason"] == "PATH_OUTSIDE_REPOSITORY"


def test_tracked_symlink_to_untracked_local_file_is_refused(tmp_path: Path) -> None:
    local = tmp_path / "local.env"
    local.write_text("SECRET=hidden", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(local)
    except OSError:
        return
    result = collect_requested_sources(
        ("link.txt",), workspace=tmp_path, tracked=("link.txt",)
    )
    assert result["served"] == []


def test_symlink_classification_is_enforced_without_windows_privilege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "link.txt"
    link.write_text("SECRET=hidden", encoding="utf-8")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == link or original(path))

    result = collect_requested_sources(
        ("link.txt",), workspace=tmp_path, tracked=("link.txt",)
    )

    assert result["served"] == []


def test_oversized_source_is_refused_before_reading_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "large.py"
    source.write_bytes(b"x" * 1_000)
    original = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path == source:
            raise AssertionError("oversized source content should not be read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)

    result = collect_requested_sources(
        ("large.py",),
        workspace=tmp_path,
        tracked=("large.py",),
        max_total_bytes=100,
    )

    assert result["served"] == []
    assert result["refused"] == [
        {"path": "large.py", "reason": "TOTAL_BUDGET_EXHAUSTED"}
    ]


def test_requested_path_count_limit_refuses_extra_files(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text("first = 1\n", encoding="utf-8")
    (tmp_path / "second.py").write_text("second = 2\n", encoding="utf-8")

    result = collect_requested_sources(
        ("first.py", "second.py"),
        workspace=tmp_path,
        tracked=("first.py", "second.py"),
        max_requests=1,
    )

    assert [item["path"] for item in result["served"]] == ["first.py"]
    assert result["refused"] == [
        {"path": "second.py", "reason": "REQUEST_LIMIT_EXCEEDED"}
    ]


def test_escaped_source_never_truncates_prompt_artifact(tmp_path: Path) -> None:
    (tmp_path / "newlines.py").write_bytes(b"\n" * 128_000)
    identity = CheckpointIdentity(
        analysis_id="analysis-1",
        workspace_id="workspace-1",
        commit_id="a" * 40,
        hypothesis_id="hypothesis-1",
    )
    artifacts = SimpleArtifactRepository(tmp_path / "data", identity)

    result = collect_requested_sources(
        ("newlines.py",)
        + tuple(f"missing-{index}-" + "x" * 220 for index in range(31)),
        workspace=tmp_path,
        tracked=("newlines.py",),
        max_total_bytes=128_000,
        max_requests=32,
        max_artifact_bytes=240_000,
    )
    ref = artifacts.put_json(result)
    context = json.loads(artifacts.prompt_context((ref,)))

    assert context["exact_inputs"][0]["data"]["kind"] == "simple_requested_sources"
    assert result["served"] == []
    assert len(result["refused"]) == 32
    assert result["refused"][-1] == {
        "path": "newlines.py",
        "reason": "PROMPT_BUDGET_EXHAUSTED",
    }


def test_large_source_is_served_as_a_leading_excerpt(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("value = 1\n" * 400, encoding="utf-8")
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")

    result = collect_requested_sources(
        ("big.py", "small.py"),
        workspace=tmp_path,
        tracked=("big.py", "small.py"),
        max_total_bytes=1_000,
        max_file_bytes=400,
        max_artifact_bytes=96_000,
    )

    served = {item["path"]: item for item in result["served"]}
    assert served["big.py"]["excerpt"] is True
    assert served["big.py"]["content"].endswith("\n")
    assert len(served["big.py"]["content"]) <= 400
    assert "excerpt" not in served["small.py"]
    assert result["served_bytes"] <= 1_000
    assert result["refused"] == []


def test_pinned_blob_larger_than_budget_is_excerpted(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "init"), cwd=repo, check=True, capture_output=True)
    (repo / "big.py").write_text("value = 1\n" * 200, encoding="utf-8")
    subprocess.run(("git", "add", "big.py"), cwd=repo, check=True, capture_output=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-m",
            "init",
        ),
        cwd=repo,
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=repo, text=True
    ).strip()

    result = collect_requested_sources(
        ("big.py",),
        workspace=repo,
        tracked=("big.py",),
        pinned_commit=commit,
        max_total_bytes=2_000,
        max_file_bytes=400,
    )

    assert result["served"][0]["excerpt"] is True
    assert result["served_bytes"] <= 400
    assert result["refused"] == []
