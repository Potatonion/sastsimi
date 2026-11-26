"""Bounded, tracked-only source reads requested by an untrusted Agent."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from sastsimi.contracts.canonical_json import canonical_bytes
from sastsimi.contracts.prompt_redaction import (
    redact_projected_json,
    redact_untrusted_text,
)

from .code_redaction import redact_code
from .facts import safe_tracked_file

_SPAN = re.compile(r"^(?P<path>[^:]+):(?P<start>\d+)(?:-(?P<end>\d+))?$")
_MAX_TOTAL_BYTES = 256_000
_MAX_EXCERPT_SOURCE_BYTES = 1_048_576
_MIN_EXCERPT_BYTES = 256


def collect_requested_sources(
    requests: Iterable[str],
    *,
    workspace: Path,
    tracked: Sequence[str] = (),
    already_supplied: Sequence[str] = (),
    max_total_bytes: int = _MAX_TOTAL_BYTES,
    pinned_commit: str | None = None,
    git_executable: str = "git",
    max_requests: int | None = None,
    max_artifact_bytes: int | None = None,
    max_file_bytes: int | None = None,
) -> dict[str, Any]:
    if max_requests is not None and max_requests < 0:
        raise ValueError("max_requests must be non-negative")
    if max_artifact_bytes is not None and max_artifact_bytes < 1024:
        raise ValueError("max_artifact_bytes must be at least 1024")
    if max_file_bytes is not None and max_file_bytes < _MIN_EXCERPT_BYTES:
        raise ValueError("max_file_bytes must be at least 256")
    available = set(tracked)
    supplied = set(already_supplied)
    served: list[dict[str, Any]] = []
    served_sizes: list[int] = []
    refused: list[dict[str, str]] = []
    seen: set[str] = set()
    total = 0
    for index, request in enumerate(requests):
        if max_requests is not None and index >= max_requests:
            refused.append({"path": str(request), "reason": "REQUEST_LIMIT_EXCEEDED"})
            continue
        if not isinstance(request, str):
            refused.append({"path": str(request), "reason": "NOT_A_PATH"})
            continue
        match = _SPAN.fullmatch(request)
        path = match.group("path") if match else request
        if (
            not path
            or path.startswith(("/", "\\"))
            or "\\" in path
            or ":" in path
            or "\x00" in path
            or ".." in Path(path).parts
        ):
            refused.append({"path": request, "reason": "PATH_OUTSIDE_REPOSITORY"})
            continue
        if path not in available:
            refused.append({"path": request, "reason": "NOT_TRACKED"})
            continue
        if request in seen or path in supplied:
            continue
        seen.add(request)
        excerpted = False
        file_limit = max_total_bytes - total
        if max_file_bytes is not None:
            file_limit = min(file_limit, max_file_bytes)
        try:
            if pinned_commit is not None:
                raw, reason = _read_pinned_blob(
                    workspace,
                    path,
                    commit=pinned_commit,
                    git_executable=git_executable,
                    remaining=max_total_bytes - total,
                    max_file_bytes=max_file_bytes,
                )
                if reason is not None:
                    refused.append({"path": request, "reason": reason})
                    continue
                if raw is None:
                    raise OSError("pinned Git blob is unavailable")
            else:
                candidate = safe_tracked_file(workspace, path)
                if candidate is None:
                    refused.append(
                        {"path": request, "reason": "PATH_OUTSIDE_REPOSITORY"}
                    )
                    continue
                size = candidate.stat().st_size
                if size > file_limit and (
                    max_file_bytes is None
                    or size > _MAX_EXCERPT_SOURCE_BYTES
                    or file_limit < _MIN_EXCERPT_BYTES
                ):
                    refused.append(
                        {"path": request, "reason": "TOTAL_BUDGET_EXHAUSTED"}
                    )
                    continue
                raw = candidate.read_bytes()
            if max_file_bytes is not None and len(raw) > file_limit:
                excerpt = _leading_excerpt(raw, file_limit)
                excerpted = len(excerpt) < len(raw)
                raw = excerpt
            if not raw or len(raw) + total > max_total_bytes:
                refused.append({"path": request, "reason": "TOTAL_BUDGET_EXHAUSTED"})
                continue
            text = raw.decode("utf-8")
        except (OSError, UnicodeError):
            refused.append({"path": request, "reason": "UNREADABLE"})
            continue
        if match:
            start = int(match.group("start"))
            end = int(match.group("end") or start)
            lines = text.splitlines()
            if start < 1 or end < start or end > len(lines):
                refused.append({"path": request, "reason": "LINES_OUTSIDE_FILE"})
                continue
            text = "\n".join(
                f"{line}|{lines[line - 1]}" for line in range(start, end + 1)
            )
        item = {"path": request, "content": redact_code(text, workspace=workspace)}
        if excerpted:
            item["excerpt"] = True
        served.append(item)
        served_sizes.append(len(raw))
        total += len(raw)
    result: dict[str, Any] = {
        "kind": "simple_requested_sources",
        "served": served,
        "refused": refused,
        "served_bytes": total,
    }
    if max_artifact_bytes is not None:
        while _projected_size(result) > max_artifact_bytes and served:
            removed = served.pop()
            total -= served_sizes.pop()
            refused.append(
                {"path": removed["path"], "reason": "PROMPT_BUDGET_EXHAUSTED"}
            )
            result["served_bytes"] = total
        if _projected_size(result) > max_artifact_bytes:
            result["omitted_refusals"] = len(refused)
            result["refused"] = [
                {"path": "<multiple>", "reason": "PROMPT_BUDGET_EXHAUSTED"}
            ]
    return result


def _projected_size(value: object) -> int:
    raw = canonical_bytes(value)
    try:
        return len(redact_projected_json(raw).data)
    except ValueError:
        return len(redact_untrusted_text(raw).data)


def _leading_excerpt(raw: bytes, limit: int) -> bytes:
    cut = raw[:limit]
    newline = cut.rfind(b"\n")
    if newline >= limit // 2:
        cut = cut[: newline + 1]
    while cut:
        try:
            cut.decode("utf-8")
        except UnicodeError:
            cut = cut[:-1]
        else:
            return cut
    return b""


def _read_pinned_blob(
    workspace: Path,
    path: str,
    *,
    commit: str,
    git_executable: str,
    remaining: int,
    max_file_bytes: int | None = None,
) -> tuple[bytes | None, str | None]:
    def git(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            (git_executable, "-C", str(workspace), *args),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )

    try:
        tree = git("--literal-pathspecs", "ls-tree", "-z", commit, "--", path)
        if tree.returncode != 0:
            return None, "PINNED_SOURCE_UNAVAILABLE"
        exact_suffix = b"\t" + path.encode("utf-8")
        entry = next(
            (item for item in tree.stdout.split(b"\0") if item.endswith(exact_suffix)),
            None,
        )
        if entry is None:
            return None, "NOT_IN_PINNED_COMMIT"
        mode, kind, object_id = entry.split(b"\t", 1)[0].split()
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            return None, "PATH_OUTSIDE_REPOSITORY"
        oid = object_id.decode("ascii")
        size = git("cat-file", "-s", oid)
        if size.returncode != 0:
            return None, "PINNED_SOURCE_UNAVAILABLE"
        blob_size = int(size.stdout.strip())
        file_limit = remaining
        if max_file_bytes is not None:
            file_limit = min(file_limit, max_file_bytes)
        if blob_size > file_limit and (
            max_file_bytes is None
            or blob_size > _MAX_EXCERPT_SOURCE_BYTES
            or file_limit < _MIN_EXCERPT_BYTES
        ):
            return None, "TOTAL_BUDGET_EXHAUSTED"
        content = git("cat-file", "blob", oid)
        if content.returncode != 0:
            return None, "PINNED_SOURCE_UNAVAILABLE"
        return content.stdout, None
    except (OSError, ValueError, UnicodeError, subprocess.TimeoutExpired):
        return None, "PINNED_SOURCE_UNAVAILABLE"
