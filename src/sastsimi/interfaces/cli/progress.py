"""Terminal progress rendering without synthetic time-based progress."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Protocol, TextIO

from sastsimi.progress.models import ProgressSnapshot

_SAFE_ANALYSIS_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class ProgressEvent(Protocol):
    event_id: str
    status: str
    stage: str
    agent_role: str
    tool_name: str | None
    summary_ko: str
    started_at: datetime


class ProgressRenderer:
    def __init__(
        self,
        *,
        stream: TextIO,
        is_tty: bool,
        width: int = 24,
        log_dir: Path | None = None,
        event_reader: Callable[[str], Iterable[ProgressEvent]] | None = None,
        active_call_reader: Callable[[str], tuple[str, str, int] | None] | None = None,
    ) -> None:
        self._stream = stream
        self._is_tty = is_tty
        self._width = width
        self._log_dir = log_dir
        self._event_reader = event_reader
        self._active_call_reader = active_call_reader
        self._seen_event_ids: set[str] = set()
        self._last_stage: str | None = None
        self._last_snapshot: ProgressSnapshot | None = None
        self._unchanged_since = monotonic()
        self._last_wait_report = self._unchanged_since

    def render(self, snapshot: ProgressSnapshot) -> None:
        stage = snapshot.current_stage or "준비 중"
        now = monotonic()
        if snapshot != self._last_snapshot:
            self._unchanged_since = now
            self._last_wait_report = now
        self._last_snapshot = snapshot
        waiting = int(now - self._unchanged_since)
        waiting_suffix = (
            f" · 진척 변경 없이 대기 {waiting}초"
            if snapshot.status == "RUNNING" and waiting >= 30
            else ""
        )
        if waiting_suffix and self._active_call_reader is not None:
            try:
                active = self._active_call_reader(snapshot.analysis_id)
            except (LookupError, OSError, ValueError, sqlite3.Error):
                active = None
            if active is not None:
                agent, model, elapsed = active
                waiting_suffix += f" · LLM {agent} / {model} 호출 {elapsed}초"
        self._append_log(snapshot, stage)
        self._render_new_events(snapshot.analysis_id)
        if not self._is_tty:
            if stage != self._last_stage:
                self._stream.write(
                    f"현재 단계: {stage} "
                    f"({snapshot.completed_units}/{snapshot.known_units})\n"
                )
                self._last_stage = stage
            elif waiting_suffix and now - self._last_wait_report >= 30:
                self._stream.write(f"현재 단계: {stage}{waiting_suffix}\n")
                self._last_wait_report = now
            self._stream.flush()
            return
        filled = round(self._width * snapshot.percent / 100)
        bar = "█" * filled + "-" * (self._width - filled)
        end = "\n" if snapshot.status in {"COMPLETE", "BLOCKED", "FAILED"} else ""
        color = {
            "COMPLETE": "\033[32m",
            "BLOCKED": "\033[33m",
            "FAILED": "\033[31m",
            "RUNNING": "\033[36m",
        }.get(snapshot.status, "\033[0m")
        self._stream.write(
            f"\r{color}[{bar}] {snapshot.percent:3d}% "
            f"{snapshot.completed_units}/{snapshot.known_units} {stage}"
            f"{waiting_suffix}"
            f"\033[0m{end}"
        )
        self._stream.flush()
        self._last_stage = stage

    def _append_log(self, snapshot: ProgressSnapshot, stage: str) -> None:
        if self._log_dir is None:
            return
        if _SAFE_ANALYSIS_ID.fullmatch(snapshot.analysis_id) is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / f"{snapshot.analysis_id}.log"
        timestamp = datetime.now(UTC).isoformat()
        hypothesis = snapshot.current_hypothesis_id or "-"
        line = (
            f"{timestamp} status={snapshot.status} stage={stage} "
            f"hypothesis={hypothesis} progress={snapshot.percent}% "
            f"units={snapshot.completed_units}/{snapshot.known_units} "
            f"error={snapshot.error_code or '-'}\n"
        )
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(line)

    def _render_new_events(self, analysis_id: str) -> None:
        if self._event_reader is None:
            return
        try:
            events = tuple(self._event_reader(analysis_id))
        except (LookupError, OSError, ValueError):
            return
        for event in events:
            if event.event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event.event_id)
            tool = event.tool_name or "-"
            summary = " ".join(event.summary_ko.splitlines())
            line = (
                f"[{event.status}] {event.stage} · {event.agent_role} · "
                f"tool={tool} · {summary}"
            )
            color = (
                "\033[31m"
                if event.status in {"FAILED", "BLOCKED"}
                else "\033[32m"
                if event.status == "SUCCEEDED"
                else "\033[36m"
            )
            if self._is_tty:
                self._stream.write(f"\r\033[K{color}{line}\033[0m\n")
            else:
                self._stream.write(line + "\n")
            self._append_event_log(analysis_id, event, tool)

    def _append_event_log(
        self,
        analysis_id: str,
        event: ProgressEvent,
        tool: str,
    ) -> None:
        if self._log_dir is None or _SAFE_ANALYSIS_ID.fullmatch(analysis_id) is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        path = self._log_dir / f"{analysis_id}.log"
        line = (
            json.dumps(
                {
                    "timestamp": event.started_at.isoformat(),
                    "event": event.event_id,
                    "status": event.status,
                    "stage": event.stage,
                    "agent": event.agent_role,
                    "tool": tool,
                    "summary": event.summary_ko,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(line)


__all__ = ["ProgressRenderer"]
