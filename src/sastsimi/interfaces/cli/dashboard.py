"""Blocking local dashboard command."""

from __future__ import annotations

import ipaddress
import sys
from pathlib import Path
from typing import TextIO

from sastsimi.config.runtime_paths import RuntimePaths
from sastsimi.dashboard.query import DashboardNotFound, DashboardQuery
from sastsimi.dashboard.server import serve_dashboard

from .progress import ProgressRenderer


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class DashboardConfigurationError(ValueError):
    pass


def run(data_dir: Path, host: str, port: int) -> None:
    if not _loopback(host):
        raise ValueError("DASHBOARD_LOOPBACK_ONLY")
    if not 1 <= port <= 65535:
        raise ValueError("DASHBOARD_PORT_INVALID")
    sys.stdout.write(
        f"대시보드: http://{host}:{port}\n"
        f"데이터 경로: {data_dir.resolve()}\n종료: Ctrl+C\n"
    )
    try:
        serve_dashboard(data_dir, host, port)
    except KeyboardInterrupt:
        return
    except DashboardNotFound as error:
        raise DashboardConfigurationError(
            "DASHBOARD_DATABASE_NOT_FOUND: 분석 DB가 없습니다. "
            "분석 때 사용한 XDG_CONFIG_HOME 또는 --data-dir을 지정하세요."
        ) from error


def progress_renderer(
    data_dir: Path, stream: TextIO, *, is_tty: bool
) -> ProgressRenderer:
    """Build the CLI renderer at the dashboard adapter boundary."""

    query = DashboardQuery(data_dir)
    return ProgressRenderer(
        stream=stream,
        is_tty=is_tty,
        log_dir=RuntimePaths(data_dir).logs,
        event_reader=query.list_events,
        active_call_reader=query.active_llm_call,
    )


__all__ = ["progress_renderer", "run"]
