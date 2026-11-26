"""Cursor Agent SDK boundary for schema-constrained, text-only analysis calls.

The SDK is an agent API, not a JSON completion API.  This module therefore
validates every response locally and never treats malformed output as evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Protocol, TextIO
from uuid import uuid4

from sastsimi.contracts.canonical_json import canonical_bytes
from sastsimi.contracts.refs import StoredDataRef

from .artifacts import SimpleArtifactRepository
from .models import StageFailure
from .provider import (
    SimpleLLMCallResult,
    SimpleLLMClient,
    _validate_schema,
)
from .store import SimpleCheckpointStore
from .usage_values import canonical_cost, cost_minor_units, token_count

_LOG = logging.getLogger(__name__)
_MAX_RAW_BYTES = 512 * 1024
_POC_REFUSAL_HEAD = re.compile(
    r"^\s*(?:(?:i(?:'m|\s+am)\s+sorry)[,.]?\s+)?(?:"
    r"i\s+(?:can(?:not|['’]t)|won['’]t|will\s+not)\s+"
    r"(?:help\b|assist\b|provide\b|write\b|produce\b)|"
    r"(?:죄송(?:하지만|합니다)?[,.]?\s*)?"
    r"(?:(?:해당\s+)?(?:요청|내용|코드)(?:을|를)?\s+)?"
    r"(?:도와드릴|제공할|작성할|생성할|쓸)\s*수\s*없|"
    r"못\s*(?:쓰겠|작성하겠|제공하겠|도와드리겠)"
    r")",
    re.IGNORECASE,
)
_POC_REFUSAL_SUBJECT = re.compile(
    r"\b(?:poc|proof[\s.-]*of[\s.-]*concept|exploit|script)\b|"
    r"개념\s*증명|재현\s*(?:코드|스크립트)|공격\s*(?:코드|스크립트)|스크립트",
    re.IGNORECASE,
)
_POC_TECHNICAL_INABILITY = re.compile(
    r"\b(?:invalid|malformed|schema|json|parse|missing|insufficient|"
    r"dependency|prerequisite|source\s+context|evidence|cannot\s+verify|"
    r"failed\s+test|does\s+not\s+reproduce)\b|"
    r"형식|스키마|파싱|누락|부족|의존성|선행\s*조건|소스|증거|검증할\s*수\s*없|"
    r"재현되지\s*않",
    re.IGNORECASE,
)
_POC_POLICY_REFUSAL = re.compile(
    r"\b(?:safety|policy|harmful|malicious|unauthori[sz]ed|cybersecurity|"
    r"cross-user|credential|attack|abuse)\b|"
    r"안전|정책|악용|공격|무단|권한\s*침해|위험|사이버",
    re.IGNORECASE,
)


def _explicit_poc_refusal(raw: str) -> bool:
    """Recognize a direct policy refusal, not quality or parse commentary."""

    text = raw.strip()[:4_096]
    if not _POC_REFUSAL_HEAD.match(text) or not _POC_REFUSAL_SUBJECT.search(text):
        return False
    if _POC_TECHNICAL_INABILITY.search(text) and not _POC_POLICY_REFUSAL.search(text):
        return False
    # A short, direct refusal often omits the policy explanation. Longer
    # responses need an explicit safety signal so quality feedback is not
    # mislabeled as a provider refusal.
    return len(text) <= 512 or bool(_POC_POLICY_REFUSAL.search(text))


class CursorTransport(Protocol):
    async def list_models(self, api_key: str) -> set[str]: ...

    async def complete(
        self, *, api_key: str, model: str, prompt: str, timeout: float
    ) -> tuple[str, dict[str, int | float | None]]: ...


class OfficialCursorTransport:
    """One SDK bridge per request so cancellation always tears down the child."""

    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

    async def list_models(self, api_key: str) -> set[str]:
        from cursor_sdk import AsyncClient, AsyncCursor

        async with await AsyncClient.launch_bridge(
            workspace=self._workspace, client_timeout=30, max_retries=0
        ) as client:
            models = await AsyncCursor.models.list(client=client, api_key=api_key)
            return {item.id for item in models}

    async def complete(
        self, *, api_key: str, model: str, prompt: str, timeout: float
    ) -> tuple[str, dict[str, int | float | None]]:
        from cursor_sdk import AgentOptions, AsyncClient, LocalAgentOptions

        agent = None
        run = None
        async with await AsyncClient.launch_bridge(
            workspace=self._workspace,
            client_timeout=timeout,
            max_retries=0,
        ) as client:
            try:
                agent = await client.agents.create(
                    AgentOptions(
                        model=model,
                        api_key=api_key,
                        tools=[],
                        local=LocalAgentOptions(
                            cwd=self._workspace, setting_sources=[]
                        ),
                    )
                )
                run = await agent.send(prompt)
                result = await run.wait()
                if str(result.status).lower() != "finished":
                    raise RuntimeError("CURSOR_RUN_UNSUCCESSFUL")
                usage = result.usage
                metrics: dict[str, int | float | None] = {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "cost_minor_units": None,
                }
                try:
                    settled = await agent.get_usage()
                    cents = getattr(
                        getattr(settled, "cost", None), "charged_cents", None
                    )
                    metrics["cost_minor_units"] = cents
                except Exception:
                    # Usage settlement can lag the response; token counts remain.
                    pass
                return result.result, metrics
            except asyncio.CancelledError:
                if run is not None:
                    try:
                        await asyncio.wait_for(run.cancel(), timeout=5)
                    except Exception:
                        pass
                raise
            finally:
                if agent is not None:
                    try:
                        await asyncio.wait_for(agent.close(), timeout=5)
                    except Exception:
                        pass


class OfficialCursorCLITransport:
    """Official headless CLI using its own browser-login session, never a key."""

    _MODEL_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._:/+-]{0,159}) - .+$")
    _MAX_PROMPT_ARG_BYTES = 96 * 1024
    _MAX_STREAM_LINE_BYTES = 2 * 1024 * 1024
    _MAX_STREAM_EVENTS = 2_000
    _TRACE_PROGRESS_INTERVAL = 250
    _MAX_STDERR_BYTES = 64 * 1024

    def __init__(
        self,
        executable: str,
        *,
        seed_prompt_sources: bool = False,
        stream_json: bool = False,
        trace_dir: Path | None = None,
        trace_label: str = "cursor",
    ) -> None:
        self._executable = executable
        self._script = os.path.join(os.path.dirname(executable), "index.js")
        self._seed_prompt_sources = seed_prompt_sources
        self._stream_json = stream_json
        self._trace_dir = trace_dir
        if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", trace_label) is None:
            raise ValueError("CURSOR_TRACE_LABEL_INVALID")
        self._trace_label = trace_label
        if not os.path.isfile(self._executable) or not os.path.isfile(self._script):
            raise ValueError("CURSOR_CLI_NOT_INSTALLED")

    async def _run(
        self,
        *args: str,
        timeout: float,
        prompt_sources: str | None = None,
        trace_prompt: str | None = None,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="sastsimi-cursor-") as workspace:
            if prompt_sources is not None:
                self._stage_prompt_sources(Path(workspace), prompt_sources)
            flags = (
                int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if os.name == "nt"
                else 0
            )
            allowed_env = {
                "PATH",
                "PATHEXT",
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "PROGRAMFILES",
                "PROGRAMDATA",
                "USERPROFILE",
                "HOME",
                "HOMEDRIVE",
                "HOMEPATH",
                "APPDATA",
                "LOCALAPPDATA",
                "TEMP",
                "TMP",
                "TMPDIR",
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "SSL_CERT_FILE",
                "CURSOR_API_ENDPOINT",
            }
            child_env = {
                name: value
                for name, value in os.environ.items()
                if name.upper() in allowed_env
            }
            trace: TextIO | None = None
            if (
                self._stream_json
                and trace_prompt is not None
                and self._trace_dir is not None
            ):
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = (
                    self._trace_dir
                    / f"cursor-poc-{self._trace_label}-{uuid4().hex}.jsonl"
                )
                trace = trace_path.open("x", encoding="utf-8")
                if os.name != "nt":
                    trace_path.chmod(0o600)
                self._trace_event(
                    trace,
                    "started",
                    phase=(
                        "plan"
                        if trace_prompt.startswith(
                            "You are planning a local, harmless PoC"
                        )
                        else "candidate"
                    ),
                    prompt_bytes=len(trace_prompt.encode("utf-8")),
                    prompt_sha256=hashlib.sha256(
                        trace_prompt.encode("utf-8")
                    ).hexdigest(),
                    timeout_seconds=timeout,
                )
            try:
                process = await asyncio.create_subprocess_exec(
                    self._executable,
                    self._script,
                    *args,
                    cwd=workspace,
                    env=child_env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=flags,
                    limit=self._MAX_STREAM_LINE_BYTES if self._stream_json else 2**16,
                )
            except BaseException:
                if trace is not None:
                    trace.close()
                raise
            try:
                if self._stream_json and trace_prompt is not None:
                    stdout, stderr = await asyncio.wait_for(
                        self._collect_stream_json(process, trace), timeout=timeout
                    )
                else:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
            except BaseException as error:
                if trace is not None:
                    self._trace_event(
                        trace,
                        "timed_out" if isinstance(error, TimeoutError) else "aborted",
                        error_type=type(error).__name__,
                    )
                if os.name == "nt" and process.pid is not None:
                    try:
                        terminator = await asyncio.create_subprocess_exec(
                            "taskkill",
                            "/PID",
                            str(process.pid),
                            "/T",
                            "/F",
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                            creationflags=flags,
                        )
                        await asyncio.wait_for(terminator.wait(), timeout=5)
                    except Exception:
                        pass
                if process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                with suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=5)
                raise
            finally:
                if trace is not None:
                    trace.close()
            if process.returncode != 0:
                self._raise_cli_error(stderr + stdout)
            if self._stream_json and trace_prompt is not None and not stdout:
                raise CursorCLIExecutionError()
            return stdout.decode("utf-8", errors="strict")

    @classmethod
    async def _collect_stream_json(
        cls, process: asyncio.subprocess.Process, trace: TextIO | None
    ) -> tuple[bytes, bytes]:
        assert process.stdout is not None and process.stderr is not None
        stderr_task = asyncio.create_task(cls._drain_stderr(process.stderr))
        terminal: dict[str, Any] | None = None
        event_count = 0
        stream_bytes = 0
        try:
            async for line in process.stdout:
                stream_bytes += len(line)
                if len(line) > cls._MAX_STREAM_LINE_BYTES:
                    raise CursorCLIExecutionError()
                try:
                    event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CursorCLIExecutionError() from error
                if not isinstance(event, dict):
                    raise CursorCLIExecutionError()
                event_count += 1
                kind = event.get("type")
                subtype = event.get("subtype")
                safe_kind = (
                    kind
                    if isinstance(kind, str)
                    and kind in {"system", "user", "assistant", "tool_call", "result"}
                    else "other"
                )
                safe_kind_hash = (
                    hashlib.sha256(kind.encode("utf-8")).hexdigest()[:12]
                    if safe_kind == "other" and isinstance(kind, str)
                    else None
                )
                if trace is not None and event_count <= cls._MAX_STREAM_EVENTS:
                    safe_subtype = (
                        subtype
                        if subtype
                        in {"init", "started", "completed", "success", "error"}
                        else None
                    )
                    cls._trace_event(
                        trace,
                        "event",
                        index=event_count,
                        type=safe_kind,
                        type_hash=safe_kind_hash,
                        subtype=safe_subtype,
                    )
                elif trace is not None and event_count == cls._MAX_STREAM_EVENTS + 1:
                    cls._trace_event(trace, "event_limit_reached")
                elif (
                    trace is not None
                    and event_count > cls._MAX_STREAM_EVENTS
                    and event_count % cls._TRACE_PROGRESS_INTERVAL == 0
                ):
                    cls._trace_event(
                        trace,
                        "event_progress",
                        event_count=event_count,
                        stream_bytes=stream_bytes,
                        last_type=safe_kind,
                        last_type_hash=safe_kind_hash,
                    )
                if event.get("type") == "result":
                    if terminal is not None:
                        raise CursorCLIExecutionError()
                    terminal = event
            await process.wait()
            stderr = await stderr_task
            if trace is not None:
                stderr_text = stderr.decode("utf-8", errors="replace").lower()
                stderr_class = (
                    "OPENAI_CYBER_SAFETY_BLOCK"
                    if "high-risk cybersecurity" in stderr_text
                    or "flagged this request" in stderr_text
                    else None
                )
                cls._trace_event(
                    trace,
                    "exited",
                    returncode=process.returncode,
                    event_count=event_count,
                    stream_bytes=stream_bytes,
                    result_seen=terminal is not None,
                    stderr_bytes=len(stderr),
                    stderr_class=stderr_class,
                )
            return (
                json.dumps(terminal, ensure_ascii=False).encode("utf-8")
                if terminal is not None
                else b"",
                stderr,
            )
        finally:
            if not stderr_task.done():
                stderr_task.cancel()
                with suppress(asyncio.CancelledError):
                    await stderr_task

    @classmethod
    async def _drain_stderr(cls, stream: asyncio.StreamReader) -> bytes:
        captured = bytearray()
        while chunk := await stream.read(8192):
            remaining = cls._MAX_STDERR_BYTES - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
        return bytes(captured)

    @staticmethod
    def _trace_event(trace: TextIO, event: str, **fields: Any) -> None:
        trace.write(
            json.dumps(
                {"at": datetime.now(UTC).isoformat(), "event": event, **fields},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        trace.flush()

    @staticmethod
    def _raise_cli_error(output: bytes) -> None:
        detail = output.decode("utf-8", errors="replace").lower()
        if "rate limit" in detail or "too many requests" in detail or "429" in detail:
            raise CursorCLIRateLimitError()
        if "login" in detail or "authenticat" in detail:
            raise CursorCLIAuthenticationError()
        if "model" in detail and ("invalid" in detail or "unavailable" in detail):
            raise CursorCLIModelError()
        if "upgrade" in detail or "usage limit" in detail or "plan limit" in detail:
            raise CursorCLIPlanLimitError()
        if any(
            marker in detail
            for marker in (
                "internal server error",
                "service unavailable",
                "502",
                "503",
                "504",
                "network error",
                "connection failed",
            )
        ):
            raise CursorCLITemporaryError()
        if "high-risk cybersecurity" in detail or "flagged this request" in detail:
            raise CursorCLISafetyBlockError()
        raise CursorCLIExecutionError()

    async def list_models(self, api_key: str) -> set[str]:
        del api_key
        output = await self._run("models", timeout=30)
        models = {
            match.group(1)
            for line in output.splitlines()
            if (match := self._MODEL_LINE.fullmatch(line.strip())) is not None
        }
        if not models:
            raise CursorCLIModelCatalogError()
        return models

    async def complete(
        self, *, api_key: str, model: str, prompt: str, timeout: float
    ) -> tuple[str, dict[str, int | float | None]]:
        del api_key
        prompt_args = self._prompt_args(prompt)
        sandbox_args = ("--sandbox", "enabled") if self._seed_prompt_sources else ()
        output = await self._run(
            "--print",
            "--trust",
            "--mode",
            "ask",
            *sandbox_args,
            "--output-format",
            "stream-json" if self._stream_json else "json",
            "--model",
            model,
            "--",
            *prompt_args,
            timeout=timeout,
            prompt_sources=prompt if self._seed_prompt_sources else None,
            trace_prompt=prompt if self._stream_json else None,
        )
        try:
            envelope = json.loads(output)
        except (json.JSONDecodeError, TypeError) as error:
            raise CursorCLIExecutionError() from error
        if (
            not isinstance(envelope, dict)
            or envelope.get("type") != "result"
            or envelope.get("subtype") != "success"
            or envelope.get("is_error") is not False
            or not isinstance(envelope.get("result"), str)
        ):
            raise CursorCLIExecutionError()
        return envelope["result"], {}

    @staticmethod
    def _stage_prompt_sources(workspace: Path, prompt: str) -> None:
        """Expose only snippets already sent in this request, never the checkout."""

        marker = "<UNTRUSTED_EXACT_INPUTS>\n"
        end_marker = "\n</UNTRUSTED_EXACT_INPUTS>"
        if marker not in prompt:
            return
        section = prompt.split(marker, 1)[1].split(end_marker, 1)[0]
        try:
            exact_inputs = json.loads(section)["exact_inputs"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("CURSOR_PROMPT_SOURCES_INVALID") from error
        if not isinstance(exact_inputs, list):
            raise ValueError("CURSOR_PROMPT_SOURCES_INVALID")
        count = 0
        total_bytes = 0
        for item in exact_inputs:
            data = item.get("data") if isinstance(item, dict) else None
            if (
                not isinstance(data, dict)
                or data.get("kind") != "simple_requested_sources"
            ):
                continue
            served = data.get("served")
            if not isinstance(served, list):
                raise ValueError("CURSOR_PROMPT_SOURCES_INVALID")
            for source in served:
                if not isinstance(source, dict):
                    raise ValueError("CURSOR_PROMPT_SOURCES_INVALID")
                raw_path, content = source.get("path"), source.get("content")
                if isinstance(content, dict):
                    chunks = content.get("text_chunks")
                    if (
                        content.get("join_without_separator") is not True
                        or not isinstance(chunks, list)
                        or not all(isinstance(chunk, str) for chunk in chunks)
                    ):
                        raise ValueError("CURSOR_PROMPT_SOURCES_INVALID")
                    content = "".join(chunks)
                if not isinstance(raw_path, str) or not isinstance(content, str):
                    raise ValueError("CURSOR_PROMPT_SOURCES_INVALID")
                path = PurePosixPath(raw_path)
                if (
                    path.is_absolute()
                    or not path.parts
                    or any(part in {".", ".."} for part in path.parts)
                    or any(part.startswith(".") for part in path.parts)
                    or "\\" in raw_path
                    or ":" in raw_path
                ):
                    raise ValueError("CURSOR_PROMPT_SOURCE_PATH_INVALID")
                encoded = content.encode("utf-8")
                count += 1
                total_bytes += len(encoded)
                if count > 64 or total_bytes > 256 * 1024:
                    raise ValueError("CURSOR_PROMPT_SOURCES_TOO_LARGE")
                target = workspace.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.read_bytes() != encoded:
                    raise ValueError("CURSOR_PROMPT_SOURCE_CONFLICT")
                target.write_bytes(encoded)

    @classmethod
    def _prompt_args(cls, prompt: str) -> tuple[str, ...]:
        """Use Cursor's variadic prompt arguments below Linux's per-arg limit."""
        if len(prompt.encode("utf-8")) <= cls._MAX_PROMPT_ARG_BYTES:
            return (prompt,)
        if os.name == "nt":
            raise CursorCLIPromptTooLargeError()
        parts: list[str] = []
        start = 0
        last_boundary = 0
        current_bytes = 0
        in_string = False
        escaped = False
        for index, char in enumerate(prompt):
            current_bytes += len(char.encode("utf-8"))
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in {",", "\n", " "}:
                last_boundary = index + 1
            if current_bytes > cls._MAX_PROMPT_ARG_BYTES:
                if last_boundary <= start:
                    raise CursorCLIPromptTooLargeError()
                parts.append(prompt[start:last_boundary])
                start = last_boundary
                current_bytes = len(prompt[start : index + 1].encode("utf-8"))
                last_boundary = 0
        if start < len(prompt):
            parts.append(prompt[start:])
        if any(len(part.encode("utf-8")) > cls._MAX_PROMPT_ARG_BYTES for part in parts):
            raise CursorCLIPromptTooLargeError()
        return tuple(parts)


class CursorCLIAuthenticationError(Exception):
    pass


class CursorCLIRateLimitError(Exception):
    pass


class CursorCLIModelError(Exception):
    pass


class CursorCLIModelCatalogError(Exception):
    pass


class CursorCLIExecutionError(Exception):
    is_retryable = True


class CursorCLISafetyBlockError(Exception):
    """Provider refused this prompt, so the same model call will not succeed."""


class CursorCLIPromptTooLargeError(Exception):
    pass


class CursorCLIPlanLimitError(Exception):
    pass


class CursorCLITemporaryError(Exception):
    pass


class CursorModelCatalog:
    def __init__(self) -> None:
        self.models: set[str] | None = None
        self.credential_digest: str | None = None
        self.refreshed_at: float = 0.0
        self.lock = asyncio.Lock()


def _ends_inside_string(prompt: str) -> bool:
    """Use the same quote rules as CLI prompt splitting."""

    in_string = False
    escaped = False
    for char in prompt:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
    return in_string


def _invalid_output_correction(detail: str, base_prompt: str) -> str:
    """Ask for corrected JSON without appending the schema a second time.

    The first prompt already contains the schema. Repeating it can extend an
    open quoted span past Linux's per-argument limit. A prior closing quote,
    when the splitter still considers the prompt inside a string, lets the
    following newline become a real split point.
    """

    closer = '"' if _ends_inside_string(base_prompt) else ""
    return (
        closer + "\nYour previous response failed JSON/schema validation at "
        + detail
        + ". Return a corrected JSON object matching the same schema "
        + "already given above.\n"
    )


def _unwrap_json(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        first_line, separator, body = stripped.partition("\n")
        if separator and first_line.lower() in {"```", "```json"}:
            return body[:-3].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        if start > 0:
            try:
                value, end = json.JSONDecoder().raw_decode(stripped[start:])
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(value, dict) and not stripped[start + end :].strip():
                    return stripped[start : start + end]
    return stripped


def _error_code(error: BaseException) -> tuple[str, bool]:
    name = type(error).__name__
    if name in {
        "AuthenticationError",
        "PermissionError",
        "CursorCLIAuthenticationError",
    }:
        return "CURSOR_AUTH_FAILED", False
    if name in {
        "ConfigurationError",
        "InvalidModelError",
        "CursorCLIModelError",
        "CursorCLIModelCatalogError",
    }:
        return "CURSOR_CONFIGURATION_FAILED", False
    if name == "CursorCLIPlanLimitError":
        return "CURSOR_PLAN_LIMIT", False
    if name == "CursorCLIPromptTooLargeError":
        return "CURSOR_PROMPT_TOO_LARGE", False
    if name in {"RateLimitError", "CursorCLIRateLimitError"}:
        return "CURSOR_RATE_LIMITED", True
    if name in {"APITimeoutError", "TimeoutError"}:
        return "CURSOR_TIMED_OUT", True
    if name in {"NetworkError", "InternalServerError", "CursorCLITemporaryError"}:
        return "CURSOR_TEMPORARY_FAILURE", True
    if name == "CursorCLISafetyBlockError":
        return "CURSOR_MODEL_SAFETY_BLOCK", False
    return "CURSOR_REQUEST_FAILED", bool(getattr(error, "is_retryable", False))


def _token_count(value: int | float | None) -> int | None:
    return token_count(value)


def _cost_cents(value: int | float | None) -> float | None:
    return cost_minor_units(value)


class CursorProvider:
    def __init__(
        self,
        *,
        artifacts: SimpleArtifactRepository,
        default_model: str,
        agent_models: Mapping[str, str],
        timeout_seconds: int,
        max_retries: int,
        semaphore: asyncio.Semaphore,
        allow_on_demand: bool,
        fallback: SimpleLLMClient | None = None,
        transport: CursorTransport | None = None,
        poc_transport: CursorTransport | None = None,
        use_cli_login: bool = False,
        model_catalog: CursorModelCatalog | None = None,
        budget_check: Callable[[], StageFailure | None] | None = None,
    ) -> None:
        self._artifacts = artifacts
        self._default_model = default_model
        self._agent_models = dict(agent_models)
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._semaphore = semaphore
        self._allow_on_demand = allow_on_demand
        self._fallback = fallback
        self._use_cli_login = use_cli_login
        self._transport = transport or OfficialCursorTransport(str(artifacts.data_dir))
        self._poc_transport = poc_transport
        self._attempt_store = SimpleCheckpointStore(artifacts.paths.database)
        self._catalog = model_catalog or CursorModelCatalog()
        self._budget_check = budget_check

    async def _models(self, key: str) -> set[str]:
        async with self._catalog.lock:
            marker = hashlib.sha256(key.encode("utf-8")).hexdigest()
            if (
                self._catalog.credential_digest != marker
                or monotonic() - self._catalog.refreshed_at > 600
            ):
                self._catalog.models = None
                self._catalog.credential_digest = marker
            if self._catalog.models is None:
                for attempt in range(self._max_retries + 1):
                    try:
                        self._catalog.models = await asyncio.wait_for(
                            self._transport.list_models(key), timeout=self._timeout
                        )
                        self._catalog.refreshed_at = monotonic()
                        break
                    except Exception as error:
                        _, retryable = _error_code(error)
                        if not retryable or attempt >= self._max_retries:
                            raise
                        await asyncio.sleep(min(8.0, 0.5 * 2**attempt))
            assert self._catalog.models is not None
            return self._catalog.models

    async def call(
        self,
        *,
        prompt: bytes,
        output_schema: Mapping[str, Any],
        timeout_ms: int,
        agent_name: str = "agent",
    ) -> SimpleLLMCallResult | StageFailure:
        key = "" if self._use_cli_login else os.environ.get("CURSOR_API_KEY", "")
        if not self._use_cli_login and (not key or key != key.strip()):
            return StageFailure(
                code="CURSOR_AUTH_REQUIRED",
                retryable=False,
                safe_message="Set CURSOR_API_KEY to your own Cursor API key",
            )
        if not self._allow_on_demand:
            return StageFailure(
                code="CURSOR_ON_DEMAND_CONTROL_UNAVAILABLE",
                retryable=False,
                safe_message=(
                    "Cursor SDK cannot guarantee on-demand usage is disabled; "
                    "set cursor_allow_on_demand=true only after checking Team billing"
                ),
            )
        model = self._agent_models.get(agent_name, self._default_model)
        try:
            models = await self._models(key)
        except Exception as error:
            code, retryable = _error_code(error)
            failure = StageFailure(
                code=code,
                retryable=retryable,
                safe_message="Cursor model catalog could not be read",
            )
            if retryable and self._fallback is not None:
                return await self._fallback.call(
                    prompt=prompt,
                    output_schema=output_schema,
                    timeout_ms=timeout_ms,
                    agent_name=agent_name,
                )
            return failure
        if model not in models:
            return StageFailure(
                code="CURSOR_MODEL_UNAVAILABLE",
                retryable=False,
                safe_message=(
                    f"Cursor model '{model}' is unavailable for this account; "
                    "run sastsimi cursor-models and select a returned ID"
                ),
            )

        timeout = min(self._timeout, max(1, timeout_ms) / 1000)
        schema_text = json.dumps(output_schema, ensure_ascii=False, sort_keys=True)
        base_prompt = (
            prompt.decode("utf-8")
            + "\nReturn exactly one JSON object, without commentary, "
            + "matching this schema:\n"
            + schema_text
        )
        prompt_digest = hashlib.sha256(prompt).hexdigest()
        last_failure: StageFailure | None = None
        correction = ""
        switched_late = False
        async with self._semaphore:
            # The configured call timeout includes retries and backoff. Giving
            # every retry a fresh full timeout can multiply a 10-minute wait.
            deadline = monotonic() + timeout
            for attempt in range(1, self._max_retries + 3):
                if attempt > self._max_retries + 1 and not switched_late:
                    break
                switched_late = False
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                if self._budget_check is not None:
                    budget_failure = self._budget_check()
                    if budget_failure is not None:
                        return budget_failure
                started_at = datetime.now(UTC)
                started = monotonic()
                invocation_id = f"cursor-{uuid4().hex}"
                raw_ref = None
                parsed_ref = None
                usage: dict[str, int | float | None] = {}
                code = "SUCCEEDED"
                self._attempt_store.start_llm_call(
                    invocation_id=invocation_id,
                    analysis_id=self._artifacts.identity.analysis_id,
                    agent=agent_name,
                    model=model,
                    attempt_number=attempt,
                    started_at=started_at,
                )
                try:
                    transport = (
                        self._poc_transport
                        if agent_name == "poc_candidate"
                        and self._poc_transport is not None
                        else self._transport
                    )
                    raw, usage = await asyncio.wait_for(
                        transport.complete(
                            api_key=key,
                            model=model,
                            prompt=base_prompt + correction,
                            timeout=remaining,
                        ),
                        timeout=remaining,
                    )
                    raw = raw[:_MAX_RAW_BYTES]
                    raw_ref = self._artifacts.put_json(
                        {"kind": "cursor_raw_output", "text": raw}
                    )
                    try:
                        value = json.loads(_unwrap_json(raw))
                        if not isinstance(value, dict):
                            raise ValueError("$: expected object")
                        _validate_schema(value, output_schema)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        if not (
                            agent_name == "poc_candidate"
                            and _explicit_poc_refusal(raw)
                        ):
                            raise
                        code = "CURSOR_REFUSED"
                        last_failure = StageFailure(
                            code=code,
                            retryable=False,
                            safe_message="Model declined PoC generation",
                            evidence_refs=(raw_ref,),
                        )
                        self._record_attempt(
                            agent_name,
                            model,
                            attempt,
                            started,
                            code,
                            raw_ref,
                            parsed_ref,
                            usage,
                        )
                        refusal_switch = (
                            model != self._default_model
                            and self._default_model in models
                        )
                        if refusal_switch:
                            model = self._default_model
                            if attempt > self._max_retries:
                                switched_late = True
                            continue
                        break
                    canonical = canonical_bytes(value)
                    parsed_ref = self._artifacts.put_bytes(
                        canonical, "application/json"
                    )
                    result = SimpleLLMCallResult(
                        value=value,
                        prompt_digest=prompt_digest,
                        output_digest=hashlib.sha256(canonical).hexdigest(),
                        invocation_id=invocation_id,
                        provider="cursor-cli" if self._use_cli_login else "cursor-sdk",
                        model=model,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        elapsed_ms=int((monotonic() - started) * 1000),
                        raw_output_ref=raw_ref,
                        parsed_output_ref=parsed_ref,
                        input_tokens=_token_count(usage.get("input_tokens")),
                        output_tokens=_token_count(usage.get("output_tokens")),
                        cost_minor_units=_cost_cents(usage.get("cost_minor_units")),
                        on_demand_possible=True,
                    )
                    self._record_attempt(
                        agent_name,
                        model,
                        attempt,
                        started,
                        code,
                        raw_ref,
                        parsed_ref,
                        usage,
                    )
                    return result
                except asyncio.CancelledError:
                    raise
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    code = "CURSOR_INVALID_OUTPUT"
                    detail = str(error).splitlines()[0][:200]
                    correction = _invalid_output_correction(detail, base_prompt)
                    last_failure = StageFailure(
                        code=code,
                        retryable=False,
                        safe_message="Cursor returned invalid structured output",
                    )
                except Exception as error:
                    code, retryable = _error_code(error)
                    safety_switch = (
                        isinstance(error, CursorCLISafetyBlockError)
                        and model != self._default_model
                        and self._default_model in models
                    )
                    last_failure = StageFailure(
                        code=code,
                        retryable=retryable,
                        safe_message=(
                            "PoC model refused; retrying on the default model"
                            if safety_switch
                            else "Cursor request did not complete"
                        ),
                    )
                    if safety_switch:
                        self._record_attempt(
                            agent_name,
                            model,
                            attempt,
                            started,
                            code,
                            raw_ref,
                            parsed_ref,
                            usage,
                        )
                        model = self._default_model
                        if attempt > self._max_retries:
                            switched_late = True
                        continue
                    if not retryable:
                        self._record_attempt(
                            agent_name,
                            model,
                            attempt,
                            started,
                            code,
                            raw_ref,
                            parsed_ref,
                            usage,
                        )
                        break
                finally:
                    self._attempt_store.finish_llm_call(
                        invocation_id=invocation_id,
                        analysis_id=self._artifacts.identity.analysis_id,
                    )
                self._record_attempt(
                    agent_name,
                    model,
                    attempt,
                    started,
                    code,
                    raw_ref,
                    parsed_ref,
                    usage,
                )
                if attempt <= self._max_retries:
                    delay = (
                        min(8.0, 0.5 * 2 ** (attempt - 1)) + random.uniform(0, 0.25)
                    )
                    if deadline - monotonic() <= delay:
                        break
                    await asyncio.sleep(delay)
        assert last_failure is not None
        remaining_ms = int(max(0, deadline - monotonic()) * 1000)
        if (
            self._fallback is not None
            and last_failure.retryable
            and remaining_ms > 0
        ):
            return await self._fallback.call(
                prompt=prompt,
                output_schema=output_schema,
                timeout_ms=remaining_ms,
                agent_name=agent_name,
            )
        return last_failure

    def _record_attempt(
        self,
        agent: str,
        model: str,
        attempt: int,
        started: float,
        status: str,
        raw_ref: StoredDataRef | None,
        parsed_ref: StoredDataRef | None,
        usage: Mapping[str, int | float | None],
    ) -> None:
        elapsed = int((monotonic() - started) * 1000)
        _LOG.info(
            "cursor_call analysis_id=%s agent=%s model=%s "
            "attempt=%d elapsed_ms=%d status=%s",
            self._artifacts.identity.analysis_id,
            agent,
            model,
            attempt,
            elapsed,
            status,
        )
        artifact_ref = self._artifacts.put_json(
            {
                "kind": "cursor_agent_attempt",
                "analysis_id": self._artifacts.identity.analysis_id,
                "agent": agent,
                "model": model,
                "attempt": attempt,
                "elapsed_ms": elapsed,
                "status": status,
                "raw_output_ref": raw_ref.model_dump(mode="json") if raw_ref else None,
                "parsed_output_ref": parsed_ref.model_dump(mode="json")
                if parsed_ref
                else None,
                "usage": {
                    "input_tokens": _token_count(usage.get("input_tokens")),
                    "output_tokens": _token_count(usage.get("output_tokens")),
                    "cost_minor_units": canonical_cost(
                        _cost_cents(usage.get("cost_minor_units"))
                    ),
                },
                "on_demand_possible": True,
            }
        )
        self._attempt_store.record_llm_attempt(
            attempt_id=uuid4().hex,
            analysis_id=self._artifacts.identity.analysis_id,
            agent=agent,
            model=model,
            attempt_number=attempt,
            status=status,
            elapsed_ms=elapsed,
            input_tokens=_token_count(usage.get("input_tokens")),
            output_tokens=_token_count(usage.get("output_tokens")),
            cost_cents=_cost_cents(usage.get("cost_minor_units")),
            artifact_ref=artifact_ref,
        )


__all__ = [
    "CursorModelCatalog",
    "CursorProvider",
    "OfficialCursorTransport",
    "OfficialCursorCLITransport",
]
