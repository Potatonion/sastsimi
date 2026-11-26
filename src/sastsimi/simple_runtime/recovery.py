"""Bounded, typed recovery decisions for the sequential SimpleRuntime."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Protocol

from pydantic import ValidationError

from sastsimi.contracts.base import ContractModel
from sastsimi.contracts.canonical_json import canonical_bytes
from sastsimi.contracts.refs import StoredDataRef

from .artifacts import SimpleArtifactRepository
from .models import MAX_RECOVERY_ATTEMPTS, SimpleStage, StageCheckpoint, StageFailure
from .provider import SimpleLLMClient

_MAX_ENVIRONMENT_PATCH_BYTES = 8 * 1024
_RECOVERY_TIMEOUT_MS = 300_000
# Cursor CLI passes the complete prompt as one OS argument. Linux limits a
# single argument to 128 KiB even when ARG_MAX is much larger.
_MAX_RECOVERY_PROMPT_BYTES = 96 * 1024
_PLAYWRIGHT_BROWSERS = ("chromium", "firefox", "webkit")
_PLAYWRIGHT_PATCH_PREFIX = (
    "ENV PLAYWRIGHT_BROWSERS_PATH=/opt/sastsimi-playwright-browsers\n"
    "RUN python -m playwright install --with-deps "
)


class RecoveryCategory(StrEnum):
    TRANSIENT_TOOL = "TRANSIENT_TOOL"
    GENERATED_INPUT = "GENERATED_INPUT"
    ENVIRONMENT = "ENVIRONMENT"
    TERMINAL = "TERMINAL"


class RecoveryAction(StrEnum):
    RETRY_STAGE = "RETRY_STAGE"
    REBUILD_ENVIRONMENT = "REBUILD_ENVIRONMENT"
    REGENERATE_INPUT = "REGENERATE_INPUT"
    STOP = "STOP"


class RecoveryDecision(ContractModel):
    category: RecoveryCategory
    action: RecoveryAction
    diagnosis: str
    guidance: str
    environment_patch: str = ""


class RecoveryResolution(ContractModel):
    decision: RecoveryDecision
    decision_ref: StoredDataRef


class RecoveryCoordinator(Protocol):
    async def decide(
        self,
        checkpoint: StageCheckpoint,
        failure: StageFailure,
    ) -> RecoveryResolution: ...


TERMINAL_ERROR_CODES = frozenset(
    {
        "AUTH_REQUIRED",
        "AUTH_INVALID",
        "POLICY_DENIED",
        "CAPABILITY_DENIED",
        "RECOVERY_EXHAUSTED",
        "SIMPLE_RUNTIME_REFERENCE_SCOPE_MISMATCH",
    }
)

ALLOWED_ACTIONS = {
    RecoveryCategory.TRANSIENT_TOOL: frozenset(
        {RecoveryAction.RETRY_STAGE, RecoveryAction.STOP}
    ),
    RecoveryCategory.GENERATED_INPUT: frozenset(
        {RecoveryAction.REGENERATE_INPUT, RecoveryAction.STOP}
    ),
    RecoveryCategory.ENVIRONMENT: frozenset(
        {RecoveryAction.REBUILD_ENVIRONMENT, RecoveryAction.STOP}
    ),
    RecoveryCategory.TERMINAL: frozenset({RecoveryAction.STOP}),
}

_ALLOWED_PACKAGE_COMMAND_PREFIXES = (
    "python -m pip install ",
    "python3 -m pip install ",
    "pip install ",
    "pip3 install ",
    "apt-get update",
    "apt-get install ",
    "apk add ",
    "dnf install ",
    "yum install ",
    "npm ci",
    "npm install ",
    "pnpm install",
    "yarn install",
    "uv sync",
    "poetry install",
    "bundle install",
    "composer install",
    "cargo fetch",
    "go mod download",
)
_FORBIDDEN_PATCH_FRAGMENT = re.compile(
    r"(?i)(?:https?://|[a-z]:[\\/]|\\\\|/var/run/docker\.sock|"
    r"/run/docker\.sock|--mount|\$\(|`|&&|\|\||[;|<>])"
)

_DECISION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category": {
            "type": "string",
            "enum": [item.value for item in RecoveryCategory],
        },
        "action": {
            "type": "string",
            "enum": [item.value for item in RecoveryAction],
        },
        "diagnosis": {"type": "string"},
        "guidance": {"type": "string"},
        "environment_patch": {"type": "string"},
    },
    "required": [
        "category",
        "action",
        "diagnosis",
        "guidance",
        "environment_patch",
    ],
}


def validate_environment_patch(patch: str) -> str:
    """Accept bounded installer RUN lines or one fixed Playwright browser recipe."""

    normalized = "\n".join(line.strip() for line in patch.strip().splitlines())
    if not normalized or len(normalized.encode("utf-8")) > _MAX_ENVIRONMENT_PATCH_BYTES:
        raise ValueError("RECOVERY_ENVIRONMENT_PATCH_FORBIDDEN")
    if normalized in {
        _PLAYWRIGHT_PATCH_PREFIX + browser for browser in _PLAYWRIGHT_BROWSERS
    }:
        return normalized
    for line in normalized.splitlines():
        if not line.startswith("RUN ") or _FORBIDDEN_PATCH_FRAGMENT.search(line):
            raise ValueError("RECOVERY_ENVIRONMENT_PATCH_FORBIDDEN")
        command = line[4:].strip().lower()
        if not any(
            _allowed_package_command(command, item)
            for item in _ALLOWED_PACKAGE_COMMAND_PREFIXES
        ):
            raise ValueError("RECOVERY_ENVIRONMENT_PATCH_FORBIDDEN")
    return normalized


def _allowed_package_command(command: str, allowed: str) -> bool:
    base = allowed.rstrip()
    if command == base:
        return True
    return command.startswith(allowed if allowed.endswith(" ") else base + " ")


class SimpleRecoveryCoordinator:
    def __init__(
        self,
        *,
        client: SimpleLLMClient,
        artifacts: SimpleArtifactRepository,
    ) -> None:
        self._client = client
        self._artifacts = artifacts

    async def decide(
        self,
        checkpoint: StageCheckpoint,
        failure: StageFailure,
    ) -> RecoveryResolution:
        if checkpoint.identity != self._artifacts.identity:
            raise ValueError("RECOVERY_IDENTITY_SCOPE_MISMATCH")
        if not failure.retryable or failure.code in TERMINAL_ERROR_CODES:
            return self._store(
                checkpoint,
                failure,
                self._stop(
                    "failure is not eligible for automatic recovery",
                    "manual review is required",
                ),
            )

        if failure.code == "CURSOR_TIMED_OUT":
            # The provider already spent its complete call deadline, including
            # internal retries. Repeating an unchanged request automatically
            # can hold the run for many more minutes without new evidence.
            return self._store(
                checkpoint,
                failure,
                self._stop(
                    "Cursor did not complete within the configured call deadline",
                    "inspect the model call and retry manually after a change",
                ),
            )

        if failure.code == "DOCKER_DAEMON_UNAVAILABLE":
            return self._store(
                checkpoint,
                failure,
                self._stop(
                    "Docker daemon is unavailable",
                    "start Docker in the analysis environment, then retry manually",
                ),
            )

        if failure.code == "DOCKER_POC_MATERIALIZATION_FAILED":
            return self._store(
                checkpoint,
                failure,
                self._stop(
                    "Docker could not copy the PoC into the container",
                    "inspect the container state or Docker runtime, then resume",
                ),
            )

        if checkpoint.stage is SimpleStage.REPORT_DONE and failure.code in {
            "STAGE_UNEXPECTED_ERROR",
            "REPORT_UNSUPPORTED_METADATA_CLAIM",
            "REPORT_CODE_LOCATION_UNSUPPORTED",
            "REPORT_HIDDEN_REASONING_DENIED",
            "REPORT_LANGUAGE_REQUIRED",
        }:
            # A report content check used to escape as an unexpected exception.
            # Retry the reporter; the stage itself corrects the draft.
            return self._store(
                checkpoint,
                failure,
                RecoveryDecision(
                    category=RecoveryCategory.GENERATED_INPUT,
                    action=RecoveryAction.RETRY_STAGE,
                    diagnosis="Report draft failed a local content check",
                    guidance=(
                        "Retry the reporter without version, CVSS, severity, "
                        "disclosure, or unverified path citations"
                    ),
                ),
            )

        if failure.code in {"STAGE_CANCELLED", "STAGE_INTERRUPTED"}:
            # An operator stop or process interruption supplies no new
            # diagnostic evidence for a recovery LLM. Reopen the same stage;
            # the checkpoint attempt limit still bounds repeated restarts.
            return self._store(
                checkpoint,
                failure,
                RecoveryDecision(
                    category=RecoveryCategory.TRANSIENT_TOOL,
                    action=RecoveryAction.RETRY_STAGE,
                    diagnosis="Stage execution was interrupted",
                    guidance="Resume the interrupted stage from saved checkpoints",
                ),
            )

        if failure.code == "CURSOR_CLI_NOT_INSTALLED" and checkpoint.stage in {
            SimpleStage.STATIC_DONE,
            SimpleStage.HYPOTHESIS_DONE,
        }:
            # Construction of the recovery client already established that the
            # CLI is now present. Retry the exact stage without an LLM call.
            return self._store(
                checkpoint,
                failure,
                RecoveryDecision(
                    category=RecoveryCategory.TRANSIENT_TOOL,
                    action=RecoveryAction.RETRY_STAGE,
                    diagnosis="Cursor CLI is now available",
                    guidance="Retry the blocked bootstrap stage",
                ),
            )

        missing_browser = self._missing_playwright_browser(checkpoint, failure)
        if missing_browser is not None:
            return self._store(
                checkpoint,
                failure,
                RecoveryDecision(
                    category=RecoveryCategory.ENVIRONMENT,
                    action=RecoveryAction.REBUILD_ENVIRONMENT,
                    diagnosis="Python Playwright browser binary is absent",
                    guidance="Install the matching browser in the temporary image",
                    environment_patch=validate_environment_patch(
                        _PLAYWRIGHT_PATCH_PREFIX + missing_browser
                    ),
                ),
            )

        stderr = self._poc_execution_stderr(checkpoint, failure)
        if stderr is not None and (
            b"PermissionError" in stderr or b"Read-only file system" in stderr
        ):
            return self._store(
                checkpoint,
                failure,
                RecoveryDecision(
                    category=RecoveryCategory.GENERATED_INPUT,
                    action=RecoveryAction.REGENERATE_INPUT,
                    diagnosis="PoC runtime write was denied inside the container",
                    guidance=(
                        "Existing /workspace files are immutable. Let the application "
                        "create a new runtime directory beside them, put other scratch "
                        "data under /tmp, and do not chmod or edit source files"
                    ),
                ),
            )

        refs = tuple(dict.fromkeys(checkpoint.input_refs + failure.evidence_refs))
        prefix = b"\n".join(
            (
                b"Classify one failed SimpleRuntime stage and choose one "
                b"bounded action.",
                b"Never treat an execution error as a vulnerability FALSE verdict.",
                b"Do not request host changes, source edits, credentials, "
                b"or policy changes.",
                canonical_bytes(
                    {
                        "stage": checkpoint.stage.value,
                        "attempt": checkpoint.attempt_number,
                        "error_code": failure.code,
                        "retryable": failure.retryable,
                        "safe_message": failure.safe_message,
                    }
                ),
            )
        )
        context_limit = 16 * 1024
        while True:
            context = self._artifacts.prompt_context(refs, max_bytes=context_limit)
            prompt = prefix + b"\n" + context
            if len(prompt) <= _MAX_RECOVERY_PROMPT_BYTES:
                break
            if context_limit <= 1024:
                return self._store(
                    checkpoint,
                    failure,
                    self._stop(
                        "recovery context exceeds the CLI prompt limit",
                        "preserve the failure for manual review",
                    ),
                )
            context_limit //= 2
        try:
            response = await self._client.call(
                prompt=prompt,
                output_schema=_DECISION_SCHEMA,
                timeout_ms=_RECOVERY_TIMEOUT_MS,
                agent_name="recovery",
            )
        except Exception:
            response = StageFailure(
                code="RECOVERY_PROVIDER_FAILED",
                retryable=False,
                safe_message="Recovery provider did not return a decision",
            )
        if isinstance(response, StageFailure):
            decision = self._stop(
                "recovery provider did not return a decision",
                "preserve the failure for manual review",
            )
        else:
            try:
                decision = RecoveryDecision.model_validate_json(
                    canonical_bytes(response.value)
                )
                decision = self._validate_decision(decision)
            except (ValidationError, ValueError):
                decision = self._stop(
                    "recovery output failed policy validation",
                    "preserve the failure for manual review",
                )
        return self._store(checkpoint, failure, decision)

    def _missing_playwright_browser(
        self, checkpoint: StageCheckpoint, failure: StageFailure
    ) -> str | None:
        stderr = self._poc_execution_stderr(checkpoint, failure)
        if (
            stderr is None
            or b"BrowserType.launch: Executable doesn't exist at " not in stderr
        ):
            return None
        for browser in _PLAYWRIGHT_BROWSERS:
            if b"ms-playwright/" + browser.encode("ascii") in stderr:
                return browser
        return None

    def _poc_execution_stderr(
        self, checkpoint: StageCheckpoint, failure: StageFailure
    ) -> bytes | None:
        if (
            checkpoint.stage is not SimpleStage.POC_EXECUTION_DONE
            or failure.code != "POC_EXECUTION_FAILED"
        ):
            return None
        for ref in failure.evidence_refs:
            try:
                execution = json.loads(self._artifacts.read(ref))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(execution, dict) or execution.get("kind") != (
                "simple_poc_execution"
            ):
                continue
            if execution.get("attempt_id") != checkpoint.attempt_id:
                continue
            raw_stderr_ref = execution.get("stderr_ref")
            if raw_stderr_ref is None:
                continue
            try:
                stderr_ref = StoredDataRef.model_validate(raw_stderr_ref)
            except ValueError:
                continue
            if stderr_ref not in failure.evidence_refs:
                continue
            try:
                stderr = self._artifacts.read(stderr_ref)[-16_384:]
            except (OSError, ValueError):
                continue
            return stderr
        return None

    @staticmethod
    def _validate_decision(decision: RecoveryDecision) -> RecoveryDecision:
        if decision.action not in ALLOWED_ACTIONS[decision.category]:
            raise ValueError("RECOVERY_ACTION_CATEGORY_MISMATCH")
        if decision.action is RecoveryAction.REBUILD_ENVIRONMENT:
            return decision.model_copy(
                update={
                    "environment_patch": validate_environment_patch(
                        decision.environment_patch
                    )
                }
            )
        if decision.environment_patch.strip():
            raise ValueError("RECOVERY_ENVIRONMENT_PATCH_UNEXPECTED")
        return decision

    @staticmethod
    def _stop(diagnosis: str, guidance: str) -> RecoveryDecision:
        return RecoveryDecision(
            category=RecoveryCategory.TERMINAL,
            action=RecoveryAction.STOP,
            diagnosis=diagnosis,
            guidance=guidance,
            environment_patch="",
        )

    def _store(
        self,
        checkpoint: StageCheckpoint,
        failure: StageFailure,
        decision: RecoveryDecision,
    ) -> RecoveryResolution:
        decision_ref = self._artifacts.put_json(
            {
                "kind": "simple_recovery_decision",
                "identity": checkpoint.identity.model_dump(mode="json"),
                "stage": checkpoint.stage.value,
                "attempt": checkpoint.attempt_number,
                "attempt_id": checkpoint.attempt_id,
                "original_error": failure.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }
        )
        return RecoveryResolution(decision=decision, decision_ref=decision_ref)


__all__ = [
    "ALLOWED_ACTIONS",
    "MAX_RECOVERY_ATTEMPTS",
    "TERMINAL_ERROR_CODES",
    "RecoveryAction",
    "RecoveryCategory",
    "RecoveryCoordinator",
    "RecoveryDecision",
    "RecoveryResolution",
    "SimpleRecoveryCoordinator",
    "validate_environment_patch",
]
