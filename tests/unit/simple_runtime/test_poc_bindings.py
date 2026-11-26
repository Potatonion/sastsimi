from __future__ import annotations

import pytest

from sastsimi.simple_runtime.poc import PoCCandidateRejected, validate_candidate


def test_posix_for_loop_variable_is_declared() -> None:
    script = (
        b"#!/bin/sh\nset -eu\n"
        b"for candidate in /usr/bin/python3 /usr/local/bin/python3; do\n"
        b'  if [ -x "$candidate" ]; then printf "%s\\n" "$candidate"; fi\n'
        b"done\n"
    )

    assert validate_candidate(script, allowed_environment_names=frozenset())


def test_posix_read_variable_is_declared() -> None:
    script = (
        b"#!/bin/sh\nset -eu\n"
        b"while IFS= read -r candidate; do\n"
        b'  printf "%s\\n" "$candidate"\n'
        b"done < /dev/null\n"
    )

    assert validate_candidate(script, allowed_environment_names=frozenset())


def test_nested_loop_bindings_match_generated_poc_shape() -> None:
    script = (
        b"#!/bin/sh\nset -eu\n"
        b"for directory in /opt /usr/local; do\n"
        b"  for candidate in /usr/bin/python3; do\n"
        b'    printf "%s %s\\n" "$directory" "$candidate"\n'
        b"  done\n"
        b"done\n"
    )

    assert validate_candidate(script, allowed_environment_names=frozenset())


def test_loop_binding_does_not_allow_unrelated_environment_input() -> None:
    script = (
        b"#!/bin/sh\nset -eu\n"
        b"for candidate in /usr/bin/python3; do\n"
        b'  printf "%s %s\\n" "$candidate" "$UNDECLARED_INPUT"\n'
        b"done\n"
    )

    with pytest.raises(PoCCandidateRejected, match="POC_UNDECLARED_INPUT"):
        validate_candidate(script, allowed_environment_names=frozenset())


def test_commented_loop_does_not_declare_input() -> None:
    script = (
        b"#!/bin/sh\nset -eu\n"
        b"# for candidate in /usr/bin/python3; do\n"
        b'printf "%s\\n" "$candidate"\n'
    )

    with pytest.raises(PoCCandidateRejected, match="POC_UNDECLARED_INPUT"):
        validate_candidate(script, allowed_environment_names=frozenset())
