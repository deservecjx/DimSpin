"""Stage 0 smoke tests for the package skeleton."""

import re

import dimspin


def test_version_is_exposed() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", dimspin.__version__)


def test_public_api_is_empty_in_stage_0() -> None:
    assert dimspin.__all__ == []
