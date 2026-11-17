"""Tests for process-pool helpers."""

import os

import pytest

from dimspin.Parallel import (
    DEFAULT_MAX_WORKERS,
    check_worker_count,
    parallel_map,
    resolve_workers,
    set_blas_threads_one,
)


def _square(x: int) -> int:
    return x * x


def test_default_and_explicit_worker_counts() -> None:
    assert resolve_workers(None) == DEFAULT_MAX_WORKERS
    assert resolve_workers(2) == 2


@pytest.mark.parametrize("value", [0, -1, "not-an-int"])
def test_invalid_worker_counts_raise(value) -> None:
    with pytest.raises(ValueError):
        check_worker_count(value)


def test_parallel_map_preserves_order() -> None:
    assert list(parallel_map(_square, [1, 2, 3], max_workers=2)) == [1, 4, 9]


def test_set_blas_threads_one() -> None:
    set_blas_threads_one()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
