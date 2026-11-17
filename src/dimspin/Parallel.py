"""
Achieve parallel computing to increase processing speed.
On Windows every worker re-imports numpy/scipy, so the default is moderate.
Users can override it with Spec.max_workers.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor

DEFAULT_MAX_WORKERS = min(os.cpu_count() or 4, 8)
def _total_ram_mb():
    """Return total physical memory in MB, or None if it cannot be read."""
    try:
        if os.name == 'nt':
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys // (1024 * 1024))
        else:
            return int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // (1024 * 1024))
    except Exception:
        return None


def check_worker_count(n):
    """Validate a worker count.

    Raise ``ValueError`` for non-positive or non-integer values and emit a
    warning when the count exceeds the logical core count or the estimated
    memory use.
    """
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise ValueError(f'worker count must be an integer, got {n!r}')
    if n <= 0:
        raise ValueError(f'worker count must be a positive integer, got {n}')

    logical = os.cpu_count() or 1
    if n > logical:
        logging.warning(f'worker count {n} exceeds the {logical} logical cores; oversubscription may be slower.')

    # Rough estimate: each worker re-imports numpy/scipy, about 250 MB.
    total_mb = _total_ram_mb()
    if total_mb is not None and n * 250 > total_mb * 0.8:
        logging.warning(
            f'worker count {n} is estimated to need about {n * 250} MB, '
            f'close to 80% of total memory ({total_mb} MB); paging may occur.'
        )
    return n


def resolve_workers(spec_max):
    """Return the default worker count when ``spec_max`` is None."""
    if spec_max is None:
        return DEFAULT_MAX_WORKERS
    return check_worker_count(spec_max)


def set_blas_threads_one():
    # Limit BLAS/OpenMP threads so workers do not compete for cores.
    # Set them in the parent process so spawned workers inherit them.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def parallel_map(func, iterable, initializer=None, initargs=None, max_workers=None):
    """Process-pool map as a lazy generator.

    The BLAS thread limit is applied inside workers, and results are produced
    in input order. ``max_workers=None`` uses ``DEFAULT_MAX_WORKERS``;
    otherwise the value is validated with :func:`check_worker_count`.
    """
    set_blas_threads_one()
    n = resolve_workers(max_workers)
    with ProcessPoolExecutor(max_workers=n, initializer=initializer, initargs=initargs) as pool:
        yield from pool.map(func, iterable)
