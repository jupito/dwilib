"""Functionality related to tasking and parallelization."""

# https://pythonhosted.org/joblib/memory.html
# https://pythonhosted.org/joblib/parallel.html

# TODO: Configuration.

from functools import partial

import joblib

import dwi

_memory_defaults = dict(
    # cachedir='cache',
    cachedir=dwi.rcParams.cachedir,
    # mmap_mode=None,
    # compress=False,
    verbose=0,
    # bytes_limit=None,
    )
_parallel_defaults = dict(
    n_jobs=-2,
    # backend=None,
    verbose=5,
    # timeout=None,
    # pre_dispatch='2 * n_jobs',
    # batch_size='auto',
    # temp_folder=None,
    # max_nbytes='1M',
    # mmap_mode='r',
    )

Memory = partial(joblib.Memory, **_memory_defaults)

Parallel = partial(joblib.Parallel, **_parallel_defaults)
delayed = partial(joblib.delayed)

dump = partial(joblib.dump)
load = partial(joblib.load)
