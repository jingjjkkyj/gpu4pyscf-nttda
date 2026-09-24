"""Adaptive runtime parameters for NTTDA gradient/NAC GPU driver.

All parameters default to values calibrated on RTX 4060 Laptop (8 GB VRAM).
A100-specific tuning requires a discrete sweep on the actual hardware; until
then every A100 path is marked ``unvalidated_on_A100``.

Environment variables
---------------------

All variables are optional and use the ``NTTDA_`` prefix.

    NTTDA_DF_MEM_FRACTION
        Fraction of free VRAM available to DF derivative kernels.
        Default: ``0.5``.  Range: ``(0, 1)``.

    NTTDA_DF_BATCH_FACTOR
        Multiplier for DF aux basis batch size (``batch_size``).
        Default: ``1.0``.  Range: ``(0, 10]``.  Increase to use larger
        auxiliary-basis batches when VRAM permits.

    NTTDA_DF_BLK_FACTOR
        Multiplier for DF block size (``blksize``).
        Default: ``1.0``.  Range: ``(0, 10]``.

    NTTDA_DM_BLOCK
        Number of density matrices processed per CUDA kernel launch in
        ``ejk_int3c2e_ip1``.  Default: ``7``.  Range: ``[1, 16]``.
        Larger values reduce Python launch overhead but increase register
        pressure and shared-memory usage in the CUDA kernel.

    NTTDA_DF_COMPRESSED_BACKEND
        ``legacy`` (default) or experimental ``rank_batched``. The latter
        batches J-only tasks and exact rank-(2,2) K tasks while building the
        compressed DF derivative tensor. Unsupported tasks fall back to the
        legacy per-pair path.

    NTTDA_DF_COMPRESSED_PROFILE
        ``1`` records CUDA-event timings grouped by ``(J/K, left rank,
        right rank)`` for the compressed DF derivative build and the final
        derivative kernel. Default ``0`` creates no events.

    NTTDA_DF_OUTPUT_BACKEND
        ``legacy`` (default) returns one derivative row per input task.
        Experimental ``slot_aware`` pre-sums already weighted compressed
        tensors within the same ``(omega, output slot)`` group and runs the
        three- and two-center derivative kernels over the reduced output
        width. It also groups same-slot pure-K rank-(2,2) contractions
        automatically; no ``NTTDA_DF_COMPRESSED_BACKEND`` override is needed.

    NTTDA_CPHF_MAX_CYCLE
        Default maximum CPHF/Z-vector iterations.  ``None`` (use forge
        default).  Range: ``[1, 1000]`` or unset.

    NTTDA_DAVIDSON_MAX_SUBSPACE
        Upper bound on the Davidson initial subspace width after warm-start
        concatenation and linear-dependency pruning.  Default: ``12``.
        Range: ``[2, 50]``.

    NTTDA_CONTRACT_BACKEND
        Tensor contraction backend for GGA/MGGA response.  ``'cupy'``
        (default) or ``'cutensor'``.  cuTENSOR requires
        ``libcublasLt.so.12`` and ``libcutensor.so.2``; if unavailable the
        driver falls back to cupy with a warning.

    NTTDA_XC_DIRECT_BACKEND
        ``legacy`` (default) or experimental ``ao_reduce``. The latter
        batches AO-center derivatives and still requires A100 acceptance.

    NTTDA_XC_DIRECT_MAX_MEMORY_MB
        Additional AO-reduction tile budget in MiB, default 256. Does not
        include resident inputs, final outputs, or library scratch space.

    NTTDA_XC_DIRECT_PROFILE
        ``1`` enables synchronized direct-region wall/event timing.
        Default ``0`` adds no synchronization. Use only diagnostic runs.

All values are read once at import time and cached in :data:`PARAMS`.
Tests may override via :func:`reload_params`.
"""

import os

# Sentinel marking parameters that have not been validated on A100.
# A100 sweep must be done before setting non-default values on that hardware.
UNVALIDATED_ON_A100 = True


def _get_float(name, default, lo, hi):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a float in ({lo}, {hi}], got {raw!r}') from exc
    if not (lo < val <= hi):
        raise ValueError(f'{name} must be in ({lo}, {hi}], got {val}')
    return val


def _get_int(name, default, lo, hi):
    raw = os.environ.get(name)
    if raw is None or raw == '':
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be an int in [{lo}, {hi}], got {raw!r}') from exc
    if not (lo <= val <= hi):
        raise ValueError(f'{name} must be in [{lo}, {hi}], got {val}')
    return val


def _get_str(name, default, choices):
    raw = os.environ.get(name, default)
    if raw not in choices:
        raise ValueError(f'{name} must be one of {choices}, got {raw!r}')
    return raw


def _load_params():
    return {
        'df_mem_fraction': _get_float('NTTDA_DF_MEM_FRACTION', 0.5, 0.0, 1.0),
        'df_batch_factor': _get_float('NTTDA_DF_BATCH_FACTOR', 1.0, 0.0, 10.0),
        'df_blk_factor': _get_float('NTTDA_DF_BLK_FACTOR', 1.0, 0.0, 10.0),
        'dm_block': _get_int('NTTDA_DM_BLOCK', 7, 1, 16),
        'df_compressed_backend': _get_str(
            'NTTDA_DF_COMPRESSED_BACKEND',
            'legacy',
            ('legacy', 'rank_batched'),
        ),
        'df_compressed_profile': _get_str(
            'NTTDA_DF_COMPRESSED_PROFILE', '0', ('0', '1'),
        ) == '1',
        'df_output_backend': _get_str(
            'NTTDA_DF_OUTPUT_BACKEND',
            'legacy',
            ('legacy', 'slot_aware'),
        ),
        'cphf_max_cycle': (
            _get_int('NTTDA_CPHF_MAX_CYCLE', 0, 1, 1000)
            if os.environ.get('NTTDA_CPHF_MAX_CYCLE')
            else None
        ),
        'davidson_max_subspace': _get_int(
            'NTTDA_DAVIDSON_MAX_SUBSPACE', 12, 2, 50,
        ),
        'contract_backend': _get_str(
            'NTTDA_CONTRACT_BACKEND', 'cupy', ('cupy', 'cutensor'),
        ),
        'xc_direct_backend': _get_str(
            'NTTDA_XC_DIRECT_BACKEND', 'legacy', ('legacy', 'ao_reduce'),
        ),
        'xc_direct_max_memory_mb': _get_int(
            'NTTDA_XC_DIRECT_MAX_MEMORY_MB', 256, 1, 1048576,
        ),
        'xc_direct_profile': _get_str(
            'NTTDA_XC_DIRECT_PROFILE', '0', ('0', '1'),
        ) == '1',
        'finish_profile': _get_str(
            'NTTDA_FINISH_PROFILE', '0', ('0', '1'),
        ) == '1',
        'unvalidated_on_A100': UNVALIDATED_ON_A100,
    }


PARAMS = _load_params()


def reload_params():
    """Re-read environment variables.

    Useful in tests that patch ``os.environ`` after import time.
    Also useful for A100: call after setting env vars to activate
    swept parameter values without restarting the Python process.
    """
    global PARAMS
    PARAMS = _load_params()
    return PARAMS


def get(key, default=None):
    return PARAMS.get(key, default)
