# GPU NTTDA method interfaces

The GPU constructors `NTTDA_ROKS`, `NTTDA_ROKS_NoBeta`, `NTTDA_EnsembleRKS`, and
`NTTDA_EnsembleROKS` are exported by `gpu4pyscf.sftda`. They use the method
definitions in the matching CPU forge checkout; set `NTTDA_FORGE_PATH` to that
checkout. Legacy `NTTDA(mf).set(nobeta=...)` calls remain supported.

See the companion CPU checkout's `docs/nttda_methods.md` for method semantics,
interface compatibility and the shared-layer contract. Ensemble `nobeta`
does not change either the selected method or its execution policy.

The [validation record](../../pyscf-forge-nttda-opt/docs/nttda_methods_validation.md)
documents the CPU and GPU regression groups and the original/new comparison.

```python
from pyscf import gto
from gpu4pyscf.sftda import EnsembleROKS, NTTDA_EnsembleROKS
from gpu4pyscf.grad.nttda import compute_frame, make_frame_cache

mol = gto.M(atom='C 0 0 0; H 0 .8 .6; H 0 -.9 .5', basis='sto-3g', spin=2)
mf = EnsembleROKS(mol, xc='PBE').density_fit().run()
td = NTTDA_EnsembleROKS(mf).set(deltaS=-1, nstates=3).run()
guesses = make_frame_cache()
frame = compute_frame(td, active_state=1, nac_pairs=[(1, 2)], frame_cache=guesses)
```

`compute_frame` retains its return structure, state numbering and ETF/gap
conventions. Its statistics include `method_id`. All tasks in a frame use one
CPU twin and the same response and ledger backends. Independent gradient or NAC
calls get fresh response state. Create a new derivative driver after changing
the source electronic solution.

The GPU modules separate these responsibilities:

- `grad/nttda.py`: public gradient driver and joint-frame orchestration.
- `grad/nttda_bridge.py`: forge loading/verification, CPU twins, reference
  reconstruction and J/K routing.
- `grad/nttda_context.py`: fixed-evaluation ownership and GPU response factories.
- `grad/nttda_ledger.py`: exact/DF derivative integrals and DF compression.
- `grad/nttda_xc.py`: GGA/MGGA contractions with the existing AO residency policy.
- `grad/nttda_ao_reduce.py`: independent tiled AO-center contractions.
- `df/grad/ensemble_roks.py`: the selected-reference fractional-occupation DF
  Fock skeleton, including auxiliary-basis response.

Existing backend choices, SVD thresholds and DF auxiliary response are retained.
ROKS NoBeta MGGA still uses the CPU XC path. The GPU numerical tests require an
accessible CUDA device, including the four-method matrix in
`gpu4pyscf/grad/tests/test_nttda_methods.py`. A sandbox without device access is
not evidence that the host CUDA installation is unavailable.
