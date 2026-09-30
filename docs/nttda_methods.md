# GPU NTTDA gradients and nonadiabatic couplings

## Installation and entry points

Install this checkout with its normal GPU4PySCF dependencies and standard PySCF.
The GPU solver, gradients, NAC and FSSH do not import the customized pyscf-forge
checkout. `NTTDA_FORGE_PATH` is no longer used. CPU/GPU numerical comparison tests
add forge as a test-only dependency. Explicit `EnsembleRKS.to_cpu()` and
`EnsembleROKS.to_cpu()` still require forge because standard PySCF has no matching
CPU classes; ordinary GPU calculations never call these conversions.

```python
from pyscf import gto
from gpu4pyscf.sftda import EnsembleROKS, NTTDA_EnsembleROKS
from gpu4pyscf.grad.nttda import compute_frame, make_frame_cache

mol = gto.M(atom='C 0 0 0; H 0 .8 .6; H 0 -.9 .5', basis='sto-3g', spin=2)
mf = EnsembleROKS(mol, xc='PBE').density_fit().run()
td = NTTDA_EnsembleROKS(mf).set(deltaS=-1, nstates=3).run()
grad = td.Gradients().kernel(state=1)
nac = td.NAC().kernel(state_I=1, state_J=2, ediff=True, use_etfs=False)
frame = compute_frame(td, active_state=1, nac_pairs=[(1, 2)],
                      frame_cache=make_frame_cache())
```

`NTTDA_ROKS`, `NTTDA_ROKS_NoBeta`, `NTTDA_EnsembleRKS` and
`NTTDA_EnsembleROKS` remain available. Legacy `NTTDA(mf).set(nobeta=...)` remains
supported. States are one-based; gradient state 0 selects the reference.
Gradients support `deltaS=-1,0`; NAC and joint frames support `deltaS=-1`.
With `gap = omega_J - omega_I`, full NAC is `N_HF/gap + d_CSF`.
`use_etfs=True` omits `d_CSF`; `ediff=False` returns the energy-scaled result.

## Code map and ownership

| Owner | Responsibility |
|---|---|
| `sftda/nttda_methods.py` | Four immutable physical identities and ordinary function dispatch |
| `grad/nttda.py` | Static GPU `Gradients` class, validation and public results |
| `nac/nttda.py` | Interstate numerator, moving-CSF term and overlap reference |
| `grad/_nttda/delta_s_minus_one.py`, `delta_s_zero.py` | Channel algebra and explicit prepared derivative data |
| `grad/_nttda/orbital.py` | ROKS and ensemble orbital Hessians, spin weights and adjoints |
| `grad/_nttda/response.py` | One evaluation cache, strict GMRES and final contraction |
| `grad/_nttda/frame.py` | Joint-frame scheduling, reference fusion and cross-frame initial guesses |
| `grad/_nttda/derivative_jk.py`, `grad/nttda_ledger.py` | Derivative terms and GPU conventional/DF integral evaluation |
| `grad/nttda_xc.py`, `nttda_ao_reduce.py`, `nttda_xc_fused.py` | GPU quadrature and fused AO-center contractions |
| `grad/_nttda/xc.py`, `xc_host.py` | Explicit XC transfer boundary and locally owned fallback quadrature |

The former forge loader, bytecode checks, path changes, CPU twins and dynamically
created gradient subclass have been removed. Production AO/MO arrays stay on the
GPU. Public gradient/NAC results are NumPy arrays. Small SciPy GMRES vectors and
AWF combinatorics run on the host. LDA and NoBeta MGGA retain bounded host XC
quadrature using standard PySCF NumInt; standalone post-Z contractions can also
use this local quadrature. No host fallback creates a CPU SCF or NTTDA solver.

A prepared derivative contains its M matrix, direct terms, probes and J/K ledger.
It contains no function callbacks. ROKS and ensemble Hessians retain their distinct
spin/occupation factors. The selected EnsembleROKS reference contribution is
nonstationary; `compute_frame` folds its RHS shift into the state-gradient solve
and retains the stricter `min(cphf_conv_tol, 1e-12)` target.

One response cache belongs to one electronic solution. Joint-frame drivers share
that cache. Replacing orbitals/amplitudes or changing coordinates, method, channel,
XC or DF settings requires a new derivative driver. Treat arrays and grids as
immutable during an evaluation; in-place array edits are not content-hashed.
Only AO-projected adjoint guesses survive across geometries, and only a fully
successful frame updates them. GMRES acceptance uses the original equation's
absolute residual 2-norm, without a tolerance floor or extra silent iterations.

## Optimization policy

The default direct XC path is `ao_reduce`; the default DF output path is
`slot_aware`, including grouping exact factors within the same output slot.
Fused XC, exact DF exchange factors, cached Fock data, shared selected-reference
response, and direct MO Davidson XC are retained. Non-slot `rank_batched`
compressed construction and `NTTDA_DF_COMPRESSED_BACKEND` were removed.

`NTTDA_XC_DIRECT_BACKEND=legacy` and `NTTDA_DF_OUTPUT_BACKEND=legacy` remain
numerical comparison paths. Memory/tile controls and opt-in profiling remain in
`grad/nttda_params.py`. Removing an experiment does not remove exact fallback
paths required by unsupported shapes, complex factors or non-DF calculations.

Historical performance evidence is in the companion project's
`Gpu4pyscf/docs/nttda_performance_and_validation.md` and
`nttda_optimization_evolution_report.md`. Local numerical validation is not an
A100 performance measurement; see [this refactor's validation](nttda_refactor_validation.md).
