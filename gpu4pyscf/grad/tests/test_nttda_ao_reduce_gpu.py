"""Real CuPy execution gate; run explicitly on an allocated CUDA device."""

import unittest

import cupy as cp
import numpy as np

from gpu4pyscf.grad.nttda_ao_reduce import contract_centers
from gpu4pyscf.grad.nttda_xc import (
    _ao_center_derivative,
    _density_workspace,
    _pair_feature_batches,
    _contract_pair_feature_derivatives,
)


class GPUCenterReductionTests(unittest.TestCase):
    def test_original_gpu_loop_matches_tiled_reduction(self):
        rng = np.random.default_rng(734)
        for xctype, nf in (('GGA', 4), ('MGGA', 5)):
            for hermitian in (False, True):
                ao = cp.asarray(rng.normal(size=(10, 7, 13)))
                d = cp.asarray(rng.normal(size=(3, 7, 7)))
                if hermitian:
                    d = 0.5 * (d + d.transpose(0, 2, 1))
                workspace = _density_workspace(ao, d, hermitian=hermitian, xctype=xctype)
                w = cp.asarray(rng.normal(size=(3, nf, 13)))
                gw = cp.asarray(rng.normal(size=13))
                atom = cp.asarray([2, 0, 2, 1, 0, 1, 2])
                requested = [2, 4, 0, 2]
                for npair in (0, 2):
                    p = cp.asarray(rng.normal(size=(npair, 7, 7)))
                    _, pc, pt = _pair_feature_batches(ao, p)
                    pw = cp.asarray(rng.normal(size=(npair, 4, 4, 13)))
                    expected = cp.zeros((len(requested), 3))
                    for a, center in enumerate(requested):
                        local = cp.flatnonzero(atom == center)
                        if not len(local):
                            continue
                        for xyz in range(3):
                            delta = _ao_center_derivative(ao, local, xyz)
                            dr = workspace.derivatives(delta, local)
                            expected[a, xyz] = cp.einsum('nfg,nfg,g->', w, dr, gw)
                            expected[a, xyz] += _contract_pair_feature_derivatives(delta, pc, pt, local, pw, gw)
                    actual = contract_centers(
                        ao,
                        workspace.contracted,
                        workspace.contracted_transpose,
                        w,
                        atom,
                        requested,
                        xp=cp,
                        pair=(pc, pt, pw),
                        grid_weights=gw,
                        max_memory_bytes=4096,
                    )
                    np.testing.assert_allclose(cp.asnumpy(actual), cp.asnumpy(expected), atol=1e-10, rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
