"""CPU-runnable tests: independent finite differences of weighted features.

Load the algebra file directly so importing the GPU package/driver is not
required. These tests do not establish GPU execution or performance.
"""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('ao_reduce', Path(__file__).resolve().parents[1] / 'nttda_ao_reduce.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def energy(features, densities, weights, grid_weights, pair_densities, special):
    # Explicit left/right AO features; no derivative-workspace formulas.
    pairs = np.einsum('nuv,fug,hvg->nfhg', densities, features, features)
    rho = np.concatenate((pairs[:, 0, 0][:, None], pairs[:, 1:4, 0] + pairs[:, 0, 1:4]), axis=1)
    if weights.shape[1] == 5:
        tau = 0.5 * sum(pairs[:, f, f] for f in range(1, 4))
        rho = np.concatenate((rho, tau[:, None]), axis=1)
    value = np.einsum('nfg,nfg,g->', rho, weights, grid_weights)
    if len(pair_densities):
        pair = np.einsum('nuv,fug,hvg->nfhg', pair_densities, features, features)
        value += np.einsum('nfhg,nfhg,g->', pair, special, grid_weights)
    return value


class AOReductionTests(unittest.TestCase):
    def case(self, features=4, hermitian=False, with_pair=True, budget=2**20):
        rng = np.random.default_rng(734)
        ao = rng.normal(size=(10, 7, 13))
        d = rng.normal(size=(3, 7, 7))
        if hermitian:
            d = (d + d.transpose(0, 2, 1)) / 2
        p = rng.normal(size=(2 if with_pair else 0, 7, 7))
        w = rng.normal(size=(3, features, 13))
        special = rng.normal(size=(len(p), 4, 4, 13))
        gw = rng.normal(size=13)
        atoms = np.array([2, 0, 2, 1, 0, 1, 2])
        requested = [2, 4, 0, 2]  # unordered, absent center, repeated selection
        c = np.einsum('nba,fbg->nfag', d, ao[:4])
        ct = np.einsum('nab,fbg->nfag', d, ao[:4])
        pc = np.einsum('nba,fbg->nfag', p, ao[:4])
        pt = np.einsum('nab,fbg->nfag', p, ao[:4])
        stats = {}
        actual = MODULE.contract_centers(
            ao,
            c,
            ct,
            w,
            atoms,
            requested,
            grid_weights=gw,
            pair=(pc, pt, special),
            max_memory_bytes=budget,
            xp=np,
            stats=stats,
        )
        expected = np.zeros_like(actual)
        second = [[4, 5, 6], [5, 7, 8], [6, 8, 9]]
        step = 1e-5
        for i, atom in enumerate(requested):
            for xyz in range(3):
                delta = np.zeros_like(ao[:4])
                mask = atoms == atom
                delta[0, mask] = -ao[xyz + 1, mask]
                for f in range(3):
                    delta[f + 1, mask] = -ao[second[xyz][f], mask]
                expected[i, xyz] = (
                    energy(ao[:4] + step * delta, d, w, gw, p, special)
                    - energy(ao[:4] - step * delta, d, w, gw, p, special)
                ) / (2 * step)
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-8)
        self.assertLessEqual(stats['estimated_tile_bytes'], budget)
        return actual, stats

    def test_gga_nonhermitian_pair(self):
        self.case()

    def test_mgga_nonhermitian_pair(self):
        self.case(features=5)

    def test_hermitian_no_pair(self):
        self.case(features=5, hermitian=True, with_pair=False)

    def test_small_tiles_preserve_result(self):
        full, _ = self.case()
        tiled, stats = self.case(budget=4096)
        np.testing.assert_allclose(full, tiled, atol=1e-10, rtol=1e-12)
        self.assertGreater(stats['tiles'], 1)

    def test_fockz_weights_match_four_original_contractions(self):
        rng = np.random.default_rng(57)
        for nf in (4, 5):
            f = rng.normal(size=(nf, nf, 7))
            k = rng.normal(size=(nf, nf, nf, 7))
            kb = rng.normal(size=k.shape)
            pz = rng.normal(size=(3, nf, 7))
            op = rng.normal(size=(nf, 7))
            gw = rng.normal(size=7)
            dr = rng.normal(size=(6, nf, 7))
            pair = 0.5 * np.einsum('nxg,yg->nxyg', pz, op)
            aw = np.einsum('nxyg,xyzg->nzg', pair, k) * gw
            bw = np.einsum('nxyg,xyzg->nzg', pair, kb) * gw
            expected = 0.5 * np.einsum('nxg,xyg,yg,g->n', dr[:3], f, op, gw)
            expected += 0.5 * np.einsum('nxg,xyg,yg,g->n', pz, f, dr[3], gw)
            expected += np.einsum('nxyg,xyzg,zg,g->n', pair, k, dr[4], gw)
            expected += np.einsum('nxyg,xyzg,zg,g->n', pair, kb, dr[5], gw)
            for t in range(3):
                weights = MODULE.fockz_weights(f, op, pz, aw, bw, gw, t, xp=np)
                actual = np.einsum('nfg,nfg->', weights, dr[[t, 3, 4, 5]])
                self.assertAlmostEqual(actual, expected[t], places=10)

    def test_postz_weights_match_original_contractions(self):
        rng = np.random.default_rng(71)
        for nf in (4, 5):
            f = rng.normal(size=(2, nf, 2, nf, 9))
            v = rng.normal(size=(2, nf, 9))
            probe = rng.normal(size=(3, 2, nf, 9))
            dr = rng.normal(size=(8, nf, 9))
            gw = rng.normal(size=9)
            expected = np.einsum('nsxg,sxg,g->n', dr[2:].reshape(3, 2, nf, 9), v, gw)
            response = np.einsum('axg,axbyg,g->byg', dr[:2], f, gw)
            expected += np.einsum('nbyg,byg->n', probe, response)
            for t in range(3):
                weights = MODULE.postz_weights(v, f, probe, t, xp=np)
                actual = np.einsum('nfg,nfg,g->', weights, dr[[0, 1, 2 + 2 * t, 3 + 2 * t]], gw)
                self.assertAlmostEqual(actual, expected[t], places=10)

    def test_budget_too_small_and_empty_selection(self):
        with self.assertRaises(ValueError):
            MODULE.plan_tiles(10, 20, 3, 2, 4, 1)
        ao = np.ones((10, 2, 3))
        c = np.ones((1, 4, 2, 3))
        w = np.ones((1, 4, 3))
        result = MODULE.contract_centers(ao, c, c, w, np.array([0, 1]), [], xp=np)
        self.assertEqual(result.shape, (0, 3))

    def test_density_selection_and_float32_rejection(self):
        rng = np.random.default_rng(20)
        ao = rng.normal(size=(10, 4, 5))
        c = rng.normal(size=(6, 4, 4, 5))
        ct = rng.normal(size=c.shape)
        w = rng.normal(size=(2, 5, 5))
        atom = np.array([0, 1, 0, 2])
        rows = [4, 1]
        actual = MODULE.contract_centers(ao, c, ct, w, atom, [0, 1, 2], xp=np, density_indices=rows)
        expected = MODULE.contract_centers(ao, c[rows], ct[rows], w, atom, [0, 1, 2], xp=np)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        with self.assertRaises(TypeError):
            MODULE.contract_centers(ao.astype('float32'), c, ct, w, atom, [0], xp=np, density_indices=rows)


if __name__ == '__main__':
    unittest.main()
