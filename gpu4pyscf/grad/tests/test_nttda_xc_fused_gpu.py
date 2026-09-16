"""Numerical checks for compiled NTTDA XC response contractions."""

from dataclasses import dataclass
import unittest

import cupy as cp
import numpy as np

from gpu4pyscf.grad.nttda_ao_reduce import contract_centers
from gpu4pyscf.grad.nttda_xc import (
    GPUXCFrameBackend,
    _gga_pair_kernel_cross,
    _gga_pair_potential,
    _mgga_pair_kernel_cross,
    _mgga_pair_potential,
)
from gpu4pyscf.grad.nttda_xc_fused import (
    add_response_matrices,
    assemble_response_weights,
    contract_response_centers,
)


@dataclass(frozen=True)
class _Term:
    target: int
    source: int
    vref0: float
    vref1: float


class FusedXCResponseTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9274)

    def _random(self, shape):
        return cp.asarray(self.rng.normal(size=shape))

    def _weights_reference(
            self, xctype, fref, ka, kb, rho, pairs, grid_weights, terms):
        labels, features, ngrids = rho.shape
        ordinary = cp.zeros((labels, features, ngrids))
        special = cp.zeros((labels, 4, 4, ngrids))
        reference_alpha = cp.zeros((features, ngrids))
        reference_beta = cp.zeros_like(reference_alpha)
        potential = (
            _gga_pair_potential
            if xctype == "GGA" else _mgga_pair_potential
        )
        cross = (
            _gga_pair_kernel_cross
            if xctype == "GGA" else _mgga_pair_kernel_cross
        )
        pair_potentials = [potential(fref, pair) for pair in pairs]
        for term in terms:
            if term.vref0:
                ordinary[term.target] += term.vref0 * cp.einsum(
                    "xyg,yg->xg", fref, rho[term.source],
                )
                ordinary[term.source] += term.vref0 * cp.einsum(
                    "xyg,xg->yg", fref, rho[term.target],
                )
                product = term.vref0 * cp.einsum(
                    "xg,yg->xyg", rho[term.target], rho[term.source],
                )
                reference_alpha += cp.einsum(
                    "xyg,xyzg->zg", product, ka,
                )
                reference_beta += cp.einsum(
                    "xyg,xyzg->zg", product, kb,
                )
            if term.vref1:
                special[term.target] += (
                    term.vref1 * pair_potentials[term.source]
                )
                special[term.source] += (
                    term.vref1 * pair_potentials[term.target]
                )
                product = term.vref1 * cross(
                    pairs[term.target], pairs[term.source],
                )
                reference_alpha += cp.einsum(
                    "xyg,xyzg->zg", product, ka,
                )
                reference_beta += cp.einsum(
                    "xyg,xyzg->zg", product, kb,
                )
        return tuple(
            value * grid_weights
            for value in (
                ordinary, special, reference_alpha, reference_beta,
            )
        )

    def test_fused_weight_assembly_matches_formula_layer(self):
        terms = (
            _Term(0, 0, 1.0, -0.2),
            _Term(0, 1, 0.35, 0.0),
            _Term(1, 2, -0.1, 0.7),
            _Term(2, 0, 0.0, -0.4),
        )
        for xctype, features in (("GGA", 4), ("MGGA", 5)):
            ngrids = 23
            fref = self._random((features, features, ngrids))
            ka = self._random((features, features, features, ngrids))
            kb = self._random((features, features, features, ngrids))
            rho = self._random((3, features, ngrids))
            pairs = self._random((3, 4, 4, ngrids))
            grid_weights = self._random((ngrids,))
            expected = self._weights_reference(
                xctype, fref, ka, kb, rho, pairs, grid_weights, terms,
            )
            actual = assemble_response_weights(
                fref, ka, kb, rho, pairs, grid_weights,
                [term.target for term in terms],
                [term.source for term in terms],
                [0, 1, 2],
                [term.vref0 for term in terms],
                [term.vref1 for term in terms],
            )
            for reference, fused in zip(expected, actual):
                cp.testing.assert_allclose(
                    fused, reference, rtol=2e-13, atol=2e-12,
                )

    def test_batched_ao_matrices_match_existing_contractions(self):
        for xctype, features in (("GGA", 4), ("MGGA", 5)):
            ao = self._random((10, 6, 19))
            ordinary = self._random((3, features, 19))
            special = self._random((2, 4, 4, 19))
            reference_alpha = self._random((features, 19))
            reference_beta = self._random((features, 19))
            indices = cp.asarray([6, 1, 4, 2, 8, 0])
            expected = [cp.zeros((9, 9)) for _ in range(5)]
            add_xc = (
                GPUXCFrameBackend._add_gga_matrix
                if xctype == "GGA" else GPUXCFrameBackend._add_mgga_matrix
            )
            for output, weights in zip(expected[:3], ordinary):
                add_xc(output, ao, weights, indices)
            GPUXCFrameBackend._add_pair_matrix(
                expected[0], ao, special[0], indices,
            )
            GPUXCFrameBackend._add_pair_matrix(
                expected[2], ao, special[1], indices,
            )
            add_xc(expected[3], ao, reference_alpha, indices)
            add_xc(expected[4], ao, reference_beta, indices)

            actual = [cp.zeros((9, 9)) for _ in range(5)]
            add_response_matrices(
                actual[:3], actual[3], actual[4], ao,
                ordinary, special, reference_alpha, reference_beta,
                (0, 2), indices,
            )
            for reference, fused in zip(expected, actual):
                cp.testing.assert_allclose(
                    fused, reference, rtol=2e-13, atol=2e-11,
                )

    def test_fused_center_derivative_matches_tiled_reduction(self):
        for features in (4, 5):
            ao = self._random((10, 8, 17))
            contracted = self._random((5, 4, 8, 17))
            contracted_t = self._random((5, 4, 8, 17))
            weights = self._random((5, features, 17))
            pair_products = self._random((3, 4, 8, 17))
            pair_products_t = self._random((3, 4, 8, 17))
            pair_weights = self._random((3, 4, 4, 17))
            atom_ids = cp.asarray([2, 0, 2, 1, 3, 1, 0, 3])
            atmlst = (3, 0, 2, 3)
            expected = contract_centers(
                ao, contracted, contracted_t, weights, atom_ids, atmlst,
                xp=cp,
                pair=(pair_products, pair_products_t, pair_weights),
            )
            actual = contract_response_centers(
                ao, contracted, contracted_t, weights,
                pair_products, pair_products_t, pair_weights,
                atom_ids, atmlst,
            )
            cp.testing.assert_allclose(
                actual, expected, rtol=2e-13, atol=2e-11,
            )


if __name__ == "__main__":
    unittest.main()
