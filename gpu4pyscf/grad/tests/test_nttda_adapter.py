# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from gpu4pyscf.grad import nttda
from gpu4pyscf.nac import nttda as nac_nttda


class KnownValues(unittest.TestCase):
    def test_bilinear_pair_compression_preserves_exact_contraction(self):
        rng = np.random.default_rng(41)
        left = [rng.normal(size=(3, 3)) for _ in range(2)]
        right = [rng.normal(size=(3, 3)) for _ in range(2)]
        left_mix = np.array((1.25, -0.4))
        right_mix = np.array((0.7, 1.1))
        coefficients = np.outer(left_mix, right_mix)
        pairs = [
            (left[i], right[j])
            for i in range(2)
            for j in range(2)
        ]
        factors = coefficients.ravel()
        kernel = rng.normal(size=(3, 3, 3, 3))

        compressed, compressed_factors, details = (
            nttda._compress_bilinear_pairs(pairs, factors)
        )

        reference = sum(
            factor * np.einsum("pq,pqrs,rs->", ldm, kernel, rdm)
            for (ldm, rdm), factor in zip(pairs, factors)
        )
        candidate = sum(
            factor * np.einsum("pq,pqrs,rs->", ldm, kernel, rdm)
            for (ldm, rdm), factor in zip(
                compressed, compressed_factors,
            )
        )
        self.assertEqual(details["input_pairs"], 4)
        self.assertEqual(details["coefficient_rank"], 1)
        self.assertEqual(details["output_pairs"], 1)
        self.assertAlmostEqual(candidate, reference, places=11)

    def test_bilinear_pair_compression_keeps_full_rank_pair_list(self):
        rng = np.random.default_rng(57)
        left = [rng.normal(size=(2, 2)) for _ in range(2)]
        right = [rng.normal(size=(2, 2)) for _ in range(2)]
        pairs = [
            (left[0], right[0]),
            (left[1], right[1]),
        ]
        factors = np.ones(2)

        compressed, compressed_factors, details = (
            nttda._compress_bilinear_pairs(pairs, factors)
        )

        self.assertIs(compressed, pairs)
        self.assertIs(compressed_factors, factors)
        self.assertFalse(details["compressed"])
        self.assertEqual(details["coefficient_rank"], 2)

    def test_density_cache_key_uses_combination_recipe_not_temporary_buffer(self):
        source_a = np.eye(2)
        source_b = np.ones((2, 2))
        first = nttda._linear_combination(
            (source_a, source_b), np.array((1.0, -0.5)),
        )
        equivalent = nttda._linear_combination(
            (source_a, source_b), np.array((1.0, -0.5)),
        )
        different = nttda._linear_combination(
            (source_a, source_b), np.array((-0.5, 1.0)),
        )

        self.assertEqual(
            nttda._density_cache_key(first),
            nttda._density_cache_key(equivalent),
        )
        self.assertNotEqual(
            nttda._density_cache_key(first),
            nttda._density_cache_key(different),
        )

    def test_slot_pair_groups_keep_outputs_and_jk_mappings_separate(self):
        density_a = np.arange(4.0).reshape(2, 2)
        density_b = np.arange(4.0, 8.0).reshape(2, 2)
        terms = (
            ("j", types.SimpleNamespace(
                left=density_a, right=density_b,
                scale=0.25, slot="gradient",
            )),
            ("k", types.SimpleNamespace(
                left=density_a, right=density_b,
                scale=-0.75, slot="nac",
            )),
        )

        groups = nttda._expanded_slot_pair_groups(terms)

        self.assertEqual(list(groups), [("j", "gradient"), ("k", "nac")])
        j_pairs, j_factors = groups[("j", "gradient")]
        self.assertEqual(len(j_pairs), 1)
        self.assertEqual(j_factors, [0.5])
        self.assertIs(j_pairs[0][0], density_a)
        self.assertIs(j_pairs[0][1], density_b)
        k_pairs, k_factors = groups[("k", "nac")]
        self.assertEqual(len(k_pairs), 2)
        self.assertEqual(k_factors, [1.5, 1.5])
        np.testing.assert_equal(k_pairs[0][0], density_a)
        np.testing.assert_equal(k_pairs[0][1], density_b.T)
        np.testing.assert_equal(k_pairs[1][0], density_a.T)
        np.testing.assert_equal(k_pairs[1][1], density_b)

    def test_gpu_route_combines_jk_for_one_density_batch(self):
        calls = []

        class FakeGPUReference:
            mol = object()

            @staticmethod
            def get_jk(mol, dms, hermi, with_j, with_k, omega=None):
                calls.append((np.asarray(dms).shape, hermi, omega))
                return np.asarray(dms) + 1.0, np.asarray(dms) + 2.0

        cpu_mf = types.SimpleNamespace(
            mol=types.SimpleNamespace(nao_nr=lambda: 2),
        )
        numpy_backend = types.SimpleNamespace(
            asarray=np.asarray,
            asnumpy=np.asarray,
        )
        with mock.patch.object(nttda, "cp", numpy_backend):
            nttda.route_jk_to_gpu(cpu_mf, FakeGPUReference())
            dm = np.arange(4.0).reshape(2, 2)
            vj, vk = cpu_mf.get_jk(dm=dm, hermi=0)

        self.assertEqual(calls, [((1, 2, 2), 0, None)])
        np.testing.assert_allclose(vj, dm + 1.0)
        np.testing.assert_allclose(vk, dm + 2.0)
        self.assertEqual(
            cpu_mf._nttda_jk_route_stats["combined_jk_calls"], 1,
        )
        self.assertEqual(
            cpu_mf._nttda_jk_route_stats["density_matrices"], 1,
        )

    def test_cpu_forge_input_is_rejected_without_mutation(self):
        get_j = object()
        cpu_mf = types.SimpleNamespace(get_j=get_j)
        cpu_td = types.SimpleNamespace(_scf=cpu_mf)

        with self.assertRaisesRegex(TypeError, "GPU NTTDA"):
            nttda._resolve_input(cpu_td)

        self.assertIs(cpu_mf.get_j, get_j)
        self.assertFalse(hasattr(cpu_mf, "_nttda_gpu_routed"))

    def test_public_gradient_rejects_cpu_input_before_importing_forge(self):
        cpu_td = types.SimpleNamespace(_scf=types.SimpleNamespace())
        with mock.patch.object(
                nttda, "_make_gradients_class") as make_class:
            with self.assertRaisesRegex(TypeError, "GPU NTTDA"):
                nttda.Gradients(cpu_td)
        make_class.assert_not_called()

    def test_public_nac_rejects_cpu_input_before_importing_forge(self):
        cpu_td = types.SimpleNamespace(_scf=types.SimpleNamespace())
        with mock.patch.object(
                nac_nttda, "_make_nac_class") as make_class:
            with self.assertRaisesRegex(TypeError, "GPU NTTDA"):
                nac_nttda.NAC(cpu_td)
        make_class.assert_not_called()

    def test_unsupported_energy_surfaces_are_rejected(self):
        cases = (
            ("only_dfj", types.SimpleNamespace(only_dfj=True)),
            ("solvent", types.SimpleNamespace(with_solvent=object())),
            ("QM/MM", types.SimpleNamespace(mm_mol=object())),
            ("dispersion", types.SimpleNamespace(disp="d3bj")),
            ("nonlocal", types.SimpleNamespace(nlc="VV10")),
        )
        for label, reference in cases:
            with self.subTest(label=label):
                with self.assertRaises(NotImplementedError):
                    nttda._validate_supported_reference(reference)

    def test_cpu_twin_preserves_tuned_range_separation(self):
        cpu_mf = types.SimpleNamespace(
            omega=0.0,
            grids=types.SimpleNamespace(
                coords=None, weights=None, non0tab=None,
            ),
        )

        class FakeForgeTD:
            def __init__(self, mf):
                self._scf = mf

        forge_solver = types.SimpleNamespace(NTTDA=FakeForgeTD)
        gpu_mf = types.SimpleNamespace(
            mol=object(),
            xc="CAM-B3LYP",
            omega=0.37,
            verbose=0,
            max_memory=4000,
            mo_coeff=np.eye(2),
            mo_occ=np.array([1.0, 0.0]),
            mo_energy=np.array([-0.5, 0.2]),
            grids=types.SimpleNamespace(
                coords=np.zeros((2, 3)),
                weights=np.ones(2),
            ),
        )
        gpu_td = types.SimpleNamespace(
            _scf=gpu_mf,
            deltaS=-1,
            nobeta=False,
            e=np.array([0.1, 0.2]),
            xy=[(np.ones((1, 1)), 0), (np.ones((1, 1)), 0)],
            converged=np.array([True, True]),
            _nttda_gpu_fxc_ref=object(),
            _nttda_gpu_fock0_fockz=object(),
        )

        with mock.patch.object(
                nttda, "_import_forge",
                return_value=(None, None, forge_solver)), \
                mock.patch("pyscf.dft.ROKS", return_value=cpu_mf):
            cpu_td = nttda.build_cpu_twin(gpu_td)

        self.assertAlmostEqual(cpu_td._scf.omega, 0.37)
        self.assertIs(
            cpu_td._nttda_gpu_fxc_ref, gpu_td._nttda_gpu_fxc_ref,
        )
        self.assertIs(
            cpu_td._nttda_gpu_fock0_fockz,
            gpu_td._nttda_gpu_fock0_fockz,
        )

    def test_cpu_twin_does_not_assign_an_unset_range_separation(self):
        class FakeCpuMF:
            def __init__(self):
                self._omega = 0.0
                self.grids = types.SimpleNamespace(
                    coords=None, weights=None, non0tab=None,
                )

            @property
            def omega(self):
                return self._omega

            @omega.setter
            def omega(self, value):
                if value is None:
                    raise TypeError("omega must be numeric")
                self._omega = float(value)

        class FakeForgeTD:
            def __init__(self, mf):
                self._scf = mf

        cpu_mf = FakeCpuMF()
        forge_solver = types.SimpleNamespace(NTTDA=FakeForgeTD)
        gpu_mf = types.SimpleNamespace(
            mol=object(),
            xc="B3LYP",
            omega=None,
            verbose=0,
            max_memory=4000,
            mo_coeff=np.eye(2),
            mo_occ=np.array([1.0, 0.0]),
            mo_energy=np.array([-0.5, 0.2]),
            grids=types.SimpleNamespace(
                coords=np.zeros((2, 3)),
                weights=np.ones(2),
            ),
        )
        gpu_td = types.SimpleNamespace(
            _scf=gpu_mf,
            deltaS=-1,
            nobeta=False,
            e=np.array([0.1, 0.2]),
            xy=[(np.ones((1, 1)), 0), (np.ones((1, 1)), 0)],
            converged=np.array([True, True]),
        )

        with mock.patch.object(
                nttda, "_import_forge",
                return_value=(None, None, forge_solver)), \
                mock.patch("pyscf.dft.ROKS", return_value=cpu_mf):
            cpu_td = nttda.build_cpu_twin(gpu_td)

        self.assertEqual(cpu_td._scf.omega, 0.0)

    def test_cpu_twin_preserves_ensemble_reference(self):
        class FakeCpuMF:
            def __init__(self):
                self.omega = 0.0
                self.grids = types.SimpleNamespace(
                    coords=None, weights=None, non0tab=None,
                )

        class FakeForgeTD:
            def __init__(self, mf):
                self._scf = mf

        cpu_mf = FakeCpuMF()
        forge_solver = types.SimpleNamespace(NTTDA=FakeForgeTD)
        gpu_mf = types.SimpleNamespace(
            is_ensemble_rks=True,
            nopen=2,
            mol=object(),
            xc="PBE",
            omega=0.0,
            verbose=0,
            max_memory=4000,
            mo_coeff=np.eye(3),
            mo_occ=np.array([1.0, 1.0, 0.0]),
            mo_energy=np.array([-0.5, -0.3, 0.2]),
            grids=types.SimpleNamespace(
                coords=np.zeros((2, 3)), weights=np.ones(2),
            ),
        )
        gpu_td = types.SimpleNamespace(
            _scf=gpu_mf,
            deltaS=-1,
            nobeta=False,
            e=np.array([0.1]),
            xy=[(np.ones((2, 2)), 0)],
            converged=np.array([True]),
        )

        with mock.patch.object(
                nttda, "_import_forge",
                return_value=(None, None, forge_solver)), \
                mock.patch(
                    "pyscf.sftda.EnsembleRKS", return_value=cpu_mf,
                ) as ensemble_class, \
                mock.patch("pyscf.dft.ROKS") as roks_class:
            cpu_td = nttda.build_cpu_twin(gpu_td)

        ensemble_class.assert_called_once_with(gpu_mf.mol, nopen=2)
        roks_class.assert_not_called()
        self.assertTrue(cpu_td._scf.converged)
        np.testing.assert_array_equal(cpu_td._scf.mo_occ, gpu_mf.mo_occ)

    def test_requested_forge_root_must_match_loaded_module(self):
        with tempfile.TemporaryDirectory() as root:
            module = types.SimpleNamespace(
                __file__=os.path.join(root, "pyscf", "grad", "nttda.py"),
            )
            nttda._assert_module_under_root(module, root, "gradient")
            module.__file__ = "/another/forge/pyscf/grad/nttda.py"
            with self.assertRaisesRegex(ImportError, "gradient"):
                nttda._assert_module_under_root(module, root, "gradient")


if __name__ == "__main__":
    unittest.main()
