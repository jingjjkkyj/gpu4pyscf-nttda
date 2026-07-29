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
        )

        with mock.patch.object(
                nttda, "_import_forge",
                return_value=(None, None, forge_solver)), \
                mock.patch("pyscf.dft.ROKS", return_value=cpu_mf):
            cpu_td = nttda.build_cpu_twin(gpu_td)

        self.assertAlmostEqual(cpu_td._scf.omega, 0.37)

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
