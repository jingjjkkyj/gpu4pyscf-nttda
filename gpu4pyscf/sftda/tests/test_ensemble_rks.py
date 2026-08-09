# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

import unittest

import cupy as cp
import numpy as np

from pyscf import gto
from pyscf.sftda import EnsembleRKS as CPUEnsembleRKS
from pyscf.sftda.nttda import NTTDA as CPUNTTDA

from gpu4pyscf.sftda import EnsembleRKS, NTTDA
from gpu4pyscf.grad.nttda import compute_frame, make_frame_cache


class EnsembleRKSGPU(unittest.TestCase):
    @staticmethod
    def molecule():
        return gto.M(
            atom="C 0 0 0; H 0 0 2.0; H 0 1.7 -0.5",
            basis="sto-3g",
            spin=2,
            unit="Bohr",
            verbose=0,
        )

    @classmethod
    def references(cls, xc="PBE"):
        mol = cls.molecule()
        cpu = CPUEnsembleRKS(mol).set(
            xc=xc, conv_tol=1e-11, max_cycle=150, verbose=0,
        )
        gpu = EnsembleRKS(mol).set(
            xc=xc, conv_tol=1e-11, max_cycle=150, verbose=0,
        )
        cpu.grids.level = 0
        gpu.grids.level = 0
        cpu.kernel()
        gpu.kernel()
        if not cpu.converged or not gpu.converged:
            raise RuntimeError("EnsembleRKS parity references did not converge")
        return cpu, gpu

    def test_fixed_occupations_and_scf_energy_match_cpu(self):
        cpu, gpu = self.references()

        np.testing.assert_array_equal(cp.asnumpy(gpu.mo_occ), cpu.mo_occ)
        self.assertEqual(gpu.mo_occ.tolist().count(1.0), gpu.nopen)
        self.assertAlmostEqual(gpu.e_tot, cpu.e_tot, places=8)
        alpha, beta = gpu.make_rdm1s()
        np.testing.assert_allclose(
            cp.asnumpy(alpha), cp.asnumpy(beta), atol=0.0, rtol=0.0,
        )

    def test_spin_lowering_energy_matches_cpu(self):
        cpu, gpu = self.references("SVWN")
        cpu_td = CPUNTTDA(cpu).set(
            deltaS=-1, nstates=2, conv_tol=1e-9, verbose=0,
        ).run()
        gpu_td = NTTDA(gpu).set(
            deltaS=-1, nstates=2, conv_tol=1e-9, verbose=0,
        ).run()

        np.testing.assert_allclose(gpu_td.e, cpu_td.e, atol=2e-8, rtol=0)

    def test_pbe_analytic_gradient_and_nac_match_cpu(self):
        cpu, gpu = self.references("PBE")
        cpu_td = CPUNTTDA(cpu).set(
            deltaS=-1, nstates=2, conv_tol=1e-9, verbose=0,
        ).run()
        gpu_td = NTTDA(gpu).set(
            deltaS=-1, nstates=2, conv_tol=1e-9, verbose=0,
        ).run()

        cpu_gradient = cpu_td.Gradients().set(
            verbose=0, cphf_conv_tol=1e-10,
        ).kernel(state=1, atmlst=[0])
        gpu_gradient = gpu_td.Gradients().set(
            verbose=0, cphf_conv_tol=1e-10,
        ).kernel(state=1, atmlst=[0])
        np.testing.assert_allclose(
            gpu_gradient, cpu_gradient, atol=3e-6, rtol=0,
        )

        cpu_nac = cpu_td.NAC().set(
            verbose=0, cphf_conv_tol=1e-10,
        ).kernel(
            state_I=1, state_J=2, atmlst=[0],
            ediff=True, use_etfs=False,
        )
        gpu_nac = gpu_td.NAC().set(
            verbose=0, cphf_conv_tol=1e-10,
        ).kernel(
            state_I=1, state_J=2, atmlst=[0],
            ediff=True, use_etfs=False,
        )
        # Independent Davidson solves may choose opposite phases for either
        # state, so only the coupling direction up to a global sign is fixed.
        sign = 1.0 if np.vdot(gpu_nac, cpu_nac) >= 0.0 else -1.0
        np.testing.assert_allclose(
            sign * gpu_nac, cpu_nac, atol=4e-6, rtol=0,
        )

    def test_same_spin_energy_and_gradient_match_cpu(self):
        cpu, gpu = self.references("SVWN")
        cpu_td = CPUNTTDA(cpu).set(
            deltaS=0, nstates=2, conv_tol=1e-9, verbose=0,
        ).run()
        gpu_td = NTTDA(gpu).set(
            deltaS=0, nstates=2, conv_tol=1e-9, verbose=0,
        ).run()

        np.testing.assert_allclose(gpu_td.e, cpu_td.e, atol=2e-8, rtol=0)
        cpu_gradient = cpu_td.Gradients().set(
            verbose=0, cphf_conv_tol=1e-10,
        ).kernel(state=1, atmlst=[0])
        gpu_gradient = gpu_td.Gradients().set(
            verbose=0, cphf_conv_tol=1e-10,
        ).kernel(state=1, atmlst=[0])
        np.testing.assert_allclose(
            gpu_gradient, cpu_gradient, atol=3e-6, rtol=0,
        )

    def test_frame_batches_gradient_nac_and_reuses_zvector_guess(self):
        _cpu, gpu = self.references("PBE")
        tdobj = NTTDA(gpu).set(
            deltaS=-1, nstates=2, conv_tol=1e-9, verbose=0,
        ).run()
        cache = make_frame_cache()

        first = compute_frame(
            tdobj,
            active_state=1,
            nac_pairs=((1, 2),),
            cphf_conv_tol=1e-10,
            use_etfs=False,
            frame_cache=cache,
        )
        first_stats = dict(tdobj._nttda_frame_stats)
        second = compute_frame(
            tdobj,
            active_state=1,
            nac_pairs=((1, 2),),
            cphf_conv_tol=1e-10,
            use_etfs=False,
            frame_cache=cache,
        )
        second_stats = dict(tdobj._nttda_frame_stats)

        np.testing.assert_allclose(
            second["grad"], first["grad"], atol=1e-9, rtol=0,
        )
        np.testing.assert_allclose(
            second["nac"][(1, 2)], first["nac"][(1, 2)],
            atol=1e-9, rtol=0,
        )
        self.assertEqual(first_stats["zvector_batch_width"], 2)
        self.assertEqual(first_stats["zvector_cache_hits"], 0)
        self.assertEqual(second_stats["zvector_cache_hits"], 2)
        self.assertIn(
            2, second_stats["response_cache"]["response_batch_widths"],
        )


if __name__ == "__main__":
    unittest.main()
