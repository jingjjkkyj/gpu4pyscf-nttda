# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""GPU-native fixed-grid GGA/MGGA contractions for one NTTDA frame.

The CPU forge remains the formula/reference implementation.  This module
evaluates its expensive grid algebra with the native gpu4pyscf NumInt and
returns only the final MO matrices and atomic derivatives to the CPU
orchestrator.
"""

from dataclasses import dataclass
import importlib

import cupy as cp
import numpy as np

from gpu4pyscf.dft import numint as gpu_numint
from gpu4pyscf.lib.cupy_helper import add_sparse


_SECOND_DERIVATIVE = {
    (0, 0): 4,
    (0, 1): 5,
    (0, 2): 6,
    (1, 1): 7,
    (1, 2): 8,
    (2, 2): 9,
}


def _second_derivative_index(first, second):
    if first > second:
        first, second = second, first
    return _SECOND_DERIVATIVE[first, second]


def _ao_center_derivative(ao, atom_indices, xyz):
    """Derivative of AO features on one center in GPU AO layout."""
    delta = cp.empty((4, len(atom_indices), ao.shape[-1]), dtype=ao.dtype)
    delta[0] = -ao[xyz + 1, atom_indices]
    for feature in range(3):
        delta[feature + 1] = -ao[
            _second_derivative_index(xyz, feature), atom_indices,
        ]
    return delta


@dataclass
class _DensityDerivativeWorkspace:
    """Resident ``D @ AO`` products for all densities in one grid block."""

    contracted: cp.ndarray
    contracted_transpose: cp.ndarray
    xctype: str

    def derivatives(self, delta, atom_indices):
        contracted = self.contracted[:, :, atom_indices]
        contracted_t = self.contracted_transpose[:, :, atom_indices]
        density_count = len(contracted)
        grids = contracted.shape[-1]
        feature_count = 4 if self.xctype == "GGA" else 5
        output = cp.empty(
            (density_count, feature_count, grids), dtype=contracted.dtype,
        )

        output[:, 0] = cp.einsum(
            "ag,nag->ng", delta[0], contracted_t[:, 0],
        )
        output[:, 0] += cp.einsum(
            "nag,ag->ng", contracted[:, 0], delta[0],
        )
        for feature in range(1, 4):
            output[:, feature] = cp.einsum(
                "ag,nag->ng", delta[feature], contracted_t[:, 0],
            )
            output[:, feature] += cp.einsum(
                "nag,ag->ng", contracted[:, feature], delta[0],
            )
            output[:, feature] += cp.einsum(
                "ag,nag->ng", delta[0], contracted_t[:, feature],
            )
            output[:, feature] += cp.einsum(
                "nag,ag->ng", contracted[:, 0], delta[feature],
            )
        if self.xctype == "MGGA":
            output[:, 4] = 0.5 * cp.einsum(
                "fag,nfag->ng", delta[1:4], contracted_t[:, 1:4],
            )
            output[:, 4] += 0.5 * cp.einsum(
                "nfag,fag->ng", contracted[:, 1:4], delta[1:4],
            )
        return output


def _density_workspace(ao, densities, hermitian=False, xctype="GGA"):
    if xctype not in ("GGA", "MGGA"):
        raise ValueError(f"unsupported density workspace XC type {xctype}")
    densities = cp.asarray(densities)
    contracted = cp.einsum(
        "nba,fbg->nfag", densities, ao[:4],
    )
    if hermitian:
        contracted_t = contracted
    else:
        contracted_t = cp.einsum(
            "nab,fbg->nfag", densities, ao[:4],
        )
    return _DensityDerivativeWorkspace(contracted, contracted_t, xctype)


def _pair_feature_batches(ao, densities):
    densities = cp.asarray(densities)
    if len(densities) == 0:
        shape = (0, 4, 4, ao.shape[-1])
        products = cp.empty((0, 4, ao.shape[1], ao.shape[-1]))
        return cp.empty(shape), products, products
    contracted = cp.einsum("nba,fbg->nfag", densities, ao[:4])
    contracted_t = cp.einsum("nab,fbg->nfag", densities, ao[:4])
    features = cp.einsum("nlag,rag->nlrg", contracted, ao[:4])
    return features, contracted, contracted_t


def _contract_pair_feature_derivatives(
        delta, contracted, contracted_t, atom_indices,
        tensor_weights, grid_weights):
    if len(tensor_weights) == 0:
        return cp.asarray(0.0)
    contracted_atom = contracted[:, :, atom_indices]
    contracted_t_atom = contracted_t[:, :, atom_indices]
    derivative = cp.einsum(
        "lag,nbag->nlbg", delta, contracted_t_atom,
    )
    derivative += cp.einsum(
        "nlag,bag->nlbg", contracted_atom, delta,
    )
    return cp.einsum(
        "nlbg,nlbg,g->", tensor_weights, derivative, grid_weights,
    )


def _gga_pair_potential(kernel, features):
    output = cp.zeros_like(features)
    output[0, 0] = cp.einsum("abg,abg->g", kernel, features)
    output[1:4, 0] = kernel[1:4, 0] * features[0, 0]
    output[1:4, 0] += cp.einsum(
        "ijg,jg->ig", kernel[1:4, 1:4], features[0, 1:4],
    )
    output[0, 1:4] = kernel[0, 1:4] * features[0, 0]
    output[0, 1:4] += cp.einsum(
        "ijg,ig->jg", kernel[1:4, 1:4], features[1:4, 0],
    )
    output[1:4, 1:4] = kernel[1:4, 1:4] * features[0, 0]
    return output


def _gga_pair_kernel_cross(left, right):
    output = cp.zeros_like(left)
    output[0, 0] = left[0, 0] * right[0, 0]
    output[1:4, 0] = (
        left[0, 0][None] * right[1:4, 0]
        + left[1:4, 0] * right[0, 0][None]
    )
    output[0, 1:4] = (
        left[0, 0][None] * right[0, 1:4]
        + left[0, 1:4] * right[0, 0][None]
    )
    output[1:4, 1:4] = (
        left[0, 0][None, None] * right[1:4, 1:4]
        + left[1:4, 0][:, None] * right[0, 1:4][None]
        + left[0, 1:4][None] * right[1:4, 0][:, None]
        + left[1:4, 1:4] * right[0, 0][None, None]
    )
    return output


def _mgga_pair_potential(kernel, features):
    """Differentiate an MGGA kernel contracted with AO-pair features."""
    output = cp.zeros_like(features)
    output[0, 0] = cp.einsum(
        "abg,abg->g", kernel[:4, :4], features,
    )
    output[1:4, 0] = kernel[1:4, 0] * features[0, 0]
    output[1:4, 0] += cp.einsum(
        "ijg,jg->ig", kernel[1:4, 1:4], features[0, 1:4],
    )
    output[1:4, 0] += 0.5 * kernel[4, 0][None] * features[1:4, 0]
    output[1:4, 0] += 0.5 * cp.einsum(
        "jg,ijg->ig", kernel[4, 1:4], features[1:4, 1:4],
    )
    output[0, 1:4] = kernel[0, 1:4] * features[0, 0]
    output[0, 1:4] += cp.einsum(
        "ijg,ig->jg", kernel[1:4, 1:4], features[1:4, 0],
    )
    output[0, 1:4] += 0.5 * kernel[0, 4][None] * features[0, 1:4]
    output[0, 1:4] += 0.5 * cp.einsum(
        "ig,ijg->jg", kernel[1:4, 4], features[1:4, 1:4],
    )
    output[1:4, 1:4] = kernel[1:4, 1:4] * features[0, 0]
    output[1:4, 1:4] += 0.5 * cp.einsum(
        "ig,jg->ijg", kernel[1:4, 4], features[0, 1:4],
    )
    output[1:4, 1:4] += 0.5 * cp.einsum(
        "jg,ig->ijg", kernel[4, 1:4], features[1:4, 0],
    )
    output[1:4, 1:4] += (
        0.25 * kernel[4, 4][None, None] * features[1:4, 1:4]
    )
    return output


def _mgga_pair_kernel_cross(left, right):
    grids = left.shape[-1]
    output = cp.zeros((5, 5, grids), dtype=left.dtype)
    output[:4, :4] = _gga_pair_kernel_cross(left, right)
    output[4, 0] = 0.5 * cp.einsum(
        "ig,ig->g", left[1:4, 0], right[1:4, 0],
    )
    output[0, 4] = 0.5 * cp.einsum(
        "jg,jg->g", left[0, 1:4], right[0, 1:4],
    )
    output[4, 1:4] = 0.5 * cp.einsum(
        "ig,ijg->jg", left[1:4, 0], right[1:4, 1:4],
    )
    output[4, 1:4] += 0.5 * cp.einsum(
        "ijg,ig->jg", left[1:4, 1:4], right[1:4, 0],
    )
    output[1:4, 4] = 0.5 * cp.einsum(
        "jg,ijg->ig", left[0, 1:4], right[1:4, 1:4],
    )
    output[1:4, 4] += 0.5 * cp.einsum(
        "ijg,jg->ig", left[1:4, 1:4], right[0, 1:4],
    )
    output[4, 4] = 0.25 * cp.einsum(
        "ijg,ijg->g", left[1:4, 1:4], right[1:4, 1:4],
    )
    return output


def _forge_xc():
    return importlib.import_module("pyscf.grad.nttda.xc")


class GPUXCFrameBackend:
    """Geometry-fixed GPU GGA/MGGA backend shared by derivative tasks."""

    def __init__(self, gmf, cpu_td):
        self.gmf = gmf
        self.cpu_td = cpu_td
        self.mol = gmf.mol
        self.ni = gmf._numint
        self.grids = gmf.grids
        if self.grids.coords is None:
            self.grids.build(sort_grids=True)
        self.xctype = self.ni._xc_type(gmf.xc)
        if self.xctype not in ("GGA", "MGGA"):
            raise NotImplementedError(
                "GPU-native NTTDA XC currently supports GGA/MGGA functionals"
            )
        xcfuns = self.ni._init_xcfuns(gmf.xc, spin=1)
        if not all(getattr(function, "on_gpu", False)
                   for function, _weight in xcfuns):
            raise NotImplementedError(
                f"{gmf.xc} has no fully GPU-native spin-polarized XC kernel"
            )
        if self.ni.gdftopt is None:
            self.ni.build(self.mol, self.grids.coords)
        self.opt = self.ni.gdftopt
        self.sorted_mol = self.opt._sorted_mol
        self.nao = self.mol.nao_nr()
        self.mo = cp.asarray(gmf.mo_coeff)
        self.mo_sorted = self.opt.sort_orbitals(self.mo, axis=[0])
        self.mo_occ = cp.asarray(gmf.mo_occ)

        original_atom = np.empty(self.nao, dtype=np.int32)
        for atom, (_b0, _b1, p0, p1) in enumerate(
                self.mol.offset_nr_by_atom()):
            original_atom[p0:p1] = atom
        self.sorted_atom = cp.asarray(original_atom[self.opt._ao_idx])
        self.stats = {
            "grid_passes": 0,
            "grid_blocks": 0,
            "cpu_fallbacks": 0,
            "density_uploads": 0,
            "ao_cache_builds": 0,
            "ao_cache_hits": 0,
            "ao_cache_bytes": 0,
            "spin_fock_builds": 0,
            "spin_fock_reuses": 0,
            "response_builds": 0,
            "response_calls": 0,
            "response_rhs": 0,
        }
        self._ao_blocks = {}
        self._resident_ao_ids = set()
        self._reference_kernels = {}
        self._response_kernel = None
        self._responses = {}
        self._assert_high_order_xc()

    def spin_lowering_fock0_fockz(self):
        """Return the device-built ``F0/Fz`` pair used by the CPU formulas."""
        cached = getattr(self.cpu_td, "_nttda_gpu_fock0_fockz", None)
        if cached is not None:
            self.stats["spin_fock_reuses"] += 1
            return tuple(cp.asnumpy(cp.asarray(value)) for value in cached)

        from gpu4pyscf.sftda import nttda as gpu_nttda

        fxc_ref = getattr(self.cpu_td, "_nttda_gpu_fxc_ref", None)
        if fxc_ref is None:
            fxc_ref = gpu_nttda.spin_flip_reference_fxc(self.gmf)
        _response, fockz = gpu_nttda.gen_rohf_response_sfd(
            self.gmf,
            fxc_ref=fxc_ref,
            hermi=0,
            use_mo_grid_fxc1=True,
        )
        if bool(getattr(self.cpu_td, "nobeta", False)):
            density_alpha, density_beta = self.gmf.make_rdm1()
            density0 = 0.5 * (density_alpha + density_beta)
            fock = self.gmf.get_fock(dm=cp.stack((density0, density0)))
        else:
            fock = self.gmf.get_fock()
        fock0 = 0.5 * (
            cp.asarray(fock.focka) + cp.asarray(fock.fockb)
        )
        self.stats["spin_fock_builds"] += 1
        return cp.asnumpy(fock0), cp.asnumpy(cp.asarray(fockz))

    def _assert_high_order_xc(self):
        """Reject GPU libxc builds that silently return zero fxc/kxc.

        gpu4pyscf-libxc 0.5 advertised GPU execution but returned zero
        second- and third-order GGA kernels.  Those zeros give plausible
        looking energies while corrupting every NTTDA derivative.  Probe the
        actual runtime capability instead of trusting the ``on_gpu`` flag.
        """
        rho = cp.asarray((
            ((0.37, 0.43), (0.05, -0.03), (0.02, 0.04),
             (-0.01, 0.03), (0.11, 0.13)),
            ((0.21, 0.29), (-0.02, 0.01), (0.03, -0.02),
             (0.04, 0.02), (0.07, 0.09)),
        ))
        if self.xctype == "GGA":
            rho = rho[:, :4]
        try:
            values = self.ni.eval_xc_eff(
                self.gmf.xc, rho, deriv=3, xctype=self.xctype, spin=1,
            )
            fxc, kxc = values[2], values[3]
            valid = (
                bool(cp.all(cp.isfinite(fxc)))
                and bool(cp.all(cp.isfinite(kxc)))
                and float(cp.max(cp.abs(fxc))) > 1e-14
                and float(cp.max(cp.abs(kxc))) > 1e-14
            )
        except Exception as error:
            raise RuntimeError(
                f"GPU-native NTTDA derivatives require working spin-{self.xctype} "
                "fxc/kxc kernels (gpu4pyscf-libxc-cuda12x>=0.8.1)"
            ) from error
        if not valid:
            raise RuntimeError(
                f"GPU libxc returned an invalid spin-{self.xctype} fxc/kxc "
                "kernel; "
                "install gpu4pyscf-libxc-cuda12x>=0.8.1"
            )

    def _build_response_kernel(self):
        """Build the geometry-fixed spin UKS fxc data once on the GPU."""
        if self._response_kernel is None:
            mo_coeff = cp.stack((self.mo, self.mo))
            mo_occ = cp.stack((
                (self.mo_occ > 0).astype(float),
                (self.mo_occ == 2).astype(float),
            ))
            rho0, vxc, fxc = self.ni.cache_xc_kernel(
                self.mol,
                self.grids,
                self.gmf.xc,
                mo_coeff,
                mo_occ,
                1,
            )
            omega, alpha, hybrid_coefficient = (
                self.ni.rsh_and_hybrid_coeff(
                    self.gmf.xc, self.mol.spin,
                )
            )
            self._response_kernel = (
                rho0,
                vxc,
                fxc,
                float(omega),
                float(alpha),
                float(hybrid_coefficient),
                bool(self.ni.libxc.is_hybrid_xc(self.gmf.xc)),
            )
        return self._response_kernel

    def response(self, hermi):
        """Return a NumPy-in/NumPy-out GPU UKS response closure.

        The closure is the same response operator used by the forge ROKS
        adjoint: spin-resolved GGA fxc plus Coulomb and hybrid/range-separated
        exchange.  All grid algebra and J/K builds remain on the GPU.
        """
        hermi = int(hermi)
        if hermi not in self._responses:
            (
                rho0,
                vxc,
                fxc,
                omega,
                alpha,
                hybrid_coefficient,
                is_hybrid,
            ) = self._build_response_kernel()
            self.stats["response_builds"] += 1

            def apply_response(density):
                density = cp.asarray(density)
                if density.ndim <= 3:
                    width = 1
                elif density.shape[0] == 2:
                    width = int(np.prod(density.shape[1:-2]))
                else:
                    width = int(np.prod(density.shape[:-2]))
                self.stats["response_calls"] += 1
                self.stats["response_rhs"] += width
                if hermi == 2:
                    potential = cp.zeros_like(density)
                else:
                    potential = self.ni.nr_uks_fxc(
                        self.mol,
                        self.grids,
                        self.gmf.xc,
                        None,
                        density,
                        0,
                        hermi,
                        rho0,
                        vxc,
                        fxc,
                        max_memory=self.gmf.max_memory,
                    )
                if is_hybrid:
                    coulomb, exchange = self.gmf.get_jk(
                        self.mol, density, hermi=hermi,
                    )
                    exchange *= hybrid_coefficient
                    if omega > 1e-10:
                        exchange += self.gmf.get_k(
                            self.mol, density, hermi=hermi, omega=omega,
                        ) * (alpha - hybrid_coefficient)
                    potential += coulomb[0] + coulomb[1] - exchange
                else:
                    coulomb = self.gmf.get_j(
                        self.mol, density, hermi=hermi,
                    )
                    potential += coulomb[0] + coulomb[1]
                return cp.asnumpy(potential)

            self._responses[hermi] = apply_response
        return self._responses[hermi]

    def _blocks(self, deriv=2):
        self.stats["grid_passes"] += 1
        cached = self._ao_blocks.get(deriv)
        if cached is not None:
            self.stats["ao_cache_hits"] += 1
            for block in cached:
                yield block
            return

        # A full deriv=2 AO residency is intentionally opportunistic.  It is
        # normally disabled on small GPUs by the conservative worst-case
        # estimate, but an A100 can retain it and avoid reevaluating AO values
        # in the Fock-Z and post-Z grid passes.
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
        components = (deriv + 1) * (deriv + 2) * (deriv + 3) // 6
        feature_count = 4 if self.xctype == "GGA" else 5
        kernel_arrays = feature_count**2 + 2 * feature_count**3
        estimated_bytes = (
            (components * self.nao + kernel_arrays)
            * len(self.grids.weights) * 8
        )
        cache_budget = min(int(0.35 * free_bytes), 12 * 1024**3)
        cache_blocks = deriv == 2 and estimated_bytes <= cache_budget
        resident = []
        for ao, indices, weights, coords in self.ni.block_loop(
                self.sorted_mol, self.grids, self.nao, deriv,
                max_memory=None, strict_grid_order=True):
            self.stats["grid_blocks"] += 1
            if cache_blocks:
                block = (
                    ao.copy(), indices.copy(), weights.copy(), None,
                )
                resident.append(block)
                self._resident_ao_ids.add(id(block[0]))
                self.stats["ao_cache_bytes"] += sum(
                    value.nbytes for value in block[:3]
                )
                yield block
            else:
                yield ao, indices, weights, coords
        if cache_blocks:
            self._ao_blocks[deriv] = tuple(resident)
            self.stats["ao_cache_builds"] += 1

    def _sort_density(self, density):
        self.stats["density_uploads"] += 1
        density = cp.asarray(density)
        return self.opt.sort_orbitals(
            density, axis=[density.ndim - 2, density.ndim - 1],
        )

    @staticmethod
    def _masked_density(density, indices):
        return cp.take(cp.take(density, indices, axis=-2), indices, axis=-1)

    def _reference_kernel(self, ao, indices):
        key = id(ao)
        cached = self._reference_kernels.get(key)
        if cached is not None:
            return cached
        rho0 = self.ni.eval_rho2(
            self.sorted_mol,
            ao,
            self.mo_sorted[indices],
            self.mo_occ,
            None,
            self.xctype,
            with_lapl=False,
        ) * 0.5
        _exc, _vxc, fxc, kxc = self.ni.eval_xc_eff(
            self.gmf.xc,
            cp.stack((rho0, rho0)),
            deriv=3,
            xctype=self.xctype,
            spin=1,
        )
        fref = 0.5 * (
            fxc[0, :, 0] - fxc[0, :, 1]
            - fxc[1, :, 0] + fxc[1, :, 1]
        )
        kref_alpha = 0.5 * (
            kxc[0, :, 0, :, 0] - kxc[0, :, 1, :, 0]
            - kxc[1, :, 0, :, 0] + kxc[1, :, 1, :, 0]
        )
        kref_beta = 0.5 * (
            kxc[0, :, 0, :, 1] - kxc[0, :, 1, :, 1]
            - kxc[1, :, 0, :, 1] + kxc[1, :, 1, :, 1]
        )
        output = (fref, kref_alpha, kref_beta)
        if key in self._resident_ao_ids:
            self._reference_kernels[key] = output
            self.stats["ao_cache_bytes"] += sum(
                value.nbytes for value in output
            )
        return output

    @staticmethod
    def _add_gga_matrix(output, ao, weights, indices):
        weights = cp.array(weights, copy=True, order="C")
        weights[0] *= 0.5
        scaled = gpu_numint._scale_ao(ao[:4], weights)
        matrix = ao[0].dot(scaled.T)
        add_sparse(output, matrix + matrix.T, indices)

    @staticmethod
    def _add_mgga_matrix(output, ao, weights, indices):
        weights = cp.array(weights, copy=True, order="C")
        weights[0] *= 0.5
        scaled = gpu_numint._scale_ao(ao[:4], weights[:4])
        matrix = ao[0].dot(scaled.T)
        matrix += matrix.T
        matrix += gpu_numint._tau_dot(ao, ao, weights[4])
        add_sparse(output, matrix, indices)

    def _add_xc_matrix(self, output, ao, weights, indices):
        if self.xctype == "GGA":
            self._add_gga_matrix(output, ao, weights, indices)
        else:
            self._add_mgga_matrix(output, ao, weights, indices)

    @staticmethod
    def _add_pair_matrix(output, ao, tensor, indices):
        matrix = cp.zeros((len(indices), len(indices)), dtype=ao.dtype)
        for left in range(4):
            scaled = gpu_numint._scale_ao(ao[:4], tensor[left])
            matrix += ao[left].dot(scaled.T)
        add_sparse(output, matrix, indices)

    def _atom_indices(self, block_indices, atom):
        return cp.flatnonzero(self.sorted_atom[block_indices] == atom)

    def _unsort_numpy(self, matrix):
        matrix = self.opt.unsort_orbitals(
            matrix, axis=[matrix.ndim - 2, matrix.ndim - 1],
        )
        return cp.asnumpy(matrix)

    def _response_terms_batch(
            self, gradient_driver, tdobj, channel_data_batch,
            atmlst=None, with_direct=True):
        """Evaluate all NTTDA response channels in one GPU grid pass."""
        del gradient_driver
        channel_data_batch = tuple(channel_data_batch)
        if not channel_data_batch:
            return tuple()
        if atmlst is None:
            atmlst = range(self.mol.natm)
        atmlst = tuple(atmlst)

        occ_alpha = self.mo_occ > 0
        occ_beta = self.mo_occ == 2
        density_alpha = self.mo[:, occ_alpha] @ self.mo[:, occ_alpha].T
        density_beta = self.mo[:, occ_beta] @ self.mo[:, occ_beta].T
        references = self._sort_density(cp.stack((
            density_alpha, density_beta,
        )))
        contexts = []
        for channel_data in channel_data_batch:
            _spaces, _amplitudes, densities, blocks, terms = channel_data
            labels = tuple(densities)
            sorted_densities = self._sort_density(np.asarray([
                densities[label] for label in labels
            ]))
            density_map = dict(zip(labels, sorted_densities))
            pair_labels = tuple(
                label for label in labels
                if any(
                    term.vref1 and label in (term.target, term.source)
                    for term in terms
                )
            )
            pair_stack = cp.stack([
                density_map[label] for label in pair_labels
            ]) if pair_labels else cp.empty((0, self.nao, self.nao))
            contexts.append({
                "labels": labels,
                "blocks": blocks,
                "terms": terms,
                "densities": density_map,
                "density_stack": cp.concatenate((
                    sorted_densities, references,
                )),
                "pair_labels": pair_labels,
                "pair_stack": pair_stack,
                "potentials": {
                    label: cp.zeros((self.nao, self.nao)) for label in labels
                },
                "reference_alpha": cp.zeros((self.nao, self.nao)),
                "reference_beta": cp.zeros((self.nao, self.nao)),
                "direct": cp.zeros((len(atmlst), 3)),
            })

        for ao, indices, grid_weights, _coords in self._blocks(deriv=2):
            fref, kref_alpha, kref_beta = self._reference_kernel(ao, indices)
            for ctx in contexts:
                masked_stack = self._masked_density(
                    ctx["density_stack"], indices,
                )
                workspace = (
                    _density_workspace(
                        ao, masked_stack, xctype=self.xctype,
                    )
                    if with_direct else None
                )
                rho = {}
                for label, density in ctx["densities"].items():
                    rho[label] = self.ni.eval_rho(
                        self.sorted_mol,
                        ao,
                        self._masked_density(density, indices),
                        None,
                        self.xctype,
                        hermi=0,
                        with_lapl=False,
                    )
                pair_densities = self._masked_density(
                    ctx["pair_stack"], indices,
                )
                pair_values, pair_products, pair_products_t = (
                    _pair_feature_batches(ao, pair_densities)
                )
                pairs = dict(zip(ctx["pair_labels"], pair_values))
                pair_potential = (
                    _gga_pair_potential
                    if self.xctype == "GGA" else _mgga_pair_potential
                )
                pair_kernel_cross = (
                    _gga_pair_kernel_cross
                    if self.xctype == "GGA" else _mgga_pair_kernel_cross
                )
                feature_count = 4 if self.xctype == "GGA" else 5
                pair_potentials = {
                    label: pair_potential(fref, pairs[label])
                    for label in ctx["pair_labels"]
                }
                ordinary = {
                    label: cp.zeros((feature_count, len(grid_weights)))
                    for label in ctx["labels"]
                }
                special = {
                    label: cp.zeros((4, 4, len(grid_weights)))
                    for label in ctx["pair_labels"]
                }
                reference_alpha = cp.zeros(
                    (feature_count, len(grid_weights)),
                )
                reference_beta = cp.zeros_like(reference_alpha)

                for term in ctx["terms"]:
                    if term.vref0:
                        ordinary[term.target] += term.vref0 * cp.einsum(
                            "xyg,yg->xg", fref, rho[term.source],
                        )
                        ordinary[term.source] += term.vref0 * cp.einsum(
                            "xyg,xg->yg", fref, rho[term.target],
                        )
                        pair = term.vref0 * cp.einsum(
                            "xg,yg->xyg",
                            rho[term.target], rho[term.source],
                        )
                        reference_alpha += cp.einsum(
                            "xyg,xyzg->zg", pair, kref_alpha,
                        )
                        reference_beta += cp.einsum(
                            "xyg,xyzg->zg", pair, kref_beta,
                        )
                    if term.vref1:
                        special[term.target] += (
                            term.vref1 * pair_potentials[term.source]
                        )
                        special[term.source] += (
                            term.vref1 * pair_potentials[term.target]
                        )
                        pair = term.vref1 * pair_kernel_cross(
                            pairs[term.target], pairs[term.source],
                        )
                        reference_alpha += cp.einsum(
                            "xyg,xyzg->zg", pair, kref_alpha,
                        )
                        reference_beta += cp.einsum(
                            "xyg,xyzg->zg", pair, kref_beta,
                        )

                ordinary_stack = cp.stack([
                    ordinary[label] for label in ctx["labels"]
                ])
                special_stack = cp.stack([
                    special[label] for label in ctx["pair_labels"]
                ]) if ctx["pair_labels"] else cp.empty(
                    (0, 4, 4, len(grid_weights)),
                )
                for label in ctx["labels"]:
                    self._add_xc_matrix(
                        ctx["potentials"][label], ao,
                        ordinary[label] * grid_weights, indices,
                    )
                for label in ctx["pair_labels"]:
                    self._add_pair_matrix(
                        ctx["potentials"][label], ao,
                        special[label] * grid_weights, indices,
                    )
                self._add_xc_matrix(
                    ctx["reference_alpha"], ao,
                    reference_alpha * grid_weights, indices,
                )
                self._add_xc_matrix(
                    ctx["reference_beta"], ao,
                    reference_beta * grid_weights, indices,
                )

                if not with_direct:
                    continue
                for atom_index, atom in enumerate(atmlst):
                    local = self._atom_indices(indices, atom)
                    if len(local) == 0:
                        continue
                    for xyz in range(3):
                        delta = _ao_center_derivative(ao, local, xyz)
                        derivatives = workspace.derivatives(delta, local)
                        channel_count = len(ctx["labels"])
                        drho = derivatives[:channel_count]
                        drho_alpha = derivatives[channel_count]
                        drho_beta = derivatives[channel_count + 1]
                        value = cp.einsum(
                            "nfg,nfg,g->",
                            ordinary_stack, drho, grid_weights,
                        )
                        value += cp.einsum(
                            "fg,fg,g->",
                            reference_alpha, drho_alpha, grid_weights,
                        )
                        value += cp.einsum(
                            "fg,fg,g->",
                            reference_beta, drho_beta, grid_weights,
                        )
                        value += _contract_pair_feature_derivatives(
                            delta,
                            pair_products,
                            pair_products_t,
                            local,
                            special_stack,
                            grid_weights,
                        )
                        ctx["direct"][atom_index, xyz] += value

        forge_xc = _forge_xc()
        output = []
        for ctx in contexts:
            potentials = {
                label: self._unsort_numpy(value)
                for label, value in ctx["potentials"].items()
            }
            q_alpha, q_beta = forge_xc._project_channel_potentials(
                tdobj, potentials, ctx["blocks"],
            )
            forge_xc._add_reference_q(
                tdobj,
                q_alpha,
                q_beta,
                self._unsort_numpy(ctx["reference_alpha"]),
                self._unsort_numpy(ctx["reference_beta"]),
            )
            output.append(forge_xc.XCGradientTerms(
                q_alpha, q_beta, cp.asnumpy(ctx["direct"]),
            ))
        return tuple(output)

    def gga_response_terms_batch(self, *args, **kwargs):
        if self.xctype != "GGA":
            raise ValueError("GGA response requested from an MGGA backend")
        return self._response_terms_batch(*args, **kwargs)

    def mgga_response_terms_batch(self, *args, **kwargs):
        if self.xctype != "MGGA":
            raise ValueError("MGGA response requested from a GGA backend")
        return self._response_terms_batch(*args, **kwargs)

    def _fockz_terms_batch(
            self, gradient_driver, tdobj, spaces, pz_batch,
            atmlst=None, with_direct=True):
        """Evaluate every frame ``Pz:Fz`` task in one GPU grid pass."""
        del gradient_driver
        pz_batch = np.asarray(pz_batch)
        if pz_batch.size == 0:
            return tuple()
        if pz_batch.ndim == 2:
            pz_batch = pz_batch[None]
        pz_batch = 0.5 * (pz_batch + pz_batch.transpose(0, 2, 1))
        if atmlst is None:
            atmlst = range(self.mol.natm)
        atmlst = tuple(atmlst)
        ntask = len(pz_batch)

        density_open = np.asarray(spaces.c_open @ spaces.c_open.T)
        occ_alpha = self.mo_occ > 0
        occ_beta = self.mo_occ == 2
        density_alpha = self.mo[:, occ_alpha] @ self.mo[:, occ_alpha].T
        density_beta = self.mo[:, occ_beta] @ self.mo[:, occ_beta].T
        density_stack = self._sort_density(cp.concatenate((
            cp.asarray(pz_batch),
            cp.stack((
                cp.asarray(density_open), density_alpha, density_beta,
            )),
        )))
        open_sorted = density_stack[ntask]

        open_potential = cp.zeros((ntask, self.nao, self.nao))
        reference_alpha = cp.zeros_like(open_potential)
        reference_beta = cp.zeros_like(open_potential)
        direct = cp.zeros((ntask, len(atmlst), 3))

        for ao, indices, grid_weights, _coords in self._blocks(deriv=2):
            fref, kref_alpha, kref_beta = self._reference_kernel(ao, indices)
            masked_stack = self._masked_density(density_stack, indices)
            rho_pz = cp.stack([
                self.ni.eval_rho(
                    self.sorted_mol,
                    ao,
                    masked_stack[index],
                    None,
                    self.xctype,
                    hermi=1,
                    with_lapl=False,
                )
                for index in range(ntask)
            ])
            rho_open = self.ni.eval_rho(
                self.sorted_mol,
                ao,
                self._masked_density(open_sorted, indices),
                None,
                self.xctype,
                hermi=1,
                with_lapl=False,
            )
            open_weights = 0.5 * cp.einsum(
                "xyg,nyg->nxg", fref, rho_pz,
            ) * grid_weights
            for index in range(ntask):
                self._add_xc_matrix(
                    open_potential[index], ao, open_weights[index], indices,
                )
            pair = 0.5 * cp.einsum(
                "nxg,yg->nxyg", rho_pz, rho_open,
            )
            alpha_weights = cp.einsum(
                "nxyg,xyzg->nzg", pair, kref_alpha,
            ) * grid_weights
            beta_weights = cp.einsum(
                "nxyg,xyzg->nzg", pair, kref_beta,
            ) * grid_weights
            for index in range(ntask):
                self._add_xc_matrix(
                    reference_alpha[index], ao,
                    alpha_weights[index], indices,
                )
                self._add_xc_matrix(
                    reference_beta[index], ao,
                    beta_weights[index], indices,
                )

            if not with_direct:
                continue
            workspace = _density_workspace(
                ao, masked_stack, hermitian=True, xctype=self.xctype,
            )
            for atom_index, atom in enumerate(atmlst):
                local = self._atom_indices(indices, atom)
                if len(local) == 0:
                    continue
                for xyz in range(3):
                    delta = _ao_center_derivative(ao, local, xyz)
                    derivatives = workspace.derivatives(delta, local)
                    drho_pz = derivatives[:ntask]
                    drho_open = derivatives[ntask]
                    drho_alpha = derivatives[ntask + 1]
                    drho_beta = derivatives[ntask + 2]
                    direct[:, atom_index, xyz] += 0.5 * cp.einsum(
                        "nxg,xyg,yg,g->n",
                        drho_pz, fref, rho_open, grid_weights,
                    )
                    direct[:, atom_index, xyz] += 0.5 * cp.einsum(
                        "nxg,xyg,yg,g->n",
                        rho_pz, fref, drho_open, grid_weights,
                    )
                    direct[:, atom_index, xyz] += cp.einsum(
                        "nxyg,xyzg,zg,g->n",
                        pair, kref_alpha, drho_alpha, grid_weights,
                    )
                    direct[:, atom_index, xyz] += cp.einsum(
                        "nxyg,xyzg,zg,g->n",
                        pair, kref_beta, drho_beta, grid_weights,
                    )

        forge_xc = _forge_xc()
        mo = np.asarray(tdobj._scf.mo_coeff)
        output = []
        for index in range(ntask):
            q_alpha = np.zeros((mo.shape[1], mo.shape[1]))
            q_beta = np.zeros_like(q_alpha)
            potential = self._unsort_numpy(open_potential[index])
            q_alpha[:, spaces.open] += (
                mo.conj().T @ (potential + potential.T) @ spaces.c_open
            )
            forge_xc._add_reference_q(
                tdobj,
                q_alpha,
                q_beta,
                self._unsort_numpy(reference_alpha[index]),
                self._unsort_numpy(reference_beta[index]),
            )
            output.append(forge_xc.XCGradientTerms(
                q_alpha, q_beta, cp.asnumpy(direct[index]),
            ))
        return tuple(output)

    def gga_fockz_terms_batch(self, *args, **kwargs):
        if self.xctype != "GGA":
            raise ValueError("GGA Fock-Z requested from an MGGA backend")
        return self._fockz_terms_batch(*args, **kwargs)

    def mgga_fockz_terms_batch(self, *args, **kwargs):
        if self.xctype != "MGGA":
            raise ValueError("MGGA Fock-Z requested from a GGA backend")
        return self._fockz_terms_batch(*args, **kwargs)

    def _contract_vxc_derivative(
            self, density_alpha, density_beta, probe_alpha, probe_beta,
            atmlst=None):
        """Contract all post-Z fixed-grid XC derivatives on the GPU."""
        if atmlst is None:
            atmlst = range(self.mol.natm)
        atmlst = tuple(atmlst)
        probe_alpha = np.asarray(probe_alpha)
        probe_beta = np.asarray(probe_beta)
        single_probe = probe_alpha.ndim == 2
        if single_probe:
            probe_alpha = probe_alpha[None]
            probe_beta = probe_beta[None]
        probe_alpha = 0.5 * (
            probe_alpha + probe_alpha.swapaxes(-1, -2)
        )
        probe_beta = 0.5 * (
            probe_beta + probe_beta.swapaxes(-1, -2)
        )
        density_alpha = 0.5 * (
            np.asarray(density_alpha) + np.asarray(density_alpha).T
        )
        density_beta = 0.5 * (
            np.asarray(density_beta) + np.asarray(density_beta).T
        )
        probe_densities = np.stack(
            (probe_alpha, probe_beta), axis=1,
        ).reshape(-1, self.nao, self.nao)
        density_stack = self._sort_density(np.concatenate((
            np.asarray((density_alpha, density_beta)), probe_densities,
        )))
        output = cp.zeros((len(probe_alpha), len(atmlst), 3))

        for ao, indices, grid_weights, _coords in self._blocks(deriv=2):
            masked_stack = self._masked_density(density_stack, indices)
            workspace = _density_workspace(
                ao, masked_stack, hermitian=True, xctype=self.xctype,
            )
            rho = cp.stack([
                self.ni.eval_rho(
                    self.sorted_mol,
                    ao,
                    density,
                    None,
                    self.xctype,
                    hermi=1,
                    with_lapl=False,
                )
                for density in masked_stack
            ])
            reference_rho = rho[:2]
            feature_count = 4 if self.xctype == "GGA" else 5
            probe_rho = rho[2:].reshape(
                len(probe_alpha), 2, feature_count, len(grid_weights),
            )
            _exc, vxc, fxc = self.ni.eval_xc_eff(
                self.gmf.xc,
                reference_rho,
                deriv=2,
                xctype=self.xctype,
                spin=1,
            )[:3]
            for atom_index, atom in enumerate(atmlst):
                local = self._atom_indices(indices, atom)
                if len(local) == 0:
                    continue
                for xyz in range(3):
                    delta = _ao_center_derivative(ao, local, xyz)
                    derivatives = workspace.derivatives(delta, local)
                    reference_derivative = derivatives[:2]
                    probe_derivative = derivatives[2:].reshape(
                        len(probe_alpha), 2, feature_count,
                        len(grid_weights),
                    )
                    output[:, atom_index, xyz] += cp.einsum(
                        "nsxg,sxg,g->n",
                        probe_derivative, vxc, grid_weights,
                    )
                    response_weights = cp.einsum(
                        "axg,axbyg,g->byg",
                        reference_derivative, fxc, grid_weights,
                    )
                    output[:, atom_index, xyz] += cp.einsum(
                        "nbyg,byg->n", probe_rho, response_weights,
                    )
        output = cp.asnumpy(output)
        return output[0] if single_probe else output

    def contract_gga_vxc_derivative(self, *args, **kwargs):
        if self.xctype != "GGA":
            raise ValueError("GGA post-Z requested from an MGGA backend")
        return self._contract_vxc_derivative(*args, **kwargs)

    def contract_mgga_vxc_derivative(self, *args, **kwargs):
        if self.xctype != "MGGA":
            raise ValueError("MGGA post-Z requested from a GGA backend")
        return self._contract_vxc_derivative(*args, **kwargs)
