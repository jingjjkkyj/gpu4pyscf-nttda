'''Selected-reference gradients and NACs against independently displaced GPU SCF.

Finite differences retain the same DF auxiliary basis and fixed quadrature.
For NACs, exact spin-adapted wavefunction overlaps fix the root order and phase.
'''

import cupy as cp
import numpy as np
import pytest
from unittest import mock

from pyscf import gto
from gpu4pyscf.grad.nttda import compute_frame
from gpu4pyscf.grad.ensemble_roks import ReferenceGradients
from gpu4pyscf.sftda import EnsembleROKS, NTTDA


def make_reference(xc='B3LYP', source=None, coords=None, auxbasis='def2-universal-jkfit'):
    mol = gto.M(
        atom='C 0.02 -0.03 0.01; H -0.02 0.8 0.62; H 0.03 -0.91 0.5',
        basis='sto-3g', spin=2, verbose=0,
    ) if source is None else source.mol.copy().set_geom_(coords, unit='Bohr')
    mf = EnsembleROKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.grids.level = 1
    mf.small_rho_cutoff = 0.0
    if source is not None:
        mf.grids.coords = cp.array(source.grids.coords)
        mf.grids.weights = cp.array(source.grids.weights)
    mf.kernel(dm0=None if source is None else source.make_rdm1())
    assert mf.converged
    return mf


def make_td(mf, delta_s=-1):
    td = NTTDA(mf).set(
        deltaS=delta_s, nstates=3, conv_tol=1e-10, max_cycle=200, verbose=0,
    ).run()
    assert np.all(td.converged)
    assert np.min(np.abs(np.diff(td.e))) > 1e-4
    return td


def displaced_points(mf, atom=1, step=3e-4):
    for xyz in range(3):
        points = []
        for sign in (1, -1):
            coords = mf.mol.atom_coords()
            coords[atom, xyz] += sign * step
            points.append(make_reference(
                mf.xc, source=mf, coords=coords, auxbasis=mf.with_df.auxbasis,
            ))
        yield points


def test_df_reference_gradient_before_public_dispatch():
    '''Isolate the reference response from the public DF capability gate.'''
    mf = make_reference()
    driver = ReferenceGradients(mf)
    driver.conv_tol = 1e-11
    driver._build_intermediates()
    z = driver._solve_z()
    unrelaxed = driver._high_spin_unrelaxed_gradient()
    correction = cp.einsum('i,iax->ax', z, driver._build_b())
    analytic = cp.asnumpy(unrelaxed - correction)[1]
    step = 3e-4
    numerical = np.array([
        (plus.reference_energy() - minus.reference_energy()) / (2 * step)
        for plus, minus in displaced_points(mf, step=step)
    ])
    np.testing.assert_allclose(analytic, numerical, atol=1e-5, rtol=0)
    assert np.max(np.abs(cp.asnumpy(correction))) > 1e-4


def test_compute_frame_fuses_selected_reference_zvector():
    td = make_td(make_reference())
    separate = td.Gradients().set(
        cphf_conv_tol=1e-11, verbose=0,
    ).kernel(state=1)

    with mock.patch.object(
        ReferenceGradients,
        '_solve_z',
        side_effect=AssertionError(
            'compute_frame must use the fused NTTDA adjoint solve'
        ),
    ):
        frame = compute_frame(
            td, active_state=1, cphf_conv_tol=1e-11,
        )

    np.testing.assert_allclose(frame['grad'], separate, atol=5e-9, rtol=0)
    stats = td._nttda_frame_stats['reference_gradient']
    assert stats['calls'] == 1
    assert stats['fused']
    assert stats['z_b_backend'] == 'fused_nttda'
    assert stats['z_solver']['converged']


def test_compute_frame_batches_multiple_gradients_and_nac():
    td = make_td(make_reference())
    gradients = {
        state: td.Gradients().set(
            cphf_conv_tol=1e-11, verbose=0,
        ).kernel(state=state)
        for state in (1, 2, 3)
    }
    expected_nac = td.NAC().set(
        cphf_conv_tol=1e-11, ediff=True, verbose=0,
    ).kernel(state_I=1, state_J=3, use_etfs=True)

    with mock.patch.object(
        ReferenceGradients,
        '_solve_z',
        side_effect=AssertionError(
            'compute_frame must use the fused NTTDA adjoint solve'
        ),
    ):
        frame = compute_frame(
            td,
            active_state=2,
            gradient_states=(1, 2, 3),
            nac_pairs=((1, 3),),
            cphf_conv_tol=1e-11,
        )

    assert frame['grad'] is frame['gradients'][2]
    for state, expected in gradients.items():
        np.testing.assert_allclose(
            frame['gradients'][state], expected, atol=5e-9, rtol=0,
        )
    np.testing.assert_allclose(
        frame['nac'][(1, 3)], expected_nac, atol=5e-9, rtol=0,
    )
    stats = td._nttda_frame_stats
    assert stats['gradient_states'] == (1, 2, 3)
    assert stats['gradient_count'] == 3
    assert stats['zvector_batch_width'] == 4
    assert stats['xc_response_channels'] == 5
    assert stats['xc_fockz_tasks'] == 4
    assert stats['reference_gradient']['calls'] == 1
    assert stats['reference_gradient']['fused']

    with pytest.raises(ValueError, match='include active_state'):
        compute_frame(td, active_state=2, gradient_states=(1, 3))
    with pytest.raises(ValueError, match='unique'):
        compute_frame(td, active_state=2, gradient_states=(1, 2, 2))


@pytest.mark.parametrize('delta_s', [-1, 0])
@pytest.mark.parametrize('xc', ['PBE', 'B3LYP', 'M06-2X', 'CAM-B3LYP'])
def test_df_total_gradient_matches_selected_energy(xc, delta_s):
    mf = make_reference(xc)
    td = make_td(mf, delta_s)
    gradient = td.Gradients().set(cphf_conv_tol=1e-11, verbose=0)
    reference = gradient.kernel(state=0, atmlst=[1])[0]
    total = gradient.kernel(state=1, atmlst=[1])[0]
    step = 3e-4
    numerical = []
    for plus, minus in displaced_points(mf, step=step):
        values = []
        for point in (plus, minus):
            point_td = make_td(point, delta_s)
            values.append(np.array([point.reference_energy(), point_td.e_tot[0]]))
        numerical.append((values[0] - values[1]) / (2 * step))
    np.testing.assert_allclose(
        np.array([reference, total]), np.array(numerical).T, atol=1e-5, rtol=0,
    )
    assert gradient.reference_z_solver_diagnostics['converged']


def test_df_nac_wavefunction_difference_and_joint_frame():
    from gpu4pyscf.nac.nttda import _align_displaced_roots, awf_overlap

    mf = make_reference()
    td = make_td(mf)
    original = td
    step = 3e-4
    numerical = []
    for plus, minus in displaced_points(mf, step=step):
        plus_td = make_td(plus)
        minus_td = make_td(minus)
        for point in (plus_td, minus_td):
            _align_displaced_roots(original, point, 0.8, required=(0, 1))
        forward = awf_overlap(minus_td, minus_td.xy[0], plus_td, plus_td.xy[1])
        backward = awf_overlap(plus_td, plus_td.xy[0], minus_td, minus_td.xy[1])
        numerical.append(np.real(forward - backward) / (4 * step))

    nac_driver = td.NAC().set(cphf_conv_tol=1e-11, ediff=True, verbose=0)
    full_nac = nac_driver.kernel(state_I=1, state_J=2, use_etfs=False)
    np.testing.assert_allclose(full_nac[1], numerical, atol=2e-5, rtol=0)
    reverse = nac_driver.kernel(state_I=2, state_J=1, use_etfs=False)
    np.testing.assert_allclose(full_nac, -reverse, atol=1e-9, rtol=0)

    separate_gradient = td.Gradients().set(
        cphf_conv_tol=1e-11, verbose=0,
    ).kernel(state=1)
    for use_etfs in (False, True):
        separate_nac = nac_driver.kernel(state_I=1, state_J=2, use_etfs=use_etfs)
        with mock.patch.object(
            ReferenceGradients,
            '_solve_z',
            side_effect=AssertionError(
                'compute_frame must use the fused NTTDA adjoint solve'
            ),
        ):
            frame = compute_frame(
                td, active_state=1, nac_pairs=((1, 2),),
                cphf_conv_tol=1e-11, use_etfs=use_etfs,
            )
        np.testing.assert_allclose(frame['grad'], separate_gradient, atol=5e-9, rtol=0)
        np.testing.assert_allclose(frame['nac'][(1, 2)], separate_nac, atol=5e-9, rtol=0)
        stats = td._nttda_frame_stats['reference_gradient']
        assert stats['calls'] == 1
        assert stats['fused']
        assert stats['z_b_backend'] == 'fused_nttda'
        assert stats['z_solver']['converged']


def test_public_finite_differences_preserve_df_reference():
    td = make_td(make_reference())
    gradient = td.Gradients().set(fixed_grid=True, cphf_conv_tol=1e-11, verbose=0)
    analytic = gradient.kernel(state=1, atmlst=[1])
    numerical = gradient.kernel(state=1, atmlst=[1], method='finite_diff', step=3e-4)
    np.testing.assert_allclose(analytic, numerical, atol=1e-5, rtol=0)
    coupling = td.NAC().set(fixed_grid=True, cphf_conv_tol=1e-11, verbose=0)
    analytic_nac = coupling.kernel(
        state_I=1, state_J=2, atmlst=[1], ediff=True, use_etfs=False,
    )
    numerical_nac = coupling.finite_difference(state_I=1, state_J=2, atmlst=[1], step=3e-4)
    np.testing.assert_allclose(analytic_nac, numerical_nac, atol=2e-5, rtol=0)
