"""Weighted fixed-grid AO-center derivatives, without a per-atom grid loop.

The caller supplies D@AO / D.T@AO and feature weights from the unchanged
formula layer. Only the order of linear contractions changes. ``xp`` is
CuPy in production and NumPy for independent, CPU-runnable algebra tests.
"""


def fockz_weights(fref, rho_open, rho_pz, alpha_weights, beta_weights, grid_weights, task, *, xp):
    """Weights of (Pz_task, open, alpha, beta); already grid-weighted."""
    pz = 0.5 * xp.einsum('xyg,yg->xg', fref, rho_open) * grid_weights
    op = 0.5 * xp.einsum('xg,xyg->yg', rho_pz[task], fref) * grid_weights
    return xp.stack((pz, op, alpha_weights[task], beta_weights[task]))


def postz_weights(vxc, fxc, probe_rho, task, *, xp):
    """Weights of (reference alpha,beta, probe alpha,beta), unweighted."""
    reference = xp.einsum('byg,axbyg->axg', probe_rho[task], fxc)
    return xp.concatenate((reference, vxc))


def plan_tiles(nao, ngrids, ndensity, npair, natoms, budget):
    """Conservative budget for explicit helper arrays, excluding inputs.

    Library scratch space, cached arrays and the small final output are not
    a hard GPU allocation limit. All large contractions below are binary;
    no einsum path can form a task-by-density-by-AO-by-grid intermediate.
    """
    cell_bytes = 8 * (64 + 24 * ndensity + 32 * npair)
    atom_bytes = 8 * (natoms + 6)
    if budget < cell_bytes + atom_bytes:
        raise ValueError('XC direct memory budget cannot hold one AO/grid tile')
    ao_tile = min(nao, 128, budget // (cell_bytes + atom_bytes))
    grid_tile = min(ngrids, (budget // ao_tile - atom_bytes) // cell_bytes)
    return ao_tile, grid_tile, ao_tile * (grid_tile * cell_bytes + atom_bytes)


def contract_centers(
    ao,
    contracted,
    contracted_t,
    weights,
    atom_ids,
    atmlst,
    *,
    xp,
    grid_weights=None,
    pair=None,
    density_indices=None,
    max_memory_bytes=256 * 1024**2,
    stats=None,
):
    """Return (len(atmlst),3) weighted derivatives for a single output task.

    AO: (10,a,g); products: (n,4,a,g); weights: (selected_n,4|5,g).
    ``pair`` optionally supplies (products, transpose_products, weights),
    with pair weights (npair,4,4,g). Ordinary and pair weights are multiplied
    by grid_weights exactly once, or are already weighted when it is None.
    ``density_indices`` selects product rows without copying full blocks.
    """
    atmlst = tuple(atmlst)
    nao, ngrids = ao.shape[1:]
    ndensity, features, wg = weights.shape
    selected = tuple(range(contracted.shape[0])) if density_indices is None else tuple(density_indices)
    if (
        features not in (4, 5)
        or wg != ngrids
        or len(selected) != ndensity
        or ao.shape[0] < 10
        or contracted.shape != contracted_t.shape
        or contracted.shape[1:] != (4, nao, ngrids)
        or any(i < 0 or i >= len(contracted) for i in selected)
        or len(atom_ids) != nao
    ):
        raise ValueError('inconsistent AO derivative workspace/weights')
    if grid_weights is not None and grid_weights.shape != (ngrids,):
        raise ValueError('grid weights must match the AO block')
    if any(array.dtype.kind != 'f' or array.dtype.itemsize != 8 for array in (ao, contracted, contracted_t, weights)):
        raise TypeError('AO reduction currently requires real float64 arrays')
    npair = 0 if pair is None else len(pair[0])
    if pair is not None and (
        pair[0].shape != (npair, 4, nao, ngrids)
        or pair[1].shape != pair[0].shape
        or pair[2].shape != (npair, 4, 4, ngrids)
    ):
        raise ValueError('inconsistent pair derivative workspace/weights')
    extra_arrays = (() if pair is None else pair) + (() if grid_weights is None else (grid_weights,))
    if any(a.dtype.kind != 'f' or a.dtype.itemsize != 8 for a in extra_arrays):
        raise TypeError('AO reduction weights and pair products must be real float64')
    output = xp.zeros((len(atmlst), 3), dtype=ao.dtype)
    if not atmlst or not nao or not ngrids:
        return output
    ab, gb, estimate = plan_tiles(nao, ngrids, ndensity, npair, len(atmlst), int(max_memory_bytes))
    if stats is not None:
        stats['estimated_tile_bytes'] = max(stats.get('estimated_tile_bytes', 0), estimate)
        stats['ao_tile'] = ab
        stats['grid_tile'] = gb
        stats['calls'] = stats.get('calls', 0) + 1
    requested = xp.asarray(atmlst)
    derivative_indices = xp.asarray([[1, 4, 5, 6], [2, 5, 7, 8], [3, 6, 8, 9]])
    rows = None if density_indices is None else xp.asarray(selected, dtype='int32')
    for a0 in range(0, nao, ab):
        aa = slice(a0, min(a0 + ab, nao))
        membership = (requested[:, None] == atom_ids[None, aa]).astype(ao.dtype)
        for g0 in range(0, ngrids, gb):
            gg = slice(g0, min(g0 + gb, ngrids))
            c = contracted[:, :, aa, gg]
            ct = contracted_t[:, :, aa, gg]
            if rows is not None:
                c, ct = c[rows], ct[rows]
            w = weights[:, :, gg]
            if grid_weights is not None:
                w = w * grid_weights[gg]
            # Coefficient of delta AO[f,a,g]. Both nonsymmetric density
            # orientations are retained. GGA gradient features have 4 terms;
            # the MGGA tau feature adds one half from each orientation.
            both = c + ct
            adjoint = xp.zeros((4, aa.stop - aa.start, gg.stop - gg.start), dtype=ao.dtype)
            adjoint[0] = xp.einsum('ng,nag->ag', w[:, 0], both[:, 0])
            adjoint[0] += xp.einsum('nfg,nfag->ag', w[:, 1:4], both[:, 1:4])
            adjoint[1:4] = xp.einsum('nfg,nag->fag', w[:, 1:4], both[:, 0])
            if features == 5:
                adjoint[1:4] += 0.5 * xp.einsum('ng,nfag->fag', w[:, 4], both[:, 1:4])
            if npair:
                pc, pt, pw = pair
                pw = pw[:, :, :, gg]
                if grid_weights is not None:
                    pw = pw * grid_weights[gg]
                adjoint += xp.einsum('nflg,nlag->fag', pw, pt[:, :, aa, gg])
                adjoint += xp.einsum('nlfg,nlag->fag', pw, pc[:, :, aa, gg])
            delta = -ao[derivative_indices, aa, gg]
            per_ao = xp.einsum('xfag,fag->ax', delta, adjoint)
            output += membership @ per_ao
            if stats is not None:
                stats['tiles'] = stats.get('tiles', 0) + 1
    return output
