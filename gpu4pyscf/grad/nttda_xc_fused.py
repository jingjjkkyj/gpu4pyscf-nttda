"""Compiled contractions for the NTTDA XC response hot path."""

import cupy as cp
import numpy as np

from gpu4pyscf.lib.cupy_helper import add_sparse


_CUDA_SOURCE = r"""
extern "C" {

__device__ __forceinline__ double pair_value(
        const double *p, int row, int col, int ngrids, int grid) {
    return p[((row * 4 + col) * ngrids) + grid];
}

__device__ __forceinline__ double kernel_value(
        const double *kernel, int row, int col, int features,
        int ngrids, int grid) {
    return kernel[((row * features + col) * ngrids) + grid];
}

__device__ double pair_potential(
        const double *kernel, const double *pair, int row, int col,
        int features, int ngrids, int grid) {
    const double p00 = pair_value(pair, 0, 0, ngrids, grid);
    if (row == 0 && col == 0) {
        double value = 0.0;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                value += kernel_value(
                    kernel, i, j, features, ngrids, grid
                ) * pair_value(pair, i, j, ngrids, grid);
        return value;
    }
    if (row > 0 && row < 4 && col == 0) {
        double value = kernel_value(
            kernel, row, 0, features, ngrids, grid
        ) * p00;
        for (int j = 1; j < 4; ++j)
            value += kernel_value(
                kernel, row, j, features, ngrids, grid
            ) * pair_value(pair, 0, j, ngrids, grid);
        if (features == 5) {
            value += 0.5 * kernel_value(
                kernel, 4, 0, features, ngrids, grid
            ) * pair_value(pair, row, 0, ngrids, grid);
            for (int j = 1; j < 4; ++j)
                value += 0.5 * kernel_value(
                    kernel, 4, j, features, ngrids, grid
                ) * pair_value(pair, row, j, ngrids, grid);
        }
        return value;
    }
    if (row == 0 && col > 0 && col < 4) {
        double value = kernel_value(
            kernel, 0, col, features, ngrids, grid
        ) * p00;
        for (int i = 1; i < 4; ++i)
            value += kernel_value(
                kernel, i, col, features, ngrids, grid
            ) * pair_value(pair, i, 0, ngrids, grid);
        if (features == 5) {
            value += 0.5 * kernel_value(
                kernel, 0, 4, features, ngrids, grid
            ) * pair_value(pair, 0, col, ngrids, grid);
            for (int i = 1; i < 4; ++i)
                value += 0.5 * kernel_value(
                    kernel, i, 4, features, ngrids, grid
                ) * pair_value(pair, i, col, ngrids, grid);
        }
        return value;
    }
    if (row > 0 && row < 4 && col > 0 && col < 4) {
        double value = kernel_value(
            kernel, row, col, features, ngrids, grid
        ) * p00;
        if (features == 5) {
            value += 0.5 * kernel_value(
                kernel, row, 4, features, ngrids, grid
            ) * pair_value(pair, 0, col, ngrids, grid);
            value += 0.5 * kernel_value(
                kernel, 4, col, features, ngrids, grid
            ) * pair_value(pair, row, 0, ngrids, grid);
            value += 0.25 * kernel_value(
                kernel, 4, 4, features, ngrids, grid
            ) * pair_value(pair, row, col, ngrids, grid);
        }
        return value;
    }
    return 0.0;
}

__device__ double pair_cross(
        const double *left, const double *right, int row, int col,
        int features, int ngrids, int grid) {
    const double l00 = pair_value(left, 0, 0, ngrids, grid);
    const double r00 = pair_value(right, 0, 0, ngrids, grid);
    if (row == 0 && col == 0)
        return l00 * r00;
    if (row > 0 && row < 4 && col == 0)
        return l00 * pair_value(right, row, 0, ngrids, grid)
             + pair_value(left, row, 0, ngrids, grid) * r00;
    if (row == 0 && col > 0 && col < 4)
        return l00 * pair_value(right, 0, col, ngrids, grid)
             + pair_value(left, 0, col, ngrids, grid) * r00;
    if (row > 0 && row < 4 && col > 0 && col < 4)
        return l00 * pair_value(right, row, col, ngrids, grid)
             + pair_value(left, row, 0, ngrids, grid)
               * pair_value(right, 0, col, ngrids, grid)
             + pair_value(left, 0, col, ngrids, grid)
               * pair_value(right, row, 0, ngrids, grid)
             + pair_value(left, row, col, ngrids, grid) * r00;
    if (features != 5)
        return 0.0;
    if (row == 4 && col == 0) {
        double value = 0.0;
        for (int i = 1; i < 4; ++i)
            value += pair_value(left, i, 0, ngrids, grid)
                   * pair_value(right, i, 0, ngrids, grid);
        return 0.5 * value;
    }
    if (row == 0 && col == 4) {
        double value = 0.0;
        for (int j = 1; j < 4; ++j)
            value += pair_value(left, 0, j, ngrids, grid)
                   * pair_value(right, 0, j, ngrids, grid);
        return 0.5 * value;
    }
    if (row == 4 && col > 0 && col < 4) {
        double value = 0.0;
        for (int i = 1; i < 4; ++i)
            value += pair_value(left, i, 0, ngrids, grid)
                   * pair_value(right, i, col, ngrids, grid)
                   + pair_value(left, i, col, ngrids, grid)
                   * pair_value(right, i, 0, ngrids, grid);
        return 0.5 * value;
    }
    if (row > 0 && row < 4 && col == 4) {
        double value = 0.0;
        for (int j = 1; j < 4; ++j)
            value += pair_value(left, 0, j, ngrids, grid)
                   * pair_value(right, row, j, ngrids, grid)
                   + pair_value(left, row, j, ngrids, grid)
                   * pair_value(right, 0, j, ngrids, grid);
        return 0.5 * value;
    }
    if (row == 4 && col == 4) {
        double value = 0.0;
        for (int i = 1; i < 4; ++i)
            for (int j = 1; j < 4; ++j)
                value += pair_value(left, i, j, ngrids, grid)
                       * pair_value(right, i, j, ngrids, grid);
        return 0.25 * value;
    }
    return 0.0;
}

__global__ void assemble_response_weights(
        const double *fref, const double *kalpha, const double *kbeta,
        const double *rho, const double *pairs, const double *grid_weights,
        const int *targets, const int *sources, const int *pair_lookup,
        const double *vref0, const double *vref1,
        double *ordinary, double *special,
        double *reference_alpha, double *reference_beta,
        int labels, int pair_count, int terms, int features, int ngrids) {
    const int grid = blockDim.x * blockIdx.x + threadIdx.x;
    if (grid >= ngrids)
        return;
    const double weight = grid_weights[grid];
    for (int term = 0; term < terms; ++term) {
        const int target = targets[term];
        const int source = sources[term];
        const double c0 = vref0[term];
        if (c0 != 0.0) {
            for (int x = 0; x < features; ++x) {
                double left = 0.0;
                double right = 0.0;
                for (int y = 0; y < features; ++y) {
                    const double fxy = fref[
                        ((x * features + y) * ngrids) + grid
                    ];
                    left += fxy * rho[
                        ((source * features + y) * ngrids) + grid
                    ];
                    right += fref[
                        ((y * features + x) * ngrids) + grid
                    ] * rho[
                        ((target * features + y) * ngrids) + grid
                    ];
                }
                ordinary[
                    ((target * features + x) * ngrids) + grid
                ] += weight * c0 * left;
                ordinary[
                    ((source * features + x) * ngrids) + grid
                ] += weight * c0 * right;
            }
            for (int z = 0; z < features; ++z) {
                double alpha = 0.0;
                double beta = 0.0;
                for (int x = 0; x < features; ++x) {
                    const double rx = rho[
                        ((target * features + x) * ngrids) + grid
                    ];
                    for (int y = 0; y < features; ++y) {
                        const double product = rx * rho[
                            ((source * features + y) * ngrids) + grid
                        ];
                        alpha += product * kalpha[
                            (((x * features + y) * features + z)
                             * ngrids) + grid
                        ];
                        beta += product * kbeta[
                            (((x * features + y) * features + z)
                             * ngrids) + grid
                        ];
                    }
                }
                reference_alpha[z * ngrids + grid] += weight * c0 * alpha;
                reference_beta[z * ngrids + grid] += weight * c0 * beta;
            }
        }
        const double c1 = vref1[term];
        if (c1 == 0.0)
            continue;
        const int target_pair = pair_lookup[target];
        const int source_pair = pair_lookup[source];
        const double *target_values = pairs
            + target_pair * 16 * ngrids;
        const double *source_values = pairs
            + source_pair * 16 * ngrids;
        for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
                special[
                    (((target_pair * 4 + x) * 4 + y) * ngrids) + grid
                ] += weight * c1 * pair_potential(
                    fref, source_values, x, y, features, ngrids, grid
                );
                special[
                    (((source_pair * 4 + x) * 4 + y) * ngrids) + grid
                ] += weight * c1 * pair_potential(
                    fref, target_values, x, y, features, ngrids, grid
                );
            }
        }
        for (int z = 0; z < features; ++z) {
            double alpha = 0.0;
            double beta = 0.0;
            for (int x = 0; x < features; ++x) {
                for (int y = 0; y < features; ++y) {
                    const double cross = pair_cross(
                        target_values, source_values, x, y,
                        features, ngrids, grid
                    );
                    alpha += cross * kalpha[
                        (((x * features + y) * features + z)
                         * ngrids) + grid
                    ];
                    beta += cross * kbeta[
                        (((x * features + y) * features + z)
                         * ngrids) + grid
                    ];
                }
            }
            reference_alpha[z * ngrids + grid] += weight * c1 * alpha;
            reference_beta[z * ngrids + grid] += weight * c1 * beta;
        }
    }
}

__global__ void contract_response_centers(
        const double *ao, const double *contracted,
        const double *contracted_t, const double *weights,
        const double *pair_products, const double *pair_products_t,
        const double *pair_weights, const int *atom_ids,
        const int *requested_atoms, double *output,
        int nao, int ngrids, int densities, int features,
        int pair_count, int requested_count) {
    const int task = blockIdx.x;
    const int total = nao * 3;
    if (task >= total)
        return;
    const int ao_index = task / 3;
    const int xyz = task - ao_index * 3;
    const int derivative_indices[3][4] = {
        {1, 4, 5, 6}, {2, 5, 7, 8}, {3, 6, 8, 9}
    };
    double value = 0.0;
    for (int grid = threadIdx.x; grid < ngrids; grid += blockDim.x) {
        double adjoint[4] = {0.0, 0.0, 0.0, 0.0};
        for (int density = 0; density < densities; ++density) {
            const long base = (
                ((long)density * 4 * nao + ao_index) * ngrids + grid
            );
            const double both0 = contracted[base] + contracted_t[base];
            adjoint[0] += weights[
                ((density * features) * ngrids) + grid
            ] * both0;
            for (int feature = 1; feature < 4; ++feature) {
                const long offset = base + (long)feature * nao * ngrids;
                const double both = contracted[offset] + contracted_t[offset];
                adjoint[0] += weights[
                    ((density * features + feature) * ngrids) + grid
                ] * both;
                adjoint[feature] += weights[
                    ((density * features + feature) * ngrids) + grid
                ] * both0;
                if (features == 5)
                    adjoint[feature] += 0.5 * weights[
                        ((density * features + 4) * ngrids) + grid
                    ] * both;
            }
        }
        for (int pair = 0; pair < pair_count; ++pair) {
            for (int feature = 0; feature < 4; ++feature) {
                for (int other = 0; other < 4; ++other) {
                    const long product_offset = (
                        (((long)pair * 4 + other) * nao + ao_index)
                        * ngrids + grid
                    );
                    adjoint[feature] += pair_weights[
                        (((pair * 4 + feature) * 4 + other)
                         * ngrids) + grid
                    ] * pair_products_t[product_offset];
                    adjoint[feature] += pair_weights[
                        (((pair * 4 + other) * 4 + feature)
                         * ngrids) + grid
                    ] * pair_products[product_offset];
                }
            }
        }
        for (int feature = 0; feature < 4; ++feature) {
            const int derivative = derivative_indices[xyz][feature];
            value -= ao[
                ((long)derivative * nao + ao_index) * ngrids + grid
            ] * adjoint[feature];
        }
    }
    extern __shared__ double partial[];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int atom = atom_ids[ao_index];
        for (int requested = 0; requested < requested_count; ++requested)
            if (requested_atoms[requested] == atom)
                atomicAdd(
                    output + requested * 3 + xyz, partial[0]
                );
    }
}

}
"""


_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        _MODULE = cp.RawModule(
            code=_CUDA_SOURCE,
            options=("--std=c++11",),
            name_expressions=(
                "assemble_response_weights",
                "contract_response_centers",
            ),
        )
    return _MODULE


def _float64_c(array):
    array = cp.asarray(array)
    if array.dtype != cp.float64:
        raise TypeError("fused NTTDA XC contractions require float64 arrays")
    return cp.ascontiguousarray(array)


def assemble_response_weights(
        fref, kref_alpha, kref_beta, rho, pair_values, grid_weights,
        targets, sources, pair_lookup, vref0, vref1):
    """Build all weighted response tensors in one compiled grid pass."""
    fref = _float64_c(fref)
    kref_alpha = _float64_c(kref_alpha)
    kref_beta = _float64_c(kref_beta)
    rho = _float64_c(rho)
    pair_values = _float64_c(pair_values)
    grid_weights = _float64_c(grid_weights)
    labels, features, ngrids = rho.shape
    pair_count = len(pair_values)
    terms = len(targets)
    if (
        features not in (4, 5)
        or fref.shape != (features, features, ngrids)
        or kref_alpha.shape != (features, features, features, ngrids)
        or kref_beta.shape != kref_alpha.shape
        or pair_values.shape != (pair_count, 4, 4, ngrids)
        or grid_weights.shape != (ngrids,)
    ):
        raise ValueError("inconsistent fused NTTDA response inputs")
    targets = cp.ascontiguousarray(cp.asarray(targets, dtype=cp.int32))
    sources = cp.ascontiguousarray(cp.asarray(sources, dtype=cp.int32))
    pair_lookup = cp.ascontiguousarray(
        cp.asarray(pair_lookup, dtype=cp.int32),
    )
    vref0 = _float64_c(vref0)
    vref1 = _float64_c(vref1)
    if (
        targets.shape != (terms,)
        or sources.shape != (terms,)
        or pair_lookup.shape != (labels,)
        or vref0.shape != (terms,)
        or vref1.shape != (terms,)
    ):
        raise ValueError("inconsistent fused NTTDA response ledger")
    ordinary = cp.zeros((labels, features, ngrids), dtype=cp.float64)
    special = cp.zeros((pair_count, 4, 4, ngrids), dtype=cp.float64)
    reference_alpha = cp.zeros((features, ngrids), dtype=cp.float64)
    reference_beta = cp.zeros_like(reference_alpha)
    if ngrids:
        kernel = _module().get_function("assemble_response_weights")
        kernel(
            ((ngrids + 127) // 128,), (128,),
            (
                fref, kref_alpha, kref_beta, rho, pair_values, grid_weights,
                targets, sources, pair_lookup, vref0, vref1,
                ordinary, special, reference_alpha, reference_beta,
                np.int32(labels), np.int32(pair_count), np.int32(terms),
                np.int32(features), np.int32(ngrids),
            ),
        )
    return ordinary, special, reference_alpha, reference_beta


def add_response_matrices(
        outputs, reference_alpha_output, reference_beta_output, ao,
        ordinary, special, reference_alpha, reference_beta,
        pair_indices, indices):
    """Accumulate all response AO matrices through batched GEMMs."""
    all_weights = cp.concatenate((
        ordinary, reference_alpha[None], reference_beta[None],
    ))
    scaled_weights = all_weights.copy()
    scaled_weights[:, 0] *= 0.5
    scaled = cp.einsum(
        "fag,nfg->nag", ao[:4], scaled_weights[:, :4],
        optimize=True,
    )
    matrices = cp.matmul(
        ao[0][None], cp.ascontiguousarray(scaled.transpose(0, 2, 1)),
    )
    matrices += matrices.transpose(0, 2, 1)
    if all_weights.shape[1] == 5:
        tau_scaled = 0.5 * (
            ao[1:4][None] * all_weights[:, None, 4, None, :]
        )
        matrices += cp.matmul(
            ao[1:4][None],
            cp.ascontiguousarray(tau_scaled.transpose(0, 1, 3, 2)),
        ).sum(axis=1)
    destinations = tuple(outputs) + (
        reference_alpha_output, reference_beta_output,
    )
    for destination, matrix in zip(destinations, matrices):
        add_sparse(destination, matrix, indices)

    if len(special):
        pair_scaled = cp.einsum(
            "rag,nlrg->nlag", ao[:4], special, optimize=True,
        )
        pair_matrices = cp.matmul(
            ao[:4][None],
            cp.ascontiguousarray(pair_scaled.transpose(0, 1, 3, 2)),
        ).sum(axis=1)
        for pair_index, matrix in zip(pair_indices, pair_matrices):
            add_sparse(outputs[pair_index], matrix, indices)


def contract_response_centers(
        ao, contracted, contracted_t, weights, pair_products,
        pair_products_t, pair_weights, atom_ids, atmlst):
    """Contract weighted response derivatives with one compiled launch."""
    ao = _float64_c(ao)
    contracted = _float64_c(contracted)
    contracted_t = _float64_c(contracted_t)
    weights = _float64_c(weights)
    pair_products = _float64_c(pair_products)
    pair_products_t = _float64_c(pair_products_t)
    pair_weights = _float64_c(pair_weights)
    atom_ids = cp.ascontiguousarray(cp.asarray(atom_ids, dtype=cp.int32))
    requested = cp.ascontiguousarray(cp.asarray(atmlst, dtype=cp.int32))
    densities, features, ngrids = weights.shape
    nao = ao.shape[1]
    pair_count = len(pair_weights)
    if (
        features not in (4, 5)
        or ao.shape != (10, nao, ngrids)
        or contracted.shape != (densities, 4, nao, ngrids)
        or contracted_t.shape != contracted.shape
        or pair_products.shape != (pair_count, 4, nao, ngrids)
        or pair_products_t.shape != pair_products.shape
        or pair_weights.shape != (pair_count, 4, 4, ngrids)
        or atom_ids.shape != (nao,)
    ):
        raise ValueError("inconsistent fused NTTDA center inputs")
    output = cp.zeros((len(requested), 3), dtype=cp.float64)
    total = nao * 3
    if total and ngrids and len(requested):
        kernel = _module().get_function("contract_response_centers")
        kernel(
            (total,), (256,),
            (
                ao, contracted, contracted_t, weights,
                pair_products, pair_products_t, pair_weights,
                atom_ids, requested, output,
                np.int32(nao), np.int32(ngrids), np.int32(densities),
                np.int32(features), np.int32(pair_count),
                np.int32(len(requested)),
            ),
            shared_mem=256 * 8,
        )
    return output
