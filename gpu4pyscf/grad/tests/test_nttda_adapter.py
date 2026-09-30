# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from gpu4pyscf.grad import nttda_ledger as nttda
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











if __name__ == "__main__":
    unittest.main()


def test_host_and_device_density_views_have_stable_distinct_keys():
    import cupy as cp
    host = np.arange(9.).reshape(3, 3)
    device = cp.asarray(host)
    assert nttda._density_cache_key(host) == nttda._density_cache_key(host.view())
    assert nttda._density_cache_key(device) == nttda._density_cache_key(device.view())
    assert nttda._density_cache_key(host) != nttda._density_cache_key(device)
    assert nttda._density_cache_key(host) != nttda._density_cache_key(host.T)
