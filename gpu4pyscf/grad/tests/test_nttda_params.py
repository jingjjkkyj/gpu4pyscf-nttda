# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest import mock

import os

from gpu4pyscf.grad.nttda_params import (
    PARAMS, reload_params, get, UNVALIDATED_ON_A100,
)


class KnownValues(unittest.TestCase):
    def test_defaults_are_calibrated_for_rtx4060(self):
        """Default params must reproduce the values used for all baselines."""
        self.assertAlmostEqual(PARAMS['df_mem_fraction'], 0.5)
        self.assertAlmostEqual(PARAMS['df_batch_factor'], 1.0)
        self.assertAlmostEqual(PARAMS['df_blk_factor'], 1.0)
        self.assertEqual(PARAMS['dm_block'], 7)
        self.assertEqual(PARAMS['df_compressed_backend'], 'legacy')
        self.assertFalse(PARAMS['df_compressed_profile'])
        self.assertEqual(PARAMS['df_output_backend'], 'legacy')
        self.assertIsNone(PARAMS['cphf_max_cycle'])
        self.assertEqual(PARAMS['davidson_max_subspace'], 12)
        self.assertEqual(PARAMS['contract_backend'], 'cupy')
        self.assertFalse(PARAMS['finish_profile'])

    def test_a100_unvalidated_flag(self):
        """A100 paths must be marked unvalidated until a sweep is done."""
        self.assertTrue(UNVALIDATED_ON_A100)
        self.assertTrue(PARAMS['unvalidated_on_A100'])

    def test_env_override_takes_effect(self):
        """reload_params picks up env vars set after import time."""
        env = {
            'NTTDA_DF_MEM_FRACTION': '0.7',
            'NTTDA_DM_BLOCK': '12',
            'NTTDA_DF_COMPRESSED_BACKEND': 'rank_batched',
            'NTTDA_DF_COMPRESSED_PROFILE': '1',
            'NTTDA_DF_OUTPUT_BACKEND': 'slot_aware',
            'NTTDA_DAVIDSON_MAX_SUBSPACE': '20',
            'NTTDA_CONTRACT_BACKEND': 'cutensor',
            'NTTDA_CPHF_MAX_CYCLE': '50',
            'NTTDA_FINISH_PROFILE': '1',
        }
        with mock.patch.dict(os.environ, env, clear=False):
            new_params = reload_params()
        self.assertAlmostEqual(new_params['df_mem_fraction'], 0.7)
        self.assertEqual(new_params['dm_block'], 12)
        self.assertEqual(new_params['df_compressed_backend'], 'rank_batched')
        self.assertTrue(new_params['df_compressed_profile'])
        self.assertEqual(new_params['df_output_backend'], 'slot_aware')
        self.assertEqual(new_params['davidson_max_subspace'], 20)
        self.assertEqual(new_params['contract_backend'], 'cutensor')
        self.assertEqual(new_params['cphf_max_cycle'], 50)
        self.assertTrue(new_params['finish_profile'])
        # Restore defaults for subsequent tests
        reload_params()

    def test_invalid_float_raises(self):
        with mock.patch.dict(os.environ, {'NTTDA_DF_MEM_FRACTION': 'abc'}):
            with self.assertRaises(ValueError):
                reload_params()

    def test_out_of_range_float_raises(self):
        with mock.patch.dict(os.environ, {'NTTDA_DF_MEM_FRACTION': '1.5'}):
            with self.assertRaises(ValueError):
                reload_params()

    def test_out_of_range_int_raises(self):
        with mock.patch.dict(os.environ, {'NTTDA_DM_BLOCK': '0'}):
            with self.assertRaises(ValueError):
                reload_params()

    def test_invalid_backend_raises(self):
        with mock.patch.dict(os.environ, {'NTTDA_CONTRACT_BACKEND': 'numpy'}):
            with self.assertRaises(ValueError):
                reload_params()
        with mock.patch.dict(
                os.environ, {'NTTDA_DF_COMPRESSED_BACKEND': 'padded'}):
            with self.assertRaises(ValueError):
                reload_params()
        with mock.patch.dict(
                os.environ, {'NTTDA_DF_COMPRESSED_PROFILE': 'yes'}):
            with self.assertRaises(ValueError):
                reload_params()
        with mock.patch.dict(
                os.environ, {'NTTDA_DF_OUTPUT_BACKEND': 'global'}):
            with self.assertRaises(ValueError):
                reload_params()

    def test_get_returns_default_for_unknown_key(self):
        self.assertIsNone(get('nonexistent_key'))
        self.assertEqual(get('nonexistent_key', 42), 42)

    def test_reload_restores_defaults(self):
        """After clearing env, reload_params restores the calibrated defaults."""
        with mock.patch.dict(os.environ, {'NTTDA_DM_BLOCK': '16'}, clear=False):
            reload_params()
        os.environ.pop('NTTDA_DM_BLOCK', None)
        reload_params()
        self.assertEqual(PARAMS['dm_block'], 7)


if __name__ == "__main__":
    unittest.main()
