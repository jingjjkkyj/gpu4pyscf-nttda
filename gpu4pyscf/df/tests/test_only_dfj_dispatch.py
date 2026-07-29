# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import types
import unittest

from gpu4pyscf.df.df_jk import _DFHF


class KnownValues(unittest.TestCase):
    def test_get_k_obeys_only_dfj_get_jk_dispatch(self):
        calls = []
        fake = types.SimpleNamespace(
            direct_scf_tol=1e-12,
            with_df=types.SimpleNamespace(
                get_jk=lambda *_args, **_kwargs: (None, "df-k"),
            ),
        )

        def get_jk(_self, mol=None, dm=None, hermi=1,
                   with_j=True, with_k=True, omega=None):
            calls.append((with_j, with_k, omega))
            return None, "conventional-k"

        fake.get_jk = types.MethodType(get_jk, fake)
        result = _DFHF.get_k(
            fake, dm="density", hermi=0, omega=0.25,
        )

        self.assertEqual(result, "conventional-k")
        self.assertEqual(calls, [(False, True, 0.25)])


if __name__ == "__main__":
    unittest.main()
