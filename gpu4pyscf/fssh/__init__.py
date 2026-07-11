# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fewest-switches surface hopping drivers.

The nuclear dynamics stays on the CPU.  Electronic energies, gradients, and
nonadiabatic couplings are evaluated by GPU4PySCF drivers.
"""

from .fssh import FSSH, PES, h5_to_xyz
from .fssh_sf import FSSH_SF
from .wigner_sampling import wigner, wigner_samples

FSSH_SFTDA = FSSH_SF
FSSH_SFTDDFT = FSSH_SF

__all__ = [
    "FSSH",
    "FSSH_SF",
    "FSSH_SFTDA",
    "FSSH_SFTDDFT",
    "PES",
    "h5_to_xyz",
    "wigner",
    "wigner_samples",
]
