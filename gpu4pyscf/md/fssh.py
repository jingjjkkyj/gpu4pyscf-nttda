# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compatibility imports for the historical ``gpu4pyscf.md.fssh`` path."""

from gpu4pyscf.fssh.fssh import FS2AUTIME, FSSH, PES, h5_to_xyz

__all__ = ["FS2AUTIME", "FSSH", "PES", "h5_to_xyz"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert an HDF5 trajectory file to XYZ format."
    )
    parser.add_argument("h5file", help="Input HDF5 trajectory")
    parser.add_argument("trajectory_file", help="Output XYZ trajectory")
    args = parser.parse_args()
    h5_to_xyz(args.h5file, args.trajectory_file)
