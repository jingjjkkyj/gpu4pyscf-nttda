# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Non-relativistic ROKS analytical nuclear gradients'''

from pyscf.grad import uks as uks_grad_cpu
from gpu4pyscf.grad import rohf as rohf_grad
from gpu4pyscf.grad import uks as uks_grad


class Gradients(rohf_grad.Gradients):
    from gpu4pyscf.lib.utils import to_gpu, device

    grid_response = uks_grad_cpu.Gradients.grid_response

    _keys = uks_grad_cpu.Gradients._keys

    def __init__(self, mf):
        rohf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    # XC energy derivative + hybrid-scaled J/K per-atom derivatives; reads the
    # stacked mo_coeff/mo_occ tags that rohf_grad.grad_elec puts on dm0.
    energy_ee = uks_grad.energy_ee


Grad = Gradients
