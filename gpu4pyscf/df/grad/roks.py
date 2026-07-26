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

'''ROKS analytical nuclear gradients with density fitting.

Same thin-composition pattern as :mod:`gpu4pyscf.df.grad.uks`: the ROHF
spatial-orbital ``grad_elec`` (stacked mo tags, focka/fockb-projected
energy-weighted density) combined with the DF ``energy_ee`` (XC + DF J/K
per-atom derivatives including the auxiliary-basis response).
'''

from gpu4pyscf.df.grad import uhf as df_uhf_grad
from gpu4pyscf.df.grad import uks as df_uks_grad
from gpu4pyscf.grad import rohf as rohf_grad


class Gradients(rohf_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def __init__(self, mf):
        rohf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    grid_response = False

    energy_ee = df_uks_grad.energy_ee
    jk_energy_per_atom = df_uhf_grad.Gradients.jk_energy_per_atom


Grad = Gradients
