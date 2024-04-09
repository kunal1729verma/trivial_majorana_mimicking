import numpy as np
import os

import conductance_project.get_system_SC_smooth as gs
from utils import systems as systems
from utils.cond_funcs import ConductanceStandard
# again need to ask gokul how exactly these imports work and make notes for it.

# Setting up parameters
def main(params):

    E = 1.5   # abs(E_max) to decide the range of energies for conductance spectrum.
    res = 50  # Range of energies and Δz for spectrum plots.
    k = 80  # no of eigenvalues/eigenvectors calculated.

    energies = np.linspace(-E * params["Δ_0"], E * params["Δ_0"], res)
    Δz_range = np.linspace(0, 3.0 * params["Δ_0"], res)

    save_path = f"{os.path.dirname(__file__)}/build/{params.pop('timestamp')}/"
    systems.Transport1D.spectrum(ConductanceStandard, energies, Δz_range, k, params, gs.scatter_def, gs.leads_def, save_path, espectrum = True, wavefn = False)