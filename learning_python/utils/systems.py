import kwant
import numpy as np
import scipy
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib
import conductance_project.get_system_SC_smooth as gs

from pathlib import Path

class Transport1D: #(here?)
    """
    Basic functions for calculating conductances, energy eigenvalues, and associated wavefunctions (eigenvectors) of a (single) given system.
    """
    def __init__(self, scatter_def, leads_def, scatter_args, leads_args):
        self.syst = kwant.Builder()

        self.finite_syst = scatter_def(self.syst, **scatter_args)  # Scattering section of the system
        self.syst = leads_def(self.finite_syst, **leads_args)      # Left and right leads

    # Class Methods
    def eigenvv_system(self, k):
      	# k = The number of eigenvalues and eigenvectors desired. 
        ham_mat = self.finite_syst.finalized().hamiltonian_submatrix(sparse = True)
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(ham_mat, k=k, sigma=0, return_eigenvectors=True)
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
    
    def conductance(self, conductance_cls, energy, params={}):
        smatrix = kwant.smatrix(self.syst.finalized(), energy, params=params)
        
        G_LL = conductance_cls.G_LL_func(smatrix)
        G_LR = conductance_cls.G_LR_func(smatrix)
        G_RL = conductance_cls.G_RL_func(smatrix)
        G_RR = conductance_cls.G_RR_func(smatrix)

        return G_LL, G_LR, G_RL, G_RR
    
    @staticmethod
    def save_cond_spectrum(G, save_path):

        plt.imshow(G[0], interpolation='none',cmap='seismic', extent=[0, 3.0, -1.5, 1.5], aspect = 'auto')
        plt.xlabel(r'$\Delta_z/\Delta$', fontsize = 13)
        plt.ylabel(r'${eV}/\Delta$', fontsize = 13)
        clb = plt.colorbar()
        clb.ax.set_title(r'$G_{LL} \: [e^2/h]$')
        plt.savefig(f"{save_path}/G_LL.jpg" , dpi=(250), edgecolor='#f0f0f0', bbox_inches="tight",facecolor='white', transparent=False) 
        plt.clf()

        plt.imshow(G[3], interpolation='none',cmap='seismic', extent=[0, 3.0, -1.5, 1.5], aspect = 'auto')
        plt.xlabel(r'$\Delta_z/\Delta$', fontsize = 13)
        plt.ylabel(r'${eV}/\Delta$', fontsize = 13)
        clb = plt.colorbar()
        clb.ax.set_title(r'$G_{RR} \: [e^2/h]$')
        plt.savefig(f"{save_path}/G_RR.jpg" , dpi=(250), edgecolor='#f0f0f0', bbox_inches="tight",facecolor='white', transparent=False) 
        plt.clf()

        Norm1=matplotlib.colors.SymLogNorm(linthresh=10**(-3),  vmin=-1.6,vmax=1.6)
        Norm2=matplotlib.colors.CenteredNorm(vcenter=0)

        plt.imshow(G[1], interpolation='nearest',cmap='seismic',norm=Norm1, extent=[0, 3.0, -1.5, 1.5], aspect = 'auto')
        plt.xlabel(r'$\Delta_z/\Delta$', fontsize = 13)
        plt.ylabel(r'${eV}/\Delta$', fontsize = 13)
        clc = plt.colorbar()
        clc.ax.set_title(r'$G_{LR} \: [e^2/h]$')
        plt.savefig(f"{save_path}/G_LR.jpg" , dpi=(250), edgecolor='#f0f0f0', bbox_inches="tight",facecolor='white', transparent=False) 
        plt.clf()

        plt.imshow(G[2], interpolation='nearest',cmap='seismic',norm=Norm1, extent=[0, 3.0, -1.5, 1.5], aspect = 'auto')
        plt.xlabel(r'$\Delta_z/\Delta$', fontsize = 13)
        plt.ylabel(r'${eV}/\Delta$', fontsize = 13)
        clc = plt.colorbar()
        clc.ax.set_title(r'$G_{RL} \: [e^2/h]$')
        plt.savefig(f"{save_path}/G_RL.jpg" , dpi=(250), edgecolor='#f0f0f0', bbox_inches="tight",facecolor='white', transparent=False) 
        plt.clf()


    @staticmethod
    def save_energy_spectrum(eigenval_array, Δz_repeat, Δ_0, save_path):

        for i in range(len(eigenval_array)):
            plt.plot(Δz_repeat[i]/Δ_0, eigenval_array[i]/Δ_0, 'r.', markersize=1.2)
            plt.xlabel(r'${\Delta_z}/\Delta_0$', fontsize = 11)
            plt.ylabel(r'${eV}/\Delta_0$', fontsize = 11)
        
        plt.ylim([-1.5, 1.5])
        plt.savefig(f"{save_path}/energy_spectrum.jpg" , dpi=(250), edgecolor='#f0f0f0', bbox_inches="tight",facecolor='white', transparent=False) 
        plt.clf()


    @classmethod
    def spectrum(cls, conductance_cls, energies, Δz_range, k, params, scatter_def, leads_def, save_path, espectrum = False, wavefn = False):

        P = gs.ProfileFunctions(**params)
        leads_args = {'prof_fns': P}
        scatter_args = {'prof_fns': P}

        G_LL_array, G_LR_array, G_RL_array, G_RR_array = [], [], [], []
        if (espectrum == True):
            eigenval_array, eigenvec_array, Δz_repeat = [], [], []
            
        for Δz_val in Δz_range:
            # print(Δz_val/P.Δ_0)
            P.Δz = Δz_val
            Δz_repeat.append(np.full((k), Δz_val))

            T = cls(scatter_def, leads_def, scatter_args, leads_args)

            T.eigenvv_system(k)
            
            if (espectrum == True):
                eigenval_array.append(T.eigenvalues)
                eigenvec_array.append(T.eigenvectors)

            # Setting up system object
            LL, LR, RL, RR = [], [], [], []

            for energy in energies:
                G_LL, G_LR, G_RL, G_RR = T.conductance(conductance_cls, energy, {'prof_fns': P})
                LL.append(G_LL)
                LR.append(G_LR)
                RL.append(G_RL)
                RR.append(G_RR)

                # print(G_LL)

            G_LL_array.append(LL)
            G_LR_array.append(LR)
            G_RL_array.append(RL)
            G_RR_array.append(RR)

        G_LL_array = np.array(G_LL_array).T
        G_LR_array = np.array(G_LR_array).T
        G_RL_array = np.array(G_RL_array).T
        G_RR_array = np.array(G_RR_array).T

        G = []
        G.append(G_LL_array)
        G.append(G_LR_array)
        G.append(G_RL_array)
        G.append(G_RR_array)
        G = np.array(G)

        Path(save_path).mkdir(parents=True, exist_ok=True)
        pickle.dump(G, open(f"{save_path}/cond_data.dump", "wb"))
        
        if (espectrum == True):
            pickle.dump(eigenval_array, open(f"{save_path}/evalues.dump", "wb"))
            if (wavefn == True):
                pickle.dump(np.array(eigenvec_array), open(f"{save_path}/wavefunc.dump", "wb"))

        
        json.dump(params, open(f"{save_path}/params.json", "w"), indent = 4, ensure_ascii = False)

        cls.save_cond_spectrum(G, save_path)
        cls.save_energy_spectrum(eigenval_array, Δz_repeat, params["Δ_0"], save_path)


# So this can also work as a live chat huh. Moshi Moshi.
# I can also make separate classses (with static methods, like above) for calculating the entire energy spectra and conductance spectra.
# I hate myself and I want to die