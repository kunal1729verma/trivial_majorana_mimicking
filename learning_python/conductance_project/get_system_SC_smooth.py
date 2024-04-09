import kwant
import numpy as np
import tinyarray
import typing

from dataclasses import dataclass, fields

Θ = np.heaviside
s_0 = tinyarray.array([[1, 0], [0, 1]])
s_x = tinyarray.array([[0, 1], [1, 0]])
s_y = tinyarray.array([[0, -1j], [1j, 0]])
s_z = tinyarray.array([[1, 0], [0, -1]])

# Do I need to carry these ^^^^ around everywhere? No put them in utils file
# And do I also need to import the libraries and its modules in every file? Yes
@dataclass
class ProfileFunctions:
    a : 'typing.Any'
    t_1: 'typing.Any'
    t_2: 'typing.Any'
    t_S: 'typing.Any'
    N_1: 'typing.Any'
    N_2: 'typing.Any'
    N_S: 'typing.Any'
    N_B1: 'typing.Any'
    N_B2: 'typing.Any'
    μ_1: 'typing.Any'
    μ_2: 'typing.Any'
    μ_S: 'typing.Any'
    γ_1: 'typing.Any'
    γ_2: 'typing.Any'
    Δ_0: 'typing.Any'
    Δz: 'typing.Any'
    α_1: 'typing.Any'
    α_2: 'typing.Any'
    μ_L: 'typing.Any'
    μ_R: 'typing.Any'
    λ_L: 'typing.Any'
    λ_R: 'typing.Any'
    λ_SL: 'typing.Any'
    λ_SR: 'typing.Any'

    #Optional
    def get_params_dict(self):
        params_list = [param.name for param in fields(ProfileFunctions)]
        return {key:value for key, value in self.__dict__.items() if key in params_list}

    @property  # removes need to call function with brackets ()
    def N(self):
        return self.N_1 + self.N_S + self.N_2

    @property
    def N_b(self):
        return self.N_1 + 1/2
    
    @property 
    def N_b_prime(self):
        return self.N_1 + self.N_S + 1/2
    
    @property
    def N_Bb(self):
        return self.N_B1 + 1/2
    
    @property
    def N_Bb_prime(self):
        return self.N - self.N_B2 + 1/2
    
    @staticmethod
    def Ω_n(x, Ni, λ):
        Ω = 1/2 * (1 + np.tanh((x - Ni)/λ))
        return Ω
    
    def Δ_n(self, x): 
        return self.Δ_0 * (self.Ω_n(x, self.N_1, self.λ_SL) - self.Ω_n(x, self.N_1 + self.N_S + 1, self.λ_SR))
    
    def μ_n(self, x):
        return self.μ_1 + (self.μ_S - self.μ_1) * self.Ω_n(x, self.N_1, self.λ_L) + (self.μ_2 - self.μ_S) * self.Ω_n(x, self.N_1 + self.N_S + 1, self.λ_R)
    
    def γ_n(self, x):
        return self.γ_1 * Θ(self.N_Bb - x, 0.5) + self.γ_2 * Θ(x - self.N_Bb_prime, 0.5)
    
    # Can't just let the heaviside functions be there because it gives 0.5 of the value at the boundaries.

    def t_n(self, x):
        # return self.t_1 * Θ(self.N_b - x, 0.5) + self.t_2 * Θ(x - self.N_b_prime, 0.5) + self.t_S * (Θ(x - self.N_b, 0.5) - Θ(x - self.N_b_prime, 0.5))
        return self.t_1
    
    def Δz_n(self, x): 
        # return self.Δz * (Θ(self.N_b - x, 0.5) + Θ(x - self.N_b_prime, 0.5))
        return self.Δz
    
    def α_n(self, x):
        # return self.α_1 * Θ(self.N_b - x, 0.5) + self.α_2 * Θ(x - self.N_b_prime, 0.5)
        return self.α_1

# Scattering Hamiltonian

def get_onsite_term(site, prof_fns : ProfileFunctions):   # signifies that the second input is an object of type ProfileFunctions.
    
    # site should go over the indices 0 to N.
    # (x,) = site.pos + 1
    x = site
    H_ii = (prof_fns.t_n(x + 1/2) + prof_fns.t_n(x - 1/2) - prof_fns.μ_n(x) + prof_fns.γ_n(x)) * np.kron(s_z, s_0) + prof_fns.Δz_n(x) * np.kron(s_0, s_z) + prof_fns.Δ_n(x) * np.kron(s_x, s_0)

    return H_ii 


def get_hopping_term(site1, site2, prof_fns : ProfileFunctions):

    x1 = site1
    x2 = site2

    # site2 should go over the indices 1 to N (and 0 to N-1 for site1).
    if (abs(x2 - x1) > 1):
        return 0
    
    elif ((x2 - x1) == 1):
        x = x1
        H_iip1 = - prof_fns.t_n(x + 1/2) * np.kron(s_z, s_0) + 1j * prof_fns.α_n(x + 1/2) * np.kron(s_z, s_z)
        return H_iip1
    
    
def scatter_def(syst, prof_fns : ProfileFunctions):
    lat = kwant.lattice.chain(norbs = 4)
    
    for x in range(1, prof_fns.N + 1):
        syst[lat(x - 1)] = get_onsite_term(x, prof_fns)

    for x in range(1, prof_fns.N):
        syst[lat(x - 1), lat(x)] = get_hopping_term(x, x+1, prof_fns)

    return syst


# Get lead terms

def get_left_lead_onsite(prof_fns : ProfileFunctions):
    return (2 * prof_fns.t_1 - prof_fns.μ_L) * np.kron(s_z, s_0) + prof_fns.Δz * np.kron(s_0, s_z)

def get_left_lead_hopping(prof_fns : ProfileFunctions):
    return - prof_fns.t_1 * np.kron(s_z, s_0) + 1j * prof_fns.α_1 * np.kron(s_z, s_z)

def get_right_lead_onsite(prof_fns : ProfileFunctions):
    return (2 * prof_fns.t_1 - prof_fns.μ_L) * np.kron(s_z, s_0) + prof_fns.Δz * np.kron(s_0, s_z)

def get_right_lead_hopping(prof_fns : ProfileFunctions):
    return - prof_fns.t_1 * np.kron(s_z, s_0) + 1j * prof_fns.α_1 * np.kron(s_z, s_z)

# Lead Hamiltonian

def leads_def(syst, prof_fns : ProfileFunctions):

    lat = kwant.lattice.chain(norbs = 4)
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-prof_fns.a,)), conservation_law = - np.kron(s_z, s_0))
    right_lead = kwant.Builder(kwant.TranslationalSymmetry((+prof_fns.a,)), conservation_law = - np.kron(s_z, s_0))
    
    left_lead[(lat(0))] = get_left_lead_onsite(prof_fns)
    left_lead[lat.neighbors()] = get_left_lead_hopping(prof_fns)
    
    right_lead[(lat(0))] = get_right_lead_onsite(prof_fns)
    right_lead[lat.neighbors()] = get_right_lead_hopping(prof_fns)
    
    syst.attach_lead(left_lead)
    syst.attach_lead(right_lead)

    return syst