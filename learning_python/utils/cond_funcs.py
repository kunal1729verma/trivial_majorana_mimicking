"""
Form of the conductance functions (local and non-local) when we have a conservation law in electron-hole d.o.f. but not in spin d.o.f.
"""
class ConductanceFuncsAbs:
    @staticmethod
    def G_LL_func(smatrix):
        raise NotImplemented
    
    @staticmethod
    def G_LR_func(smatrix):
        raise NotImplemented
    
    @staticmethod
    def G_RL_func(smatrix):
        raise NotImplemented
    
    @staticmethod
    def G_RR_func(smatrix):
        raise NotImplemented


class ConductanceStandard(ConductanceFuncsAbs):

    def G_LL_func(smatrix):
        N =    smatrix.submatrix((0, 0), (0, 0)).shape[0] # dim. of the submatrix concerning e-e reflection
        Ree_L = smatrix.transmission((0, 0), (0, 0))      # e to e reflection
        Reh_L = smatrix.transmission((0, 1), (0, 0))      # e to h reflection 
        return(N - Ree_L + Reh_L)

    def G_LR_func(smatrix):
        Te_LR = smatrix.transmission((0, 0), (1, 0))      # e to e transmission
        Ae_LR = smatrix.transmission((0, 1), (1, 0))      # e to h transmission
        return (-(Te_LR - Ae_LR))

    def G_RL_func(smatrix):
        Te_RL = smatrix.transmission((1, 0), (0, 0))      # e to e transmission
        Ae_RL = smatrix.transmission((1, 1), (0, 0))      # e to h transmission
        return (-(Te_RL - Ae_RL))

    def G_RR_func(smatrix):
        N =    smatrix.submatrix((1, 0), (1, 0)).shape[0] # dim. of the submatrix concerning e-e reflection
        Ree_R = smatrix.transmission((1, 0), (1, 0))      # e to e reflection
        Reh_R = smatrix.transmission((1, 1), (1, 0))      # e to h reflection
        return(N - Ree_R + Reh_R)