import datetime
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from conductance_project.main import main

class ProgressParallel(Parallel):
    """
    A subclass of the joblib.Parallel generator, which prints a tqdm progress bar for the parallel tasks.
    """

    def __init__(self, *args, **kwargs):
        try:
            self.ntotal = kwargs.pop("ntotal")
        except KeyError:
            self.ntotal = 0
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(ncols=100) as self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.ntotal
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def parallel_eval(func, arr, *args, **kwargs):
    pb = ProgressParallel(n_jobs=-1, ntotal=len(arr))
    return pb(delayed(func)(val, *args, **kwargs) for val in arr)


params = {
    "a" : 1,
    "t_1" : 102,
    "t_2" : 102,
    "t_S" : 102,
    "N_1" : 40,
    "N_2" : 48,
    "N_S" : 400,
    "N_B1" : 4,
    "N_B2" : 4,
    "μ_1" : 0.2,
    "μ_2" : 0.2,
    "μ_S" : 0.2,
    "γ_1" : 10,
    "γ_2" : 10,
    "Δ_0" : 0.5,
    "Δz" : None,
    "α_1" : 3.5,
    "α_2" : 3.5,
    "μ_L" : 20,
    "μ_R" : 20,
    "λ_L" : 20,
    "λ_R" : 24,
    "λ_SL" : 20,
    "λ_SR" : 24,
    "timestamp": None,
}

def function(dict_update):
    main({**params, **dict_update})
    
dict_list = [dict(λ_L=i, λ_SL=i, λ_R=j ,λ_SR=j, timestamp=datetime.datetime.now().timestamp()) for i in np.arange(4, 41, 4) for j in np.arange(4, 41, 4)]
parallel_eval(function, dict_list)