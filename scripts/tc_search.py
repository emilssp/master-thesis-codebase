import numpy as np
from .hamiltonian import Hamiltonian
from .self_consistency import bdg_sc


def tc_search(H: Hamiltonian, gap_0,
              Tmin, Tmax, symmetry,
              tol=1e-6, max_iter=100):

    Tc = 0
    Tc_list = []  # store midpoint values
    T_high_list = []
    T_low_list = []
    for n in range(max_iter):
        T_half = (Tmax + Tmin) / 2
        Tc_list.append(float(T_half))  # save current midpoint
        T_high_list.append(float(Tmax))
        T_low_list.append(float(Tmin))

        F_swave, F_dwave, F_px, F_py = bdg_sc(H, T_half, maxiter=1)

        if symmetry == 'swave':
            if gap_0 < F_swave:
                Tmin = T_half
            else:
                Tmax = T_half

        if np.abs(T_half - Tc) < tol:
            Tc = T_half
            print(f"Converged after {n+1} iterations")
            break
        Tc = T_half

    return Tc, Tc_list, T_high_list, T_low_list
