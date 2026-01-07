# flake8: noqa: E501

import cupy as cp

from .constants import *
from .hamiltonian import *


def tc_search(H, Tmin, Tmax, mu=0, U=2, gap_0=1e-6,
              tol=1e-6, max_iter=100):
    block = cp.zeros((4, 4), dtype=cp.complex128)
    block[:2, :2] = -mu * s0
    block[2:, 2:] = mu * s0
    block[:2, 2:] = -1j * gap_0 * s2
    block[2:, :2] = (-1j * gap_0 * s2).conj().T

    Tc = 0
    Tc_list = []  # store midpoint values
    T_high_list = []
    T_low_list = []
    for n in range(max_iter):
        for i in range(H.lattice.num_sites):
            H.set_block(i, i, block)

        T_half = (Tmax + Tmin) / 2
        Tc_list.append(float(T_half))  # save current midpoint
        T_high_list.append(float(Tmax))
        T_low_list.append(float(Tmin))
        gap_new = cp.mean(H.bdg_self_consistency(U, T_half, max_iter=1))

        if gap_0 < gap_new:
            Tmin = T_half
        else:
            Tmax = T_half

        if cp.abs(T_half - Tc) < tol:
            Tc = T_half
            print(f"Converged after {n+1} iterations")
            break
        Tc = T_half

    return Tc, Tc_list, T_high_list, T_low_list
