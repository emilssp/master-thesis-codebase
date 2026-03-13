# flake8: noqa: E501

import numpy as cp 

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

def g(i_site: int, comp: int) -> int:
    """Global index for site i_site (0..Nx-1) and component comp (0..3)."""
    return 4 * i_site + comp

def delta(x, eta=1e-6):  # Lorentzian approximation of delta function
    return eta / (PI * (eta**2 + x**2))


def lorentzian(x, eta=1e-6):
    return eta / (PI * (x**2 + eta**2))


def fermi_dirac(energy, T=1e-6):
    if T == 0:
        return np.zeros_like(energy)
    else:
        return 1.0 / (np.exp(energy / T) + 1.0)


def is_hermitian(matrix, atol=1e-8, rtol=1e-6):
    is_hermitian = np.allclose(matrix, matrix.conj().T, rtol=rtol, atol=atol)
    return is_hermitian
