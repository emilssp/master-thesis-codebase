# from concurrent.futures import ThreadPoolExecutor
# import os

import numpy as np
import scipy.linalg as la
from operator import add

from .constants import PI
from .hamiltonian_v2 import Hamiltonian
from .utils import fermi_dirac


def corr_k(H: Hamiltonian, H_kindep, k, temperature):
    Ny = H.lattice.Y
    # k=0, En>0
    if k == 0:  # or abs(k) == PI:
        H_k0 = H.build_H_k0()
        evals, evecs = la.eigh(H_kindep + H_k0,
                               subset_by_value=(0.0, np.inf))
    # k>0, En
    else:
        Hk = H.build_H_k(k)
        evals, evecs = la.eigh(H_kindep + Hk)
    f = fermi_dirac(evals, temperature)
    energies = np.tile(evals, (H.lattice.X, 1))
    f_E = fermi_dirac(energies, temperature)          # (Neig,)

    u_up = evecs[0::4, :]  # u = u_up
    u_dn = evecs[1::4, :]  # v = u_dn
    v_up = evecs[2::4, :]  # w = v_up
    v_dn = evecs[3::4, :]  # x = v_dn

    exp_m = np.exp(-1j * k)
    exp_p = np.exp(1j * k)

    F0 = (np.einsum('nm,nm,m->n', u_up, np.conj(v_dn), (1-f)) +
          np.einsum('nm,nm,m->n', u_dn, np.conj(v_up), f)) / Ny

    # experiment
    Fx_ux1 = u_up[:-1, :] * np.conj(v_dn[1:, :]) * (1 - f_E[:-1, :])
    Fx_ux2 = u_up[1:, :] * np.conj(v_dn[:-1, :]) * (1 - f_E[1:, :])
    Fx_vw1 = u_dn[:-1, :] * np.conj(v_up[1:, :]) * (f_E[1:, :])
    Fx_vw2 = u_dn[1:, :] * np.conj(v_up[:-1, :]) * (f_E[:-1, :])

    F_xplus = np.sum(Fx_ux1 + Fx_vw2, axis=1) / Ny
    F_xmin = np.sum(Fx_ux2 + Fx_vw1, axis=1) / Ny

    Fy_ux = u_up * np.conj(v_dn) * (1 - f_E)
    Fy_vw = u_dn * np.conj(v_up) * f_E

    F_yplus = np.sum(Fy_ux * exp_p + Fy_vw * exp_m, axis=1) / Ny
    F_ymin = np.sum(Fy_ux * exp_m + Fy_vw * exp_p, axis=1) / Ny

    # end experiment

    Fuu_xplus = (
            np.einsum('nm,nm,m->n',
                      u_up[:-1, :], np.conj(v_up[1:, :]), (1 - f)) +
            np.einsum('nm,nm,m->n',
                      u_up[1:, :], np.conj(v_up)[:-1, :], f)
    ) / Ny

    Fuu_xmin = (
        np.einsum('nm,nm,m->n',
                  u_up[1:, :], np.conj(v_up[:-1, :]), (1 - f)) +
        np.einsum('nm,nm,m->n',
                  u_up[:-1, :], np.conj(v_up)[1:, :], f)
    ) / Ny

    Fuu_yplus = (
        np.einsum('nm,nm,m->n', u_up, np.conj(v_up), (1.0 - f)) * exp_p +
        np.einsum('nm,nm,m->n', u_up, np.conj(v_up), f) * exp_m
    ) / Ny

    Fuu_ymin = (
        np.einsum('nm,nm,m->n', u_up, np.conj(v_up), (1.0 - f)) * exp_m +
        np.einsum('nm,nm,m->n', u_up, np.conj(v_up), f) * exp_p
    ) / Ny

    Fdd_xplus = (
        np.einsum('nm,nm,m->n', u_dn[:-1, :], np.conj(v_dn[1:, :]), (1 - f)) +
        np.einsum('nm,nm,m->n', u_dn[1:, :], np.conj(v_dn)[:-1, :], f)
    ) / Ny

    Fdd_xmin = (
        np.einsum('nm,nm,m->n', u_dn[1:, :], np.conj(v_dn[:-1, :]), (1 - f)) +
        np.einsum('nm,nm,m->n', u_dn[:-1, :], np.conj(v_dn)[1:, :], f)
    ) / Ny

    Fdd_yplus = (
        np.einsum('nm,nm,m->n', u_dn, np.conj(v_dn), (1.0 - f)) * exp_p +
        np.einsum('nm,nm,m->n', u_dn, np.conj(v_dn), f) * exp_m
    ) / Ny

    Fdd_ymin = (
        np.einsum('nm,nm,m->n', u_dn, np.conj(v_dn), (1.0 - f)) * exp_m +
        np.einsum('nm,nm,m->n', u_dn, np.conj(v_dn), f) * exp_p
    ) / Ny

    acc = (F0, F_xplus, F_xmin, F_yplus, F_ymin,
           Fuu_xplus, Fuu_xmin, Fuu_yplus, Fuu_ymin,
           Fdd_xplus, Fdd_xmin, Fdd_yplus, Fdd_ymin)

    # comps = (FS_x, FS_y, FT_xplus, FT_xmin, FT_yplus,FT_ymin)
    return acc


# SERIALIZE THIS JUST IN CASE OF ERROR UNDER THE PARALELIZATION
def bdg_self_consistency_step(H: Hamiltonian, H_kindep, temperature=0):
    Ny = H.lattice.Y
    ky_list = np.linspace(PI/Ny, PI, Ny, endpoint=False)

    acc = corr_k(H, H_kindep, 0, temperature)

    for k in ky_list:
        corr = corr_k(H, H_kindep, k, temperature)
        acc = tuple(map(add, acc, corr))

    return acc


def bdg_self_consistency(H: Hamiltonian, temperature=0,  # Hamiltonian
                         atol=1e-6, rtol=1e-4, maxiter=100, verbose=False):
    converged = False
    for iteration in range(maxiter):
        corr = H.get_correlations()

        H_kindep = H.build_H_kindep()

        corr_new = bdg_self_consistency_step(H, H_kindep, temperature)

        abs_diff = np.array([np.linalg.norm(c2-c1)
                             for c1, c2 in zip(corr, corr_new)])

        rel_diff = np.array([np.linalg.norm((c2-c1)/(c1 + 1e-12))
                             for c1, c2 in zip(corr, corr_new)])

        if verbose:
            print('============================================')
            print(f"Iteration {iteration + 1}:")
            correlation_strings = [
                "F0", "F_xplus", "F_xmin", "F_yplus", "F_ymin",
                "Fuu_xplus", "Fuu_xmin", "Fuu_yplus", "Fuu_ymin",
                "Fdd_xplus", "Fdd_xmin", "Fdd_yplus", "Fdd_ymin"
            ]
            for idx in range(len(abs_diff)):
                print(f"{correlation_strings[idx]}:")
                print(f"Absolute error: {abs_diff[idx]}")
                print(f"Relative error: {rel_diff[idx]}")
                print(f"Average_old: {np.mean(corr[idx])}")
                print(f"Average_new: {np.mean(corr_new[idx])}")

        H.set_correlations(corr_new)
        if np.all(abs_diff < atol) and np.all(rel_diff < rtol):
            print(f"Converged after {iteration + 1} iterations")
            converged = True
            break

    if not converged:
        print(f"WARNING: Failed to converge after {maxiter} iterations")
