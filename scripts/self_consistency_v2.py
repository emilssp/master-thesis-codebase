# from concurrent.futures import ThreadPoolExecutor
# import os

import numpy as np
import scipy.linalg as la
from concurrent.futures import ProcessPoolExecutor

from .constants import PI
from .hamiltonian_v2 import Hamiltonian, fermi_dirac


def corr_k(H: Hamiltonian, k, temperature):
    Ny = H.lattice.Y
    Hk = H.build_Hk(k)
    # k=0, En>0
    # if k == 0:
    evals, evecs = la.eigh(H.matrix+Hk, subset_by_value=(0.0, np.inf))
    # k>0, En
    # else:
    #     evals, evecs = la.eigh(H.matrix+Hk)
    f = fermi_dirac(evals, temperature)          # (Neig,)

    Nx = evecs.shape[0] // 4

    u_up = evecs[0:4*Nx:4, :]  # u0 = u_up
    u_dn = evecs[1:4*Nx:4, :]  # v0 = u_dn
    v_up = evecs[2:4*Nx:4, :]  # w0 = v_up
    v_dn = evecs[3:4*Nx:4, :]  # x0 = v_dn

    exp_m = np.exp(-1j * k)
    exp_p = np.exp(1j * k)

    F0 = (np.einsum('nm,nm,m->n', u_up, np.conj(v_dn), (1.0 - f)) +
          np.einsum('nm,nm,m->n', u_dn, np.conj(v_up), f)) / Ny

    F_xplus = (np.einsum('nm,nm,m->n',
                         u_up[:-1, :], np.conj(v_dn[1:, :]), (1.0 - f)) +
               np.einsum('nm,nm,m->n',
                         u_dn[1:, :], np.conj(v_up[:-1, :]), f))/Ny

    F_xmin = (np.einsum('nm,nm,m->n',
                        u_up[1:, :], np.conj(v_dn[:-1, :]), (1.0 - f)) +
              np.einsum('nm,nm,m->n',
                        u_dn[:-1, :], np.conj(v_up[1:, :]), f))/Ny

    Fy1 = np.einsum('nm,nm,m->n', u_up, np.conj(v_dn), (1.0 - f))
    Fy2 = np.einsum('nm,nm,m->n', u_dn, np.conj(v_up), f)

    F_yplus = (Fy1 * exp_p + Fy2 * exp_m) / Ny
    F_ymin = (Fy1 * exp_m + Fy2 * exp_p) / Ny

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


def _corr_k_worker(args):
    H, ky, temperature = args
    return corr_k(H, ky, temperature)


def bdg_self_consistency_step(H: Hamiltonian, temperature=0):
    Ny = H.lattice.Y
    ky_list = np.linspace(-PI, PI, Ny, endpoint=False)

    args = ((H, ky, temperature) for ky in ky_list)

    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_corr_k_worker, args))

    acc = tuple(sum(vals) for vals in zip(*results))
    return acc


def bdg_self_consistency(H: Hamiltonian, temperature=0,  # Hamiltonian
                         atol=1e-6, rtol=1e-4, maxiter=100, verbose=False):
    converged = False
    for iteration in range(maxiter):
        corr = H.get_correlations()
        corr_new = bdg_self_consistency_step(H, temperature)

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

        if np.all(abs_diff < atol) and np.all(rel_diff < rtol):
            print(f"Converged after {iteration + 1} iterations")
            converged = True
            H.set_correlations(corr_new)
            break
        H.set_correlations(corr_new)

    if not converged:
        print(f"WARNING: Failed to converge after {maxiter} iterations")
