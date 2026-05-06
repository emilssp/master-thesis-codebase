import numpy as np
import scipy.linalg as la
from scipy import special
from collections import namedtuple

from .constants import PI
from .Hamiltonian import Hamiltonian
from .utils import fermi_dirac, is_converged


Corr = namedtuple(
    "Corr",
    [
        "F0",
        "F_xplus", "F_xmin", "F_yplus", "F_ymin",
        "Fuu_xplus", "Fuu_xmin", "Fuu_yplus", "Fuu_ymin",
        "Fdd_xplus", "Fdd_xmin", "Fdd_yplus", "Fdd_ymin",
    ]
)


def bdg_sc(H: Hamiltonian, temperature,  # Hamiltonian
           atol=1e-6, rtol=1e-4, maxiter=100,
           mixing=1.0, verbose=False):

    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must satisfy 0 < mixing <= 1")

    converged = False
    Ny = H.lattice.Y
    Nx = H.lattice.X
    ky_list = np.linspace(-PI, PI, Ny, endpoint=False)

    F0 = H.F0.copy()
    F_xplus = H.F_xplus.copy()
    F_xmin = H.F_xmin.copy()
    F_yplus = H.F_yplus.copy()
    F_ymin = H.F_ymin.copy()

    Fuu_xplus = H.Fuu_xplus.copy()
    Fuu_xmin = H.Fuu_xmin.copy()
    Fuu_yplus = H.Fuu_yplus.copy()
    Fuu_ymin = H.Fuu_ymin.copy()

    Fdd_xplus = H.Fdd_xplus.copy()
    Fdd_xmin = H.Fdd_xmin.copy()
    Fdd_yplus = H.Fdd_yplus.copy()
    Fdd_ymin = H.Fdd_ymin.copy()

    for iteration in range(maxiter):

        F0_new = 0

        F_xplus_new = 0
        F_xmin_new = 0
        F_yplus_new = 0
        F_ymin_new = 0

        Fuu_xplus_new = 0
        Fuu_xmin_new = 0
        Fuu_yplus_new = 0
        Fuu_ymin_new = 0

        Fdd_xplus_new = 0
        Fdd_xmin_new = 0
        Fdd_yplus_new = 0
        Fdd_ymin_new = 0

        H_kindep = H.build_H_kindep()
        free = 0

        for ky in ky_list:
            H_k = H.build_H_k(ky)
            H2 = H_kindep + H_k

            ep = np.exp(1j * ky)
            em = np.exp(-1j * ky)

            eigval, eigvec = la.eigh(
                H2,
                subset_by_value=(0, np.inf)
            )

            u_up = eigvec[0::4, :]  # electron ↑
            u_dn = eigvec[1::4, :]  # electron ↓
            v_up = eigvec[2::4, :]  # hole ↑
            v_dn = eigvec[3::4, :]  # hole ↓

            Energies = np.tile(eigval.reshape(1, -1), (Nx, 1))
            f_E = fermi_dirac(Energies, temperature)

            if not temperature == 0:
                beta = 1/temperature
                S = np.sum(special.softplus(-eigval*beta))/beta
            else:
                S = 0

            free = free - np.sum(eigval)/2 - S

            # Diagonal ↑↓
            F_ux = np.einsum('nm,nm,nm->n', u_up, np.conj(v_dn), (1-f_E))
            F_vw = np.einsum('nm,nm,nm->n', u_dn, np.conj(v_up), f_E)

            # Offdiagonal ↑↓
            Fx_ux1 = np.einsum('nm,nm,nm->n', u_up[:-1, :],
                               np.conj(v_dn[1:, :]), (1 - f_E[:-1, :]))

            Fx_ux2 = np.einsum('nm,nm,nm->n', u_up[1:, :],
                               np.conj(v_dn[:-1, :]), (1 - f_E[1:, :]))

            Fx_vw1 = np.einsum('nm,nm,nm->n', u_dn[:-1, :],
                               np.conj(v_up[1:, :]), f_E[1:, :])

            Fx_vw2 = np.einsum('nm,nm,nm->n', u_dn[1:, :],
                               np.conj(v_up[:-1, :]), f_E[:-1, :])

            # Triplet ↑↑
            Fuu_xplus_new += (
                np.einsum('nm,nm,nm->n',
                          u_up[:-1, :], np.conj(v_up[1:, :]), (1-f_E[:-1, :]))
                + np.einsum('nm,nm,nm->n',
                            u_up[1:, :], np.conj(v_up)[:-1, :], f_E[1:, :])
            ) / Ny

            Fuu_xmin_new += (
                np.einsum('nm,nm,nm->n',
                          u_up[1:, :], np.conj(v_up[:-1, :]), (1-f_E[1:, :]))
                + np.einsum('nm,nm,nm->n',
                            u_up[:-1, :], np.conj(v_up)[1:, :], f_E[:-1, :])
            ) / Ny

            Fuu_yplus_new += (
                np.einsum('nm,nm,nm->n', u_up, np.conj(v_up), (1-f_E))*ep +
                np.einsum('nm,nm,nm->n', u_up, np.conj(v_up), f_E)*em
            ) / Ny

            Fuu_ymin_new += (
                np.einsum('nm,nm,nm->n', u_up, np.conj(v_up), (1-f_E))*em +
                np.einsum('nm,nm,nm->n', u_up, np.conj(v_up), f_E)*ep
            ) / Ny

            # Triplet ↓↓
            Fdd_xplus_new += (
                np.einsum('nm,nm,nm->n',
                          u_dn[:-1, :], np.conj(v_dn[1:, :]), (1-f_E[:-1, :]))
                + np.einsum('nm,nm,nm->n',
                            u_dn[1:, :], np.conj(v_dn)[:-1, :], f_E[1:, :])
            ) / Ny

            Fdd_xmin_new += (
                np.einsum('nm,nm,nm->n',
                          u_dn[1:, :], np.conj(v_dn[:-1, :]), (1-f_E[1:, :]))
                + np.einsum('nm,nm,nm->n',
                            u_dn[:-1, :], np.conj(v_dn)[1:, :], f_E[:-1, :])
            ) / Ny

            Fdd_yplus_new += (
                np.einsum('nm,nm,nm->n', u_dn, np.conj(v_dn), (1-f_E))*ep +
                np.einsum('nm,nm,nm->n', u_dn, np.conj(v_dn), f_E)*em
            ) / Ny

            Fdd_ymin_new += (
                np.einsum('nm,nm,nm->n', u_dn, np.conj(v_dn), (1-f_E))*em +
                np.einsum('nm,nm,nm->n', u_dn, np.conj(v_dn), f_E)*ep
            ) / Ny

            # New parameters
            F0_new += (F_ux + F_vw) / Ny
            F_xplus_new += (Fx_ux1 + Fx_vw2) / Ny
            F_xmin_new += (Fx_ux2 + Fx_vw1) / Ny
            F_yplus_new += (F_ux * ep + F_vw * em) / Ny
            F_ymin_new += (F_ux * em + F_vw * ep) / Ny

        corr = Corr(
            F0,
            F_xplus, F_xmin, F_yplus, F_ymin,
            Fuu_xplus, Fuu_xmin, Fuu_yplus, Fuu_ymin,
            Fdd_xplus, Fdd_xmin, Fdd_yplus, Fdd_ymin,
        )
        corr_raw = Corr(
            F0_new,
            F_xplus_new, F_xmin_new, F_yplus_new, F_ymin_new,
            Fuu_xplus_new, Fuu_xmin_new, Fuu_yplus_new, Fuu_ymin_new,
            Fdd_xplus_new, Fdd_xmin_new, Fdd_yplus_new, Fdd_ymin_new
        )

        corr_new = Corr(*[
            (1.0 - mixing) * old + mixing * new
            for old, new in zip(corr, corr_raw)
        ])

        if verbose:
            print('============================================')
            correlation_strings = [
                    "F0", "F_xplus", "F_xmin", "F_yplus", "F_ymin",
                    "Fuu_xplus", "Fuu_xmin", "Fuu_yplus", "Fuu_ymin",
                    "Fdd_xplus", "Fdd_xmin", "Fdd_yplus", "Fdd_ymin"
                    ]
            for idx in range(len(correlation_strings)):
                print(f"{correlation_strings[idx]}:")
                abs_diff = np.linalg.norm(corr_new[idx] - corr[idx])
                rel_diff = np.linalg.norm(
                    (corr_new[idx] - corr[idx])/(corr[idx] + 1e-12)
                )
                print(f"Absolute error: {abs_diff}")
                print(f"Relative error: {rel_diff}")
                print(f"Average old: {np.mean(corr[idx])}")
                print(f"Average: {np.mean(corr_new[idx])}")
            print(f"Iteration {iteration + 1}.")

        (
            F0,
            F_xplus, F_xmin, F_yplus, F_ymin,
            Fuu_xplus, Fuu_xmin, Fuu_yplus, Fuu_ymin,
            Fdd_xplus, Fdd_xmin, Fdd_yplus, Fdd_ymin,
        ) = corr_new

        H.set_correlations(corr_new)
        if is_converged(corr, corr_new, atol, rtol):
            print("==================================================")
            print(f"Converged after {iteration+1} iterations")
            print("==================================================")
            converged = True
            break
    E_S = H.free_energy_const_term()
    H.free = free - E_S
    if not converged:
        print("==================================================")
        print(f"WARNING: Failed to converge after {iteration+1} iterations")
        print("==================================================")

    H.converged = converged
