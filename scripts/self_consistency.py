import numpy as np
import scipy.linalg as la
from scipy import special

from .constants import PI
from .hamiltonian import Hamiltonian
from .utils import fermi_dirac, is_converged


def bdg_sc(H: Hamiltonian, temperature,  # Hamiltonian
           atol=1e-6, rtol=1e-4, maxiter=100,
           verbose=False):

    converged = False
    Ny = H.lattice.Y
    Nx = H.lattice.X
    ky_list = np.linspace(PI / Ny, PI, Ny, endpoint=True)

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

        H_kindep = H.build_H_kindep()
        H_k0 = H.build_H_k0()

        eigval0, eigvec0 = la.eigh(
            H_kindep + H_k0,
            subset_by_value=(0, np.inf)
        )

        u_up0 = eigvec0[0::4, :]
        u_dn0 = eigvec0[1::4, :]
        v_up0 = eigvec0[2::4, :]
        v_dn0 = eigvec0[3::4, :]

        Energies0 = np.tile(eigval0.reshape(1, -1), (Nx, 1))
        f_E0 = fermi_dirac(Energies0, temperature)

        if not temperature == 0:
            beta = 1/temperature
            S = np.sum(special.softplus(-eigval0*beta))/beta
        else:
            S = 0
        free = -sum(eigval0)/2 - S

        # Onsite
        F0_ux = np.einsum('nm,nm,nm->n', u_up0, np.conj(v_dn0), (1 - f_E0))
        F0_vw = np.einsum('nm,nm,nm->n', u_dn0, np.conj(v_up0), f_E0)

        # x-bond pieces
        Fx0_ux1 = np.einsum('nm,nm,nm->n', u_up0[:-1, :],
                            np.conj(v_dn0[1:, :]), (1 - f_E0[:-1, :]))

        Fx0_ux2 = np.einsum('nm,nm,nm->n', u_up0[1:, :],
                            np.conj(v_dn0[:-1, :]), (1 - f_E0[1:, :]))

        Fx0_vw1 = np.einsum('nm,nm,nm->n', u_dn0[:-1, :],
                            np.conj(v_up0[1:, :]), f_E0[1:, :])

        Fx0_vw2 = np.einsum('nm,nm,nm->n', u_dn0[1:, :],
                            np.conj(v_up0[:-1, :]), f_E0[:-1, :])

        F0_new = (F0_ux + F0_vw) / Ny
        F_xplus_new = (Fx0_ux1 + Fx0_vw2) / Ny
        F_xmin_new = (Fx0_ux2 + Fx0_vw1) / Ny
        F_yplus_new = F0_new.copy()  # for k=0 the same as onsite
        F_ymin_new = F0_new.copy()  # for k=0 the same as onsite

        # Triplet ↑↑
        Fuu_xplus_new = (
            np.einsum('nm,nm,nm->n',
                      u_up0[:-1, :], np.conj(v_up0[1:, :]), (1 - f_E0[:-1, :]))
            + np.einsum('nm,nm,nm->n',
                        u_up0[1:, :], np.conj(v_up0)[:-1, :], f_E0[1:, :])
        ) / Ny

        Fuu_xmin_new = (
            np.einsum('nm,nm,nm->n',
                      u_up0[1:, :], np.conj(v_up0[:-1, :]), (1 - f_E0[1:, :]))
            + np.einsum('nm,nm,nm->n',
                        u_up0[:-1, :], np.conj(v_up0)[1:, :], f_E0[:-1, :])
        ) / Ny

        Fuu_yplus_new = (
            np.einsum('nm,nm,nm->n', u_up0, np.conj(v_up0), (1.0 - f_E0)) +
            np.einsum('nm,nm,nm->n', u_up0, np.conj(v_up0), f_E0)
        ) / Ny

        Fuu_ymin_new = (
            np.einsum('nm,nm,nm->n', u_up0, np.conj(v_up0), (1.0 - f_E0)) +
            np.einsum('nm,nm,nm->n', u_up0, np.conj(v_up0), f_E0)
        ) / Ny

        # Triplet ↓↓
        Fdd_xplus_new = (
            np.einsum('nm,nm,nm->n',
                      u_dn0[:-1, :], np.conj(v_dn0[1:, :]), (1 - f_E0[:-1, :]))
            + np.einsum('nm,nm,nm->n',
                        u_dn0[1:, :], np.conj(v_dn0)[:-1, :], f_E0[1:, :])
        ) / Ny

        Fdd_xmin_new = (
            np.einsum('nm,nm,nm->n',
                      u_dn0[1:, :], np.conj(v_dn0[:-1, :]), (1 - f_E0)[1:, :])
            + np.einsum('nm,nm,nm->n',
                        u_dn0[:-1, :], np.conj(v_dn0)[1:, :], f_E0[:-1, :])
        ) / Ny

        Fdd_yplus_new = (
            np.einsum('nm,nm,nm->n', u_dn0, np.conj(v_dn0), (1.0 - f_E0)) +
            np.einsum('nm,nm,nm->n', u_dn0, np.conj(v_dn0), f_E0)
        ) / Ny

        Fdd_ymin_new = (
            np.einsum('nm,nm,nm->n', u_dn0, np.conj(v_dn0), (1.0 - f_E0)) +
            np.einsum('nm,nm,nm->n', u_dn0, np.conj(v_dn0), f_E0)
        ) / Ny

        # Singlet / triplet symmetry parts
        FS_x = (Fx0_ux1 + Fx0_ux2 + Fx0_vw1 + Fx0_vw2).conj() / (2.0 * Ny)
        FS_y = F0_new.copy()

        FT_x_plus = (Fx0_ux1 - Fx0_ux2 + Fx0_vw2 - Fx0_vw1).conj() / (2.0 * Ny)
        FT_x_min = (Fx0_ux2 - Fx0_ux1 + Fx0_vw1 - Fx0_vw2).conj() / (2.0 * Ny)
        FT_y_plus = np.zeros(Nx, dtype=np.complex128)
        FT_y_min = np.zeros(Nx, dtype=np.complex128)

        for ky in ky_list:
            H_k = H.build_H_k(ky)
            H2 = H_kindep + H_k

            ep = np.exp(1j * ky)
            em = np.exp(-1j * ky)

            eigval, eigvec = la.eigh(H2)

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
                np.einsum('nm,nm,nm->n', u_dn, np.conj(v_dn), (1.0 - f_E))*em +
                np.einsum('nm,nm,nm->n', u_dn, np.conj(v_dn), f_E)*ep
            ) / Ny

            # New parameters
            F0_new += (F_ux + F_vw) / Ny
            F_xplus_new += (Fx_ux1 + Fx_vw2) / Ny
            F_xmin_new += (Fx_ux2 + Fx_vw1) / Ny
            F_yplus_new += (F_ux * ep + F_vw * em) / Ny
            F_ymin_new += (F_ux * em + F_vw * ep) / Ny

            # Singlet component
            FS_x += (Fx_ux1 + Fx_ux2 + Fx_vw1 + Fx_vw2) / (2.0 * Ny)
            FS_y += (F_ux + F_vw) * np.cos(ky) / Ny

            # Triplet component
            FT_x_plus += (Fx_ux1 + Fx_vw2 - Fx_ux2 - Fx_vw1) / (2.0 * Ny)
            FT_x_min += (Fx_ux2 - Fx_ux1 + Fx_vw1 - Fx_vw2) / (2.0 * Ny)

            FT_y_plus += 1j * (F_ux - F_vw) * np.sin(ky) / Ny
            FT_y_min -= 1j * (F_ux - F_vw) * np.sin(ky) / Ny

        # Filtering out s, d, px, py (same algebra as MATLAB)
        F_swave = (np.r_[0, FS_x] + np.r_[FS_x, 0] + 2.0 * FS_y) / 4.0
        F_dwave = (np.r_[0, FS_x] + np.r_[FS_x, 0] - 2.0 * FS_y) / 4.0
        F_px = (np.r_[0, FT_x_plus] - np.r_[FT_x_min, 0]) / 2.0
        F_py = (FT_y_plus - FT_y_min) / 2.0

        corr = [
            F0, F_xplus, F_xmin, F_yplus, F_ymin,
            Fuu_xplus, Fuu_xmin, Fuu_yplus, Fuu_ymin,
            Fdd_xplus, Fdd_xmin, Fdd_yplus, Fdd_ymin
        ]
        corr_new = [
            F0_new, F_xplus_new, F_xmin_new, F_yplus_new, F_ymin_new,
            Fuu_xplus_new, Fuu_xmin_new, Fuu_yplus_new, Fuu_ymin_new,
            Fdd_xplus_new, Fdd_xmin_new, Fdd_yplus_new, Fdd_ymin_new
        ]

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

        F0 = F0_new.copy()

        F_xplus = F_xplus_new.copy()
        F_xmin = F_xmin_new.copy()
        F_yplus = F_yplus_new.copy()
        F_ymin = F_ymin_new.copy()

        Fuu_xplus = Fuu_xplus_new.copy()
        Fuu_xmin = Fuu_xmin_new.copy()
        Fuu_yplus = Fuu_yplus_new.copy()
        Fuu_ymin = Fuu_ymin_new.copy()

        Fdd_xplus = Fdd_xplus_new.copy()
        Fdd_xmin = Fdd_xmin_new.copy()
        Fdd_yplus = Fdd_yplus_new.copy()
        Fdd_ymin = Fdd_ymin_new.copy()

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

    return F_swave, F_dwave, F_px, F_py
