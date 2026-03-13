import numpy as np
import scipy.linalg as la

from .constants import PI
from .hamiltonian import Hamiltonian
from .utils import fermi_dirac


def bdg_sc(H: Hamiltonian, temperature=0,  # Hamiltonian
           atol=1e-6, rtol=1e-4, maxiter=100, verbose=False):

    converged = False
    Ny = H.lattice.Y
    Nx = H.lattice.X
    ky_list = np.linspace(PI / Ny, PI, Ny)

    F0 = H.F0.copy()
    F_xplus = H.F_xplus.copy()
    F_xmin = H.F_xmin.copy()
    F_yplus = H.F_yplus.copy()
    F_ymin = H.F_ymin.copy()

    for iteration in range(maxiter):

        H_kindep = H.build_H_kindep()
        H_k0 = H.build_H_k0()

        eigval0, eigvec0 = la.eigh(
            H_kindep + H_k0,
            subset_by_value=(0, np.inf)
        )

        u0 = eigvec0[0::4, :]
        v0 = eigvec0[1::4, :]
        w0 = eigvec0[2::4, :]
        x0 = eigvec0[3::4, :]

        f = fermi_dirac(eigval0, temperature)
        f_E0 = np.tile(f, (Nx, 1))

        # Onsite
        F0_ux = np.einsum('nm,nm,nm->n', u0, np.conj(x0), (1 - f_E0))
        F0_vw = np.einsum('nm,nm,nm->n', v0, np.conj(w0), f_E0)

        # x-bond pieces
        Fx0_ux1 = np.einsum('nm,nm,nm->n', u0[:-1, :],
                            np.conj(x0[1:, :]), (1 - f_E0[:-1, :]))

        Fx0_ux2 = np.einsum('nm,nm,nm->n', u0[1:, :],
                            np.conj(x0[:-1, :]), (1 - f_E0[1:, :]))

        Fx0_vw1 = np.einsum('nm,nm,nm->n', v0[:-1, :],
                            np.conj(w0[1:, :]), f_E0[1:, :])

        Fx0_vw2 = np.einsum('nm,nm,nm->n', v0[1:, :],
                            np.conj(w0[:-1, :]), f_E0[:-1, :])

        # y-bond pieces
        Fy0_ux = np.einsum('nm,nm,nm->n', u0, np.conj(x0), (1 - f_E0))
        Fy0_vw = np.einsum('nm,nm,nm->n', v0, np.conj(w0), f_E0)

        F0_new = (F0_ux + F0_vw) / Ny
        F_xplus_new = (Fx0_ux1 + Fx0_vw2) / Ny
        F_xmin_new = (Fx0_ux2 + Fx0_vw1) / Ny
        K_y0 = (Fy0_ux + Fy0_vw) / Ny
        F_yplus_new = K_y0.copy()
        F_ymin_new = K_y0.copy()

        # Singlet / triplet symmetry parts
        FS_x = (Fx0_ux1 + Fx0_ux2 + Fx0_vw1 + Fx0_vw2).conj() / (2.0 * Ny)
        FS_y = K_y0.copy()

        FT_x_plus = (Fx0_ux1 - Fx0_ux2 + Fx0_vw2 - Fx0_vw1).conj() / (2.0 * Ny)
        FT_x_min = (Fx0_ux2 - Fx0_ux1 + Fx0_vw1 - Fx0_vw2).conj() / (2.0 * Ny)
        FT_y_plus = np.zeros(Nx, dtype=np.complex128)
        FT_y_min = np.zeros(Nx, dtype=np.complex128)

        for ky in ky_list:
            H_k = H.build_H_k(ky)
            H2 = H_kindep + H_k

            eigval, eigvec = la.eigh(H2)

            u = eigvec[0::4, :]
            v = eigvec[1::4, :]
            w = eigvec[2::4, :]
            x = eigvec[3::4, :]

            Energies = np.tile(eigval, (Nx, 1))
            f_E = fermi_dirac(Energies, temperature)

            # Onsite
            F_ux = np.einsum('nm,nm,nm->n', u, np.conj(x), (1 - f_E))
            F_vw = np.einsum('nm,nm,nm->n', v, np.conj(w), f_E)

            # x-bond pieces
            Fx_ux1 = np.einsum('nm,nm,nm->n', u[:-1, :],
                               np.conj(x[1:, :]), (1 - f_E[:-1, :]))

            Fx_ux2 = np.einsum('nm,nm,nm->n', u[1:, :],
                               np.conj(x[:-1, :]), (1 - f_E[1:, :]))

            Fx_vw1 = np.einsum('nm,nm,nm->n', v[:-1, :],
                               np.conj(w[1:, :]), f_E[1:, :])

            Fx_vw2 = np.einsum('nm,nm,nm->n', v[1:, :],
                               np.conj(w[:-1, :]), f_E[:-1, :])

            # y-bond pieces
            Fy_ux = np.einsum('nm,nm,nm->n', u, np.conj(x), (1 - f_E))
            Fy_vw = np.einsum('nm,nm,nm->n', v, np.conj(w), f_E)

            ep = np.exp(1j * ky)
            em = np.exp(-1j * ky)

            F0_new += (F_ux + F_vw) / Ny
            F_xplus_new += (Fx_ux1 + Fx_vw2) / Ny
            F_xmin_new += (Fx_ux2 + Fx_vw1) / Ny
            F_yplus_new += (Fy_ux * ep + Fy_vw * em) / Ny
            F_ymin_new += (Fy_ux * em + Fy_vw * ep) / Ny

            # Singlet component
            FS_x += (Fx_ux1 + Fx_ux2 + Fx_vw1 + Fx_vw2) / (2.0 * Ny)
            FS_y += (Fy_ux + Fy_vw) * np.cos(ky) / Ny

            # Triplet component
            FT_x_plus += (Fx_ux1 - Fx_ux2 + Fx_vw2 - Fx_vw1) / (2.0 * Ny)
            FT_x_min += (Fx_ux2 - Fx_ux1 + Fx_vw1 - Fx_vw2) / (2.0 * Ny)

            FT_y_plus += 1j * (Fy_ux - Fy_vw) * np.sin(ky) / Ny
            FT_y_min -= 1j * (Fy_ux - Fy_vw) * np.sin(ky) / Ny

        # Filtering out s, d, px, py (same algebra as MATLAB)
        F_swave = (np.r_[0, FS_x] + np.r_[FS_x, 0] + 2.0 * FS_y) / 4.0
        F_dwave = (np.r_[0, FS_x] + np.r_[FS_x, 0] - 2.0 * FS_y) / 4.0
        F_px = (np.r_[0, FT_x_plus] - np.r_[FT_x_min, 0]) / 2.0
        F_py = (FT_y_plus - FT_y_min) / 2.0

        # Convergence metrics
        absdiff0 = np.linalg.norm(F0_new - F0)
        absdiff1 = np.linalg.norm(F_xplus_new - F_xplus)
        absdiff2 = np.linalg.norm(F_xmin_new - F_xmin)
        absdiff3 = np.linalg.norm(F_yplus_new - F_yplus)
        absdiff4 = np.linalg.norm(F_ymin_new - F_ymin)

        # avoid divide-by-zero blow-ups
        eps = 1e-12
        reldiff0 = np.linalg.norm((F0_new - F0)/(F0 + eps))
        reldiff1 = np.linalg.norm((F_xplus_new - F_xplus)/(F_xplus + eps))
        reldiff2 = np.linalg.norm((F_xmin_new - F_xmin)/(F_xmin + eps))
        reldiff3 = np.linalg.norm((F_yplus_new - F_yplus)/(F_yplus + eps))
        reldiff4 = np.linalg.norm((F_ymin_new - F_ymin)/(F_ymin + eps))

        c1 = ((absdiff0 < atol) and (absdiff1 < atol) and (absdiff2 < atol)
              and (absdiff3 < atol) and (absdiff4 < atol))
        c2 = ((reldiff0 < rtol) and (reldiff1 < rtol) and (reldiff2 < rtol)
              and (reldiff3 < rtol) and (reldiff4 < rtol))

        corr = [F0, F_xplus, F_xmin, F_yplus, F_ymin]
        corr_new = [F0_new, F_xplus_new, F_xmin_new, F_yplus_new, F_ymin_new]

        if verbose:
            abs_diff = [absdiff0, absdiff1, absdiff2, absdiff3, absdiff4]
            rel_diff = [reldiff0, reldiff1, reldiff2, reldiff3, reldiff4]

            print('============================================')
            print(f"Iteration {iteration + 1}:")
            correlation_strings = [
                    "F0", "F_xplus", "F_xmin", "F_yplus", "F_ymin"]
            for idx in range(len(abs_diff)):
                print(f"{correlation_strings[idx]}:")
                print(f"Absolute error: {abs_diff[idx]}")
                print(f"Relative error: {rel_diff[idx]}")
                print(f"Average old: {np.mean(corr[idx])}")
                print(f"Average: {np.mean(corr_new[idx])}")

        F0 = F0_new.copy()

        F_xplus = F_xplus_new.copy()
        F_xmin = F_xmin_new.copy()
        F_yplus = F_yplus_new.copy()
        F_ymin = F_ymin_new.copy()

        F_xplus = F_xplus_new.copy()
        F_xmin = F_xmin_new.copy()
        F_yplus = F_yplus_new.copy()
        F_ymin = F_ymin_new.copy()

        F_xplus = F_xplus_new.copy()
        F_xmin = F_xmin_new.copy()
        F_yplus = F_yplus_new.copy()
        F_ymin = F_ymin_new.copy()
        H.set_correlations(corr_new)

        if c1 and c2:
            print("==================================================")
            print(f"Converged after {iteration+1} iterations")
            print("==================================================")
            converged = True
            break

    if not converged:
        print("==================================================")
        print(f"WARNING: Failed to converge after {iteration+1} iterations")
        print("==================================================")

    return F_swave, F_dwave, F_px, F_py
