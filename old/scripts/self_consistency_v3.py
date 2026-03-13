# from concurrent.futures import ThreadPoolExecutor
# import os

import numpy as np
import scipy.linalg as la
# from operator import add

from .constants import PI
from .hamiltonian_v2 import Hamiltonian
from .utils import fermi_dirac


def bdg_sc(H: Hamiltonian, temperature=0,  # Hamiltonian
           atol=1e-6, rtol=1e-4, maxiter=100, verbose=False):

    Ny = H.lattice.Y
    Nx = H.lattice.X
    ky_list = np.linspace(PI / Ny, PI, Ny)

    Fx_plus = H.F_xplus.copy()
    Fx_min = H.F_xmin.copy()
    Fy_plus = H.F_yplus.copy()
    Fy_min = H.F_ymin.copy()

    for iteration in range(maxiter):

        H_kindep = H.build_H_kindep()
        H_k0 = H.build_H_k0()

        eigval0, eigvec0 = la.eigh(H_kindep + H_k0)

        u0 = eigvec0[0::4, :]
        v0 = eigvec0[1::4, :]
        w0 = eigvec0[2::4, :]
        x0 = eigvec0[3::4, :]

        Energies0 = np.tile(eigval0, (Nx, 1))
        f_E0 = fermi_dirac(Energies0, temperature)
        pos_energy0 = Energies0 > 0

        # Pair correlations (k=0), filtered to positive energies as in MATLAB
        Fx0_ux1 = u0[:-1, :] * np.conj(x0[1:, :]) * (1 - f_E0[:-1, :]) * pos_energy0[:-1, :]
        Fx0_ux2 = u0[1:, :] * np.conj(x0[:-1, :]) * (1 - f_E0[1:, :]) * pos_energy0[1:, :]
        Fx0_vw1 = v0[:-1, :] * np.conj(w0[1:, :]) * (f_E0[1:, :]) * pos_energy0[1:, :]
        Fx0_vw2 = v0[1:, :] * np.conj(w0[:-1, :]) * (f_E0[:-1, :]) * pos_energy0[:-1, :]

        Fy0_ux = u0 * np.conj(x0) * (1 - f_E0) * pos_energy0
        Fy0_vw = v0 * np.conj(w0) * (f_E0) * pos_energy0

        Fx_plus_new = np.sum(Fx0_ux1 + Fx0_vw2, axis=1) / Ny
        Fx_min_new = np.sum(Fx0_ux2 + Fx0_vw1, axis=1) / Ny
        K_y0 = np.sum(Fy0_ux + Fy0_vw,   axis=1) / Ny
        Fy_plus_new = K_y0.copy()
        Fy_min_new = K_y0.copy()

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

            Fx_ux1 = u[:-1, :] * np.conj(x[1:, :]) * (1 - f_E[:-1, :])
            Fx_ux2 = u[1:, :] * np.conj(x[:-1, :]) * (1 - f_E[1:, :])
            Fx_vw1 = v[:-1, :] * np.conj(w[1:, :]) * (f_E[1:, :])
            Fx_vw2 = v[1:, :] * np.conj(w[:-1, :]) * (f_E[:-1, :])

            Fy_ux = u * np.conj(x) * (1 - f_E)
            Fy_vw = v * np.conj(w) * (f_E)

            e = np.exp(1j * ky)
            em = np.exp(-1j * ky)

            Fx_plus_new += np.sum(Fx_ux1 + Fx_vw2, axis=1) / Ny
            Fx_min_new += np.sum(Fx_ux2 + Fx_vw1, axis=1) / Ny
            Fy_plus_new += np.sum(Fy_ux * e + Fy_vw * em, axis=1) / Ny
            Fy_min_new += np.sum(Fy_ux * em + Fy_vw * e,  axis=1) / Ny

        Fx_plus = H.F_xplus
        Fx_min = H.F_xmin
        Fy_plus = H.F_yplus
        Fy_min = H.F_ymin

        # Convergence metrics
        absdiff1 = np.linalg.norm(Fx_plus_new - Fx_plus)
        absdiff2 = np.linalg.norm(Fx_min_new - Fx_min)
        absdiff3 = np.linalg.norm(Fy_plus_new - Fy_plus)
        absdiff4 = np.linalg.norm(Fy_min_new - Fy_min)

        # avoid divide-by-zero blow-ups
        eps = 1e-12
        reldiff1 = np.linalg.norm((Fx_plus_new - Fx_plus)/(Fx_plus + eps))
        reldiff2 = np.linalg.norm((Fx_min_new - Fx_min)/(Fx_min + eps))
        reldiff3 = np.linalg.norm((Fy_plus_new - Fy_plus)/(Fy_plus + eps))
        reldiff4 = np.linalg.norm((Fy_min_new - Fy_min)/(Fy_min + eps))

        c1 = (absdiff1 < atol) and (absdiff2 < atol) and (absdiff3 < atol) and (absdiff4 < atol)
        c2 = (reldiff1 < rtol) and (reldiff2 < rtol) and (reldiff3 < rtol) and (reldiff4 < rtol)
        corr_new = [Fx_plus_new, Fx_min_new, Fy_plus_new, Fy_min_new]

        if verbose:
            abs_diff = [absdiff1, absdiff2, absdiff3, absdiff4]
            rel_diff = [reldiff1, reldiff2, reldiff3, reldiff4]

            print('============================================')
            print(f"Iteration {iteration + 1}:")
            correlation_strings = [
                "F0", "F_xplus", "F_xmin", "F_yplus", "F_ymin"]
            for idx in range(len(abs_diff)):
                print(f"{correlation_strings[idx]}:")
                print(f"Absolute error: {abs_diff[idx]}")
                print(f"Relative error: {rel_diff[idx]}")
                print(f"Average: {np.mean(corr_new[idx])}")

        if c1 and c2:
            print(f"Converged after {iteration+1} iterations")
            break
    corr = [Fx_plus, Fx_min, Fy_plus, Fy_min]
    H.set_correlations(corr)
