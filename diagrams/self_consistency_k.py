import numpy as np
import scipy.linalg as la
from scipy import special
from collections import namedtuple

from utils import fermi_dirac, is_converged, is_hermitian

PI = np.pi

Corr = namedtuple(
    "Corr",
    [
        "F0",
        "F_xplus", "F_xmin", "F_yplus", "F_ymin",
        "Fuu_xplus", "Fuu_xmin", "Fuu_yplus", "Fuu_ymin",
        "Fdd_xplus", "Fdd_xmin", "Fdd_yplus", "Fdd_ymin",
    ]
)

Result = namedtuple(
    "Result",
    [
        "corr",
        "F_onsite",
        "F_swave",
        "F_dwave",
        "F_px",
        "F_py",
        "Fuu_px",
        "Fuu_py",
        "Fdd_px",
        "Fdd_py",
        "free_energy",
        "converged",
        "iterations"
    ]
)


def build_bdg_hamiltonian(kx, ky, t, mu, h,
                          F0, F, Fuu, Fdd,
                          U, V, V_prime):

    hx = h[0]
    hy = h[1]
    hz = h[2]

    eps = -2.0 * t * np.cos(kx) - 2.0 * t * np.cos(ky) - mu

    emx = np.exp(-1j * kx)
    epx = np.exp(1j * kx)
    emy = np.exp(-1j * ky)
    epy = np.exp(1j * ky)

    F_xplus = F[0]
    F_xmin = F[1]
    F_yplus = F[2]
    F_ymin = F[3]

    Fuu_x = Fuu[0]
    Fuu_y = Fuu[2]

    Fdd_x = Fdd[0]
    Fdd_y = Fdd[2]

    Delta0 = U * F0

    Fk1 = V * (F_xplus * emx + F_xmin * epx +
               F_yplus * emy + F_ymin * epy)
    Fk2 = -V * (F_xplus * epx + F_xmin * emx +
                F_yplus * epy + F_ymin * emy)
    Fk3 = -V * (np.conj(F_xplus) * emx + np.conj(F_xmin) * epx +
                np.conj(F_yplus) * emy + np.conj(F_ymin) * epy)
    Fk4 = np.conj(Fk1)

    Fuu_k = -2j * V_prime * (Fuu_x * np.sin(kx) +
                             Fuu_y * np.sin(ky))

    Fdd_k = -2j * V_prime * (Fdd_x * np.sin(kx) +
                             Fdd_y * np.sin(ky))

    H = np.zeros((4, 4), dtype=np.complex128)

    H[0, 0] = eps + hz
    H[0, 1] = hx - 1j * hy
    H[0, 2] = Fuu_k
    H[0, 3] = Delta0 + Fk1

    H[1, 0] = hx + 1j * hy
    H[1, 1] = eps - hz
    H[1, 2] = -Delta0 + Fk2
    H[1, 3] = Fdd_k

    H[2, 0] = np.conj(Fuu_k)
    H[2, 1] = -np.conj(Delta0) + Fk3
    H[2, 2] = -eps - hz
    H[2, 3] = -(hx + 1j * hy)

    H[3, 0] = np.conj(Delta0) + Fk4
    H[3, 1] = np.conj(Fdd_k)
    H[3, 2] = -(hx - 1j * hy)
    H[3, 3] = -eps + hz

    if not is_hermitian(H):
        raise ValueError("Hamiltonian is not Hermitian")
    return H


def bdg_sc_full_k(t, mu, Nx, Ny, temperature=0, h=np.zeros(3),
                  U=0, V=0, V_prime=0, F0_init=0j,
                  F_init=np.zeros(4, dtype=np.complex128),
                  Fuu_init=np.zeros(4, dtype=np.complex128),
                  Fdd_init=np.zeros(4, dtype=np.complex128),
                  atol=1e-6, rtol=1e-4, maxiter=100,
                  verbose=False, verbose_free=False):

    Nk = Nx * Ny
    kx_list = 2.0 * PI * np.arange(Nx) / Nx - PI
    ky_list = 2.0 * PI * np.arange(Ny) / Ny - PI

    F0 = F0_init
    F = F_init
    Fuu = Fuu_init
    Fdd = Fdd_init

    converged = False
    free_energy = None

    for iteration in range(maxiter):
        F0_new = 0.0j
        F_xplus_new = 0.0j
        F_xmin_new = 0.0j
        F_yplus_new = 0.0j
        F_ymin_new = 0.0j

        Fuu_xplus_new = 0.0j
        Fuu_xmin_new = 0.0j
        Fuu_yplus_new = 0.0j
        Fuu_ymin_new = 0.0j

        Fdd_xplus_new = 0.0j
        Fdd_xmin_new = 0.0j
        Fdd_yplus_new = 0.0j
        Fdd_ymin_new = 0.0j

        free = 0.0

        for kx in kx_list:
            for ky in ky_list:
                Hk = build_bdg_hamiltonian(kx, ky, t, mu, h,
                                           F0, F, Fuu, Fdd,
                                           U, V, V_prime)

                eigval, eigvec = la.eigh(
                    Hk,
                    subset_by_value=(0, np.inf)
                )

                u_up = eigvec[0, :]
                u_dn = eigvec[1, :]
                v_up = eigvec[2, :]
                v_dn = eigvec[3, :]

                f_E = fermi_dirac(eigval, temperature)

                if temperature != 0:
                    beta = 1.0 / temperature
                    S = np.sum(special.softplus(-eigval * beta)) / beta
                else:
                    S = 0.0

                free += -0.5 * np.sum(eigval) - S

                epx = np.exp(1j * kx)
                emx = np.exp(-1j * kx)
                epy = np.exp(1j * ky)
                emy = np.exp(-1j * ky)

                # opposite-spin onsite
                F_ux = np.sum(u_up * np.conj(v_dn) * (1.0 - f_E))
                F_vw = np.sum(u_dn * np.conj(v_up) * f_E)

                # opposite-spin bond-resolved
                Fx_ux1 = F_ux * epx
                Fx_ux2 = F_ux * emx
                Fx_vw1 = F_vw * epx
                Fx_vw2 = F_vw * emx

                Fy_ux1 = F_ux * epy
                Fy_ux2 = F_ux * emy
                Fy_vw1 = F_vw * epy
                Fy_vw2 = F_vw * emy

                Fuu0 = (
                    np.sum(u_up * np.conj(v_up) * (1.0 - f_E))
                    + np.sum(u_up * np.conj(v_up) * f_E)
                )

                Fdd0 = (
                    np.sum(u_dn * np.conj(v_dn) * (1.0 - f_E))
                    + np.sum(u_dn * np.conj(v_dn) * f_E)
                )

                F0_new += (F_ux + F_vw) / Nk

                F_xplus_new += (Fx_ux1 + Fx_vw2) / Nk
                F_xmin_new += (Fx_ux2 + Fx_vw1) / Nk
                F_yplus_new += (Fy_ux1 + Fy_vw2) / Nk
                F_ymin_new += (Fy_ux2 + Fy_vw1) / Nk

                Fuu_xplus_new += Fuu0 * epx / Nk
                Fuu_xmin_new += Fuu0 * emx / Nk
                Fuu_yplus_new += Fuu0 * epy / Nk
                Fuu_ymin_new += Fuu0 * emy / Nk

                Fdd_xplus_new += Fdd0 * epx / Nk
                Fdd_xmin_new += Fdd0 * emx / Nk
                Fdd_yplus_new += Fdd0 * epy / Nk
                Fdd_ymin_new += Fdd0 * emy / Nk

        corr = Corr(
            F0, F[0], F[1], F[2], F[3],
            Fuu[0], Fuu[1], Fuu[2], Fuu[3],
            Fdd[0], Fdd[1], Fdd[2], Fdd[3]
        )

        corr_new = Corr(
            F0_new,
            F_xplus_new, F_xmin_new, F_yplus_new, F_ymin_new,
            Fuu_xplus_new, Fuu_xmin_new, Fuu_yplus_new, Fuu_ymin_new,
            Fdd_xplus_new, Fdd_xmin_new, Fdd_yplus_new, Fdd_ymin_new
        )

        F0 = F0_new

        F[0] = F_xplus_new
        F[1] = F_xmin_new
        F[2] = F_yplus_new
        F[3] = F_ymin_new

        Fuu[0] = Fuu_xplus_new
        Fuu[1] = Fuu_xmin_new
        Fuu[2] = Fuu_yplus_new
        Fuu[3] = Fuu_ymin_new

        Fdd[0] = Fdd_xplus_new
        Fdd[1] = Fdd_xmin_new
        Fdd[2] = Fdd_yplus_new
        Fdd[3] = Fdd_ymin_new

        F_swave = 0.25 * (F_xplus_new + F_xmin_new + F_yplus_new + F_ymin_new)
        F_dwave = 0.25 * (F_xplus_new + F_xmin_new - F_yplus_new - F_ymin_new)
        F_px = 0.5 * (F_xplus_new - F_xmin_new)
        F_py = 0.5 * (F_yplus_new - F_ymin_new)

        Fuu_px = 0.5 * (Fuu_xplus_new - Fuu_xmin_new)
        Fuu_py = 0.5 * (Fuu_yplus_new - Fuu_ymin_new)

        Fdd_px = 0.5 * (Fdd_xplus_new - Fdd_xmin_new)
        Fdd_py = 0.5 * (Fdd_yplus_new - Fdd_ymin_new)

        if verbose:
            print("============================================")
            names = Corr._fields
            for name, old, new in zip(names, corr, corr_new):
                abs_diff = np.abs(new - old)
                rel_diff = abs_diff / (np.abs(old) + 1e-12)
                print(f"{name}:")
                print(f"old = {old}")
                print(f"new = {new}")
                print(f"abs error = {abs_diff}")
                print(f"rel error = {rel_diff}")
            print(f"Iteration {iteration + 1}")
            print(f"F_swave = {F_swave}")
            print(f"F_dwave = {F_dwave}")
            print(f"F_px = {F_px}")
            print(f"F_py = {F_py}")
            print(f"Fuu_px = {Fuu_px}")
            print(f"Fuu_py = {Fuu_py}")
            print(f"Fdd_px = {Fdd_px}")
            print(f"Fdd_py = {Fdd_py}")

        if is_converged(corr, corr_new, atol=atol, rtol=rtol):
            if verbose:
                print("==================================================")
                print(f"Converged after {iteration + 1} iterations")
                print("==================================================")
            corr = corr_new
            free_energy = free
            converged = True
            break

        corr = corr_new
        free_energy = free

    E_S = 0
    E_S += np.sum(U * np.abs(F0)**2)

    E_S += np.sum(V * np.abs(F[0])**2)  # * Nk
    E_S += np.sum(V * np.abs(F[1])**2)  # * Nk
    E_S += np.sum(V * np.abs(F[2])**2)  # * Nk
    E_S += np.sum(V * np.abs(F[3])**2)  # * Nk

    E_S += np.sum(V_prime * np.abs(Fuu[0])**2)  # * Nk
    E_S += np.sum(V_prime * np.abs(Fuu[1])**2)  # * Nk
    E_S += np.sum(V_prime * np.abs(Fuu[2])**2)  # * Nk
    E_S += np.sum(V_prime * np.abs(Fuu[3])**2)  # * Nk

    E_S += np.sum(V_prime * np.abs(Fdd[0])**2)  # * Nk
    E_S += np.sum(V_prime * np.abs(Fdd[1])**2)  # * Nk
    E_S += np.sum(V_prime * np.abs(Fdd[2])**2)  # * Nk
    E_S += np.sum(V_prime * np.abs(Fdd[3])**2)  # * Nk

    if verbose_free:
        print("==================================================")
        print(f"Free energy = {free_energy}")
        print(f"S = {S}")
        print(f"E_S = {E_S}")
        print(f"Total free energy = {free_energy + E_S}")
        print("==================================================")
    free_energy += E_S
    if not converged:
        print("==================================================")
        print(f"WARNING: Failed to converge after {maxiter} iterations")
        print("==================================================")

    return Result(
        corr=corr,
        F_onsite=F0,
        F_swave=F_swave,
        F_dwave=F_dwave,
        F_px=F_px,
        F_py=F_py,
        Fuu_px=Fuu_px,
        Fuu_py=Fuu_py,
        Fdd_px=Fdd_px,
        Fdd_py=Fdd_py,
        free_energy=free_energy,  # -4635.626493896028
        converged=converged,
        iterations=iteration + 1,
    )
