import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.linalg as la
from scipy import special
from collections import namedtuple

from .utils import sl, lorentzian
from .constants import PI
from .Lattice import Lattice

PairCorrelations = namedtuple(
    "PairCorrelations",
    [
        "swave", "dwave", "px", "py"
    ]
)


class Hamiltonian:
    def __init__(self, t, mu, lattice: Lattice,
                 U=None, V=None, V_prime=None,
                 F0_init=0, F_init=np.zeros(4),
                 Fuu_init=np.zeros(4),
                 Fdd_init=np.zeros(4),
                 hx=None, hy=None, hz=None):
        '''
        Class that stores the Hamiltonian parameters and initial conditions,
        as well as functions constructing the Hamiltonian and extracting re-
        levant quantities.
        '''
        self.t = t
        self.mu = mu
        self.lattice = lattice
        Nx = lattice.X
        self.dim = 4 * Nx
        self.free = 0

        if U is None:
            U = np.zeros(Nx)
        if V is None:
            V = np.zeros(Nx)
        if V_prime is None:
            V_prime = np.zeros(Nx)

        if hx is None:
            hx = np.zeros(Nx)
        if hy is None:
            hy = np.zeros(Nx)
        if hz is None:
            hz = np.zeros(Nx)

        self.U = U
        self.V = V
        self.V_prime = V_prime

        self.hx = hx
        self.hy = hy
        self.hz = hz

        self.F0 = np.zeros(Nx, dtype=np.complex128)
        self.F0[np.where(U != 0)] = F0_init
        self.F_xplus = np.zeros(Nx-1, dtype=np.complex128)
        self.F_xplus[np.where(V[1:] != 0)] = F_init[0]
        self.F_xmin = np.zeros(Nx-1, dtype=np.complex128)
        self.F_xmin[np.where(V[:-1] != 0)] = F_init[1]
        self.F_yplus = np.zeros(Nx, dtype=np.complex128)
        self.F_yplus[np.where(V != 0)] = F_init[2]
        self.F_ymin = np.zeros(Nx, dtype=np.complex128)
        self.F_ymin[np.where(V != 0)] = F_init[3]

        self.Fuu_xplus = np.zeros(Nx-1, dtype=np.complex128)
        self.Fuu_xplus[np.where(V_prime[:-1] != 0)] = Fuu_init[0]
        self.Fuu_xmin = np.zeros(Nx-1, dtype=np.complex128)
        self.Fuu_xmin[np.where(V_prime[1:] != 0)] = Fuu_init[1]
        self.Fuu_yplus = np.zeros(Nx, dtype=np.complex128)
        self.Fuu_yplus[np.where(V_prime != 0)] = Fuu_init[2]
        self.Fuu_ymin = np.zeros(Nx, dtype=np.complex128)
        self.Fuu_ymin[np.where(V_prime != 0)] = Fuu_init[3]

        self.Fdd_xplus = np.zeros(Nx-1, dtype=np.complex128)
        self.Fdd_xplus[np.where(V_prime[:-1] != 0)] = Fdd_init[0]
        self.Fdd_xmin = np.zeros(Nx-1, dtype=np.complex128)
        self.Fdd_xmin[np.where(V_prime[1:] != 0)] = Fdd_init[1]
        self.Fdd_yplus = np.zeros(Nx, dtype=np.complex128)
        self.Fdd_yplus[np.where(V_prime != 0)] = Fdd_init[2]
        self.Fdd_ymin = np.zeros(Nx, dtype=np.complex128)
        self.Fdd_ymin[np.where(V_prime != 0)] = Fdd_init[3]

    def get_dim(self):
        return self.dim

    def build_H_k0(self):
        H = np.zeros((self.dim, self.dim), dtype=np.complex128)

        gap1 = self.V * (self.F_yplus + self.F_ymin)
        gap2 = -self.V * (self.F_yplus + self.F_ymin)

        for i in range(self.lattice.X):
            H[sl(i, 0), sl(i, 0)] = -2 * self.t
            H[sl(i, 1), sl(i, 1)] = -2 * self.t
            H[sl(i, 2), sl(i, 2)] = 2 * self.t
            H[sl(i, 3), sl(i, 3)] = 2 * self.t

            H[sl(i, 0), sl(i, 3)] = gap1[i]
            H[sl(i, 1), sl(i, 2)] = gap2[i]
            H[sl(i, 2), sl(i, 1)] = np.conj(gap2[i])
            H[sl(i, 3), sl(i, 0)] = np.conj(gap1[i])
        return H

    def build_H_kindep(self):
        gap0 = self.U * self.F0
        gap1 = self.V[:-1] * self.F_xmin
        gap2 = self.V[:-1] * self.F_xplus

        gap1_uu = self.V_prime[1:] * self.Fuu_xmin
        gap2_uu = self.V_prime[:-1] * self.Fuu_xplus
        gap1_dd = self.V_prime[1:] * self.Fdd_xmin
        gap2_dd = self.V_prime[:-1] * self.Fdd_xplus

        H = np.zeros((self.dim, self.dim), dtype=np.complex128)

        for i in range(self.lattice.X):
            H[sl(i, 0), sl(i, 0)] = -self.mu[i] + self.hz[i]
            H[sl(i, 1), sl(i, 1)] = -self.mu[i] - self.hz[i]
            H[sl(i, 2), sl(i, 2)] = self.mu[i] - self.hz[i]
            H[sl(i, 3), sl(i, 3)] = self.mu[i] + self.hz[i]

            H[sl(i, 0), sl(i, 1)] = self.hx[i] - 1j * self.hy[i]
            H[sl(i, 1), sl(i, 0)] = self.hx[i] + 1j * self.hy[i]
            H[sl(i, 2), sl(i, 3)] = -self.hx[i] - 1j * self.hy[i]
            H[sl(i, 3), sl(i, 2)] = -self.hx[i] + 1j * self.hy[i]

            H[sl(i, 0), sl(i, 3)] = gap0[i]
            H[sl(i, 1), sl(i, 2)] = -gap0[i]
            H[sl(i, 2), sl(i, 1)] = np.conj(-gap0[i])
            H[sl(i, 3), sl(i, 0)] = np.conj(gap0[i])

        for i in range(self.lattice.X - 1):
            for c, hop in enumerate([-self.t, -self.t, self.t, self.t]):
                H[sl(i, c),   sl(i + 1, c)] = hop
                H[sl(i + 1, c), sl(i, c)] = hop

            # upper
            H[sl(i, 0), sl(i+1, 3)] = gap2[i]
            H[sl(i, 1), sl(i+1, 2)] = -gap1[i]
            H[sl(i, 2), sl(i+1, 1)] = np.conj(-gap2[i])
            H[sl(i, 3), sl(i+1, 0)] = np.conj(gap1[i])

            H[sl(i, 0), sl(i+1, 2)] = gap2_uu[i]
            H[sl(i, 1), sl(i+1, 3)] = gap2_dd[i]
            H[sl(i, 2), sl(i+1, 0)] = np.conj(gap1_uu[i])
            H[sl(i, 3), sl(i+1, 1)] = np.conj(gap1_dd[i])

            # lower
            H[sl(i+1, 0), sl(i, 3)] = gap1[i]
            H[sl(i+1, 1), sl(i, 2)] = -gap2[i]
            H[sl(i+1, 2), sl(i, 1)] = np.conj(-gap1[i])
            H[sl(i+1, 3), sl(i, 0)] = np.conj(gap2[i])

            H[sl(i+1, 0), sl(i, 2)] = gap1_uu[i]
            H[sl(i+1, 1), sl(i, 3)] = gap1_dd[i]
            H[sl(i+1, 2), sl(i, 0)] = np.conj(gap2_uu[i])
            H[sl(i+1, 3), sl(i, 1)] = np.conj(gap2_dd[i])

        return H

    def build_H_k(self, k):
        H2 = np.zeros((self.dim, self.dim), dtype=np.complex128)

        epsilon = -2 * self.t * np.cos(k)
        ep = np.exp(1j * k)
        em = np.exp(-1j * k)

        # y-pairing with phase
        gap1 = self.V * (self.F_yplus * em + self.F_ymin * ep)
        gap2 = -self.V * (self.F_yplus * ep + self.F_ymin * em)
        gap_uu = -2j * self.V_prime * self.Fuu_yplus * np.sin(k)
        gap_dd = -2j * self.V_prime * self.Fdd_yplus * np.sin(k)

        for i in range(self.lattice.X):
            # dispersion
            H2[sl(i, 0), sl(i, 0)] = epsilon
            H2[sl(i, 1), sl(i, 1)] = epsilon
            H2[sl(i, 2), sl(i, 2)] = -epsilon
            H2[sl(i, 3), sl(i, 3)] = -epsilon

            H2[sl(i, 0), sl(i, 3)] = gap1[i]
            H2[sl(i, 1), sl(i, 2)] = gap2[i]
            H2[sl(i, 2), sl(i, 1)] = np.conj(gap2[i])
            H2[sl(i, 3), sl(i, 0)] = np.conj(gap1[i])

            H2[sl(i, 0), sl(i, 2)] = gap_uu[i]
            H2[sl(i, 1), sl(i, 3)] = gap_dd[i]
            H2[sl(i, 2), sl(i, 0)] = np.conj(gap_uu[i])  # TODO: Check sign
            H2[sl(i, 3), sl(i, 1)] = np.conj(gap_dd[i])

        return H2

    def set_correlations(self, corr):
        self.F0 = corr[0].copy()

        self.F_xplus = corr[1].copy()
        self.F_xmin = corr[2].copy()
        self.F_yplus = corr[3].copy()
        self.F_ymin = corr[4].copy()

        self.Fuu_xplus = corr[5].copy()
        self.Fuu_xmin = corr[6].copy()
        self.Fuu_yplus = corr[7].copy()
        self.Fuu_ymin = corr[8].copy()

        self.Fdd_xplus = corr[9].copy()
        self.Fdd_xmin = corr[10].copy()
        self.Fdd_yplus = corr[11].copy()
        self.Fdd_ymin = corr[12].copy()

    def get_correlations(self):
        # Correlations ↑↓
        FS_x = (self.F_xplus + self.F_xmin) / 2
        FS_y = (self.F_yplus + self.F_ymin) / 2

        FT_x_plus = (self.F_xplus - self.F_xmin) / 2
        FT_x_min = (self.F_xmin - self.F_xplus) / 2

        FT_y_plus = (self.F_yplus - self.F_ymin) / 2
        FT_y_min = (self.F_ymin - self.F_yplus) / 2

        F_swave = (np.r_[0, FS_x] + np.r_[FS_x, 0] + 2.0 * FS_y) / 4.0
        F_dwave = (np.r_[0, FS_x] + np.r_[FS_x, 0] - 2.0 * FS_y) / 4.0
        F_px = (np.r_[0, FT_x_plus] - np.r_[FT_x_min, 0]) / 2.0
        F_py = (FT_y_plus - FT_y_min) / 2.0

        F = PairCorrelations(
            F_swave, F_dwave, F_px, F_py
        )

        # Correlations ↑↑ and ↓↓
        Fuu_x_plus = (self.Fuu_xplus - self.Fuu_xmin) / 2
        Fuu_x_min = (self.Fuu_xmin - self.Fuu_xplus) / 2

        Fuu_y_plus = (self.Fuu_yplus - self.Fuu_ymin) / 2
        Fuu_y_min = (self.Fuu_ymin - self.Fuu_yplus) / 2

        Fdd_x_plus = (self.Fdd_xplus - self.Fdd_xmin) / 2
        Fdd_x_min = (self.Fdd_xmin - self.Fdd_xplus) / 2

        Fdd_y_plus = (self.Fdd_yplus - self.Fdd_ymin) / 2
        Fdd_y_min = (self.Fdd_ymin - self.Fdd_yplus) / 2

        Fuu_px = (np.r_[0, Fuu_x_plus] - np.r_[Fuu_x_min, 0]) / 2.0
        Fuu_py = (Fuu_y_plus - Fuu_y_min) / 2.0
        Fdd_px = (np.r_[0, Fdd_x_plus] - np.r_[Fdd_x_min, 0]) / 2.0
        Fdd_py = (Fdd_y_plus - Fdd_y_min) / 2.0

        Fuu = PairCorrelations(
            np.zeros_like(Fuu_px),
            np.zeros_like(Fuu_px),
            Fuu_px, Fuu_py
        )
        Fdd = PairCorrelations(
            np.zeros_like(Fdd_px),
            np.zeros_like(Fdd_px),
            Fdd_px, Fdd_py
        )

        return F, Fuu, Fdd

    def free_energy_const_term(self):

        E_S = 0

        E_S += np.sum(self.U * np.abs(self.F0)**2)

        E_S += np.sum(self.V * np.abs(np.r_[0, self.F_xplus])**2)
        E_S += np.sum(self.V * np.abs(np.r_[self.F_xmin, 0])**2)
        E_S += np.sum(self.V * np.abs(self.F_yplus)**2)
        E_S += np.sum(self.V * np.abs(self.F_ymin)**2)

        E_S += np.sum(self.V_prime * np.abs(np.r_[0, self.Fuu_xplus])**2)/2
        E_S += np.sum(self.V_prime * np.abs(np.r_[self.Fuu_xmin, 0])**2)/2
        E_S += np.sum(self.V_prime * np.abs(self.Fuu_yplus)**2)/2
        E_S += np.sum(self.V_prime * np.abs(self.Fuu_ymin)**2)/2

        E_S += np.sum(self.V_prime * np.abs(np.r_[0, self.Fdd_xplus])**2)/2
        E_S += np.sum(self.V_prime * np.abs(np.r_[self.Fdd_xmin, 0])**2)/2
        E_S += np.sum(self.V_prime * np.abs(self.Fdd_yplus)**2)/2
        E_S += np.sum(self.V_prime * np.abs(self.Fdd_ymin)**2)/2

        return E_S

    def free_energy(self, temperature=0):
        '''Calculates free energy and set it in H'''
        Ny = self.lattice.Y
        ky_list = np.linspace(PI / Ny, PI, Ny//2)

        H_kindep = self.build_H_kindep()
        H_k0 = self.build_H_k0()

        eigval0 = la.eigvalsh(
            H_kindep + H_k0,
            subset_by_value=(0, np.inf)
        )
        if not temperature == 0:
            beta = 1/temperature
            S = np.sum(special.softplus(-eigval0*beta))/beta
        else:
            S = 0
        free = -sum(eigval0)/2 - S

        for k in ky_list:
            H_k = self.build_H_k(k)
            eigval = la.eigvalsh(H_kindep + H_k)

            if not temperature == 0:
                beta = 1/temperature
                S = np.sum(special.softplus(-eigval*beta))/beta
            else:
                S = 0
            free = free - np.sum(eigval)/2 - S

        E_S = self.free_energy_const_term()

        free = free + E_S
        self.free = free
        return free

    def dos_k(self, energies, eta, H_kindep, k):

        H_k = self.build_H_k(k)
        H2 = H_kindep + H_k

        eigval, eigvec = la.eigh(H2)

        u_up = eigvec[0::4, :]  # electron ↑
        u_dn = eigvec[1::4, :]  # electron ↓
        v_up = eigvec[2::4, :]  # hole ↑
        v_dn = eigvec[3::4, :]  # hole ↓

        w_pos = np.sum(np.abs(u_up)**2 + np.abs(u_dn)**2, axis=0)  # (M,)
        w_neg = np.sum(np.abs(v_up)**2 + np.abs(v_dn)**2, axis=0)

        Lp = lorentzian(energies[:, None] - eigval[None, :], eta=eta)
        Ln = lorentzian(energies[:, None] + eigval[None, :], eta=eta)

        dos_values = (np.einsum('nm,m->n', Lp, w_pos)
                      + np.einsum('nm,m->n', Ln, w_neg))
        return dos_values

    def dos(self, energies, eta):
        Ny = self.lattice.Y
        ky_list = np.linspace(PI/Ny, PI, Ny, endpoint=False)

        H_kindep = self.build_H_kindep()
        H_k0 = self.build_H_k0()

        eigval0, eigvec0 = la.eigh(
            H_kindep + H_k0,
            subset_by_value=(0, np.inf)
        )

        u_up0 = eigvec0[0::4, :]
        u_dn0 = eigvec0[1::4, :]
        v_up0 = eigvec0[2::4, :]
        v_dn0 = eigvec0[3::4, :]

        w_pos = np.sum(np.abs(u_up0)**2 + np.abs(u_dn0)**2, axis=0)  # (M,)
        w_neg = np.sum(np.abs(v_up0)**2 + np.abs(v_dn0)**2, axis=0)

        Lp = lorentzian(energies[:, None] - eigval0[None, :], eta=eta)
        Ln = lorentzian(energies[:, None] + eigval0[None, :], eta=eta)

        dos = (np.einsum('nm,m->n', Lp, w_pos)
               + np.einsum('nm,m->n', Ln, w_neg))

        def work(k):
            return self.dos_k(energies, eta, H_kindep, k)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
            parts = list(ex.map(work, ky_list))

        dos += np.sum(parts, axis=0)
        return dos