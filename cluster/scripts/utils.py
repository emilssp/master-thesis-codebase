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

def neighbor_maps_square(lattice: SquareLattice):
    X, Y = lattice.X, lattice.Y
    N = lattice.num_sites

    nbr = {"+x": -cp.ones(N, dtype=int),
           "-x": -cp.ones(N, dtype=int),
           "+y": -cp.ones(N, dtype=int),
           "-y": -cp.ones(N, dtype=int)}

    for y in range(Y):
        for x in range(X):
            i = y * X + x

            # +x
            if x < X-1:
                nbr["+x"][i] = i + 1
            elif lattice.pbc_x and X > 2:
                nbr["+x"][i] = y * X

            # -x
            if x > 0:
                nbr["-x"][i] = i - 1
            elif lattice.pbc_x and X > 2:
                nbr["-x"][i] = y * X + (X-1)

            # +y 
            if y < Y-1:
                nbr["+y"][i] = (y+1) * X + x
            elif lattice.pbc_y and Y > 2:
                nbr["+y"][i] = x

            # -y
            if y > 0:
                nbr["-y"][i] = (y-1) * X + x
            elif lattice.pbc_y and Y > 2:
                nbr["-y"][i] = (Y-1) * X + x

    return nbr


def gather_at(j_idx, arr):
    # arr is (Neig,N). Return (Neig,N) with invalid neighbors zeroed.
    valid = (j_idx >= 0)
    j_safe = cp.where(valid, j_idx, 0)
    out = arr[:, j_safe]
    return out * valid[cp.newaxis, :]  # zero invalid sites


def correlators(H, T):

    nbr = neighbor_maps_square(H.lattice)
    gap = H.gap

    F0 = cp.zeros_like(gap)
    F = {}          # singlet-like (up at i, down at j)
    Fuu = {}        # triplet component uu
    Fdd = {}        # triplet component dd


    eigval, eigvec = H.diagonalize(drop_matrix=False)
    eigvec = eigvec[:, eigval >= 0]
    eigval = eigval[eigval >= 0]
    eigvec = eigvec.T.reshape((eigval.size, -1, 4))

    f = fermi_dirac(eigval, T)          # (Neig,)
    u_up = eigvec[:, :, 0]              # (Neig, Nsites)
    u_dn = eigvec[:, :, 1]
    v_up = eigvec[:, :, 2]
    v_dn = eigvec[:, :, 3]

    F0 = (
        cp.einsum('ni,ni,n->i', u_dn, cp.conj(v_up), f) +
        cp.einsum('ni,ni,n->i', u_up, cp.conj(v_dn), (1-f))
    )
    
    for d in ["+x", "-x", "+y", "-y"]:
        j = nbr[d]

        vdn_j = gather_at(j, v_dn)
        udn_j = gather_at(j, u_dn)
        vup_j = gather_at(j, v_up)
        uup_j = gather_at(j, u_up)

        F[d] = (
            cp.einsum('ni,ni,n->i', u_up, cp.conj(vdn_j), (1 - f)) +
            cp.einsum('ni,ni,n->i', v_up, cp.conj(udn_j), f)
        )
        # Eqn 2: F_{i,↑↑}^d
        Fuu[d] = (
            cp.einsum('ni,ni,n->i', u_up, cp.conj(vup_j), (1 - f)) +
            cp.einsum('ni,ni,n->i', v_up, cp.conj(uup_j), f)
        )
        # Eqn 3: F_{i,↓↓}^d
        Fdd[d] = (
            cp.einsum('ni,ni,n->i', u_dn, cp.conj(vdn_j), (1 - f)) +
            cp.einsum('ni,ni,n->i', v_dn, cp.conj(udn_j), f)
        )
    return F0, F, Fuu, Fdd

def bdg_self_consistency(H, U, V, V_prime, 
                         T, max_iter=100,
                         atol=1e-3, rtol=1e-2,
                         with_corr = False):
        flag = False
        num_sites = H.lattice.num_sites

        for iteration in range(max_iter):
            F0, F, Fuu, Fdd = correlators(H, T)
            gap_0 = -U * F0
            gap_s = V * (F["+x"] + F["-x"] + F["+y"] + F["-y"])/4
            gap_d = V * (F["+x"] + F["-x"] - F["+y"] - F["-y"])/4
            gap_px = V * (F["+x"] - F["-x"])/2
            gap_py = V * (F["+y"] - F["-y"])/2
            gap_px_uu = V_prime * (Fuu["+x"] - Fuu["-x"])/2
            gap_py_uu = V_prime * (Fuu["+y"] - Fuu["-y"])/2
            gap_px_dd = V_prime * (Fdd["+x"] - Fdd["-x"])/2
            gap_py_dd = V_prime * (Fdd["+y"] - Fdd["-y"])/2
            
            for i in range(num_sites):
                # indices for the 4x4 block of site i
                sl = slice(4*i, 4*(i+1))
                H.matrix[sl, sl][:2, 2:] = -1j * gap_0[i] * s2
                H.matrix[sl, sl][2:, :2] = (-1j * gap_0[i] * s2).conj().T

            for i, j in H.lattice.edges:
                sli = slice(4*i, 4*(i+1))
                slj = slice(4*j, 4*(j+1))
                gap_p = cp.array([gap_px[i], gap_py[i]])
                gap_p_uu = cp.array([gap_px_uu[i], gap_py_uu[i]])
                gap_p_dd = cp.array([gap_px_dd[i], gap_py_dd[i]])

                gap_ij = unconventional_gap(H.lattice.get_disp(i, j), gap_s=gap_s[i], gap_d=gap_d[i],
                                            gap_p=gap_p, gap_p_dd=gap_p_dd, gap_p_uu=gap_p_uu)
                H.matrix[sli, slj][:2, 2:] =  gap_ij
                H.matrix[sli, slj][2:, :2] = -gap_ij.T.conj()

            if (cp.max(cp.abs(gap_0 - H.gap)) < atol + rtol * cp.max(cp.abs(gap_0))
                and cp.max(cp.abs(gap_s - H.gap_s)) < atol + rtol * cp.max(cp.abs(gap_s))
                and cp.max(cp.abs(gap_d - H.gap_d)) < atol + rtol * cp.max(cp.abs(gap_d))
                and cp.max(cp.abs(gap_px - H.gap_px)) < atol + rtol * cp.max(cp.abs(gap_px))
                and cp.max(cp.abs(gap_py - H.gap_py)) < atol + rtol * cp.max(cp.abs(gap_py))
                and cp.max(cp.abs(gap_px_uu - H.gap_px_uu)) < atol + rtol * cp.max(cp.abs(gap_px_uu))
                and cp.max(cp.abs(gap_py_uu - H.gap_py_uu)) < atol + rtol * cp.max(cp.abs(gap_py_uu))
                and cp.max(cp.abs(gap_px_dd - H.gap_px_dd)) < atol + rtol * cp.max(cp.abs(gap_px_dd))
                and cp.max(cp.abs(gap_py_dd - H.gap_py_dd)) < atol + rtol * cp.max(cp.abs(gap_py_dd))
                ):
                print(f"Converged after {iteration+1} iterations.")
                H.gap = gap_0
                H.gap_s = gap_s
                H.gap_d = gap_d
                H.gap_px = gap_px
                H.gap_py = gap_py
                H.gap_px_uu = gap_px_uu
                H.gap_py_uu = gap_py_uu
                H.gap_px_dd = gap_px_dd
                H.gap_py_dd = gap_py_dd
                flag = True
                break
            H.gap = gap_0
            H.gap_s = gap_s
            H.gap_d = gap_d
            H.gap_px = gap_px
            H.gap_py = gap_py
            H.gap_px_uu = gap_px_uu
            H.gap_py_uu = gap_py_uu
            H.gap_px_dd = gap_px_dd
            H.gap_py_dd = gap_py_dd

        if not flag:
            print(f"Failed to converge after {iteration+1} iterations")
        if with_corr:
            return F0, F, Fuu, Fdd