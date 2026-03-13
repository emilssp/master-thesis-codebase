import cupy as cp  # type: ignore
from .constants import *
from .hamiltonian import *


def correlators(H, T):
    N = H.lattice.num_sites
    F0 = cp.zeros_like(H.gap)
    F = cp.zeros((N, N), dtype=cp.complex128)
    Fuu = cp.zeros((N, N), dtype=cp.complex128)
    Fdd = cp.zeros((N, N), dtype=cp.complex128)

    eigval, eigvec = H.diagonalize(drop_matrix=False)
    eigvec = eigvec[:, eigval >= 0]
    eigval = eigval[eigval >= 0]
    eigvec = eigvec.T.reshape((eigval.size, -1, 4))

    f = fermi_dirac(eigval, T)          # (Neig,)
    u_up = eigvec[:, :, 0]  # u             # (Neig, Nsites)
    u_dn = eigvec[:, :, 1]  # v
    v_up = eigvec[:, :, 2]  # w
    v_dn = eigvec[:, :, 3]  # x

    F0 = (
        cp.einsum('ni,ni,n->i', u_dn, cp.conj(v_up), f) +
        cp.einsum('ni,ni,n->i', u_up, cp.conj(v_dn), (1-f))
    )

    for i, j in H.lattice.edges:

        # Eqn 1: F_{ij,↑↓}^d
        F[i][j] = (
            cp.einsum('n,n,n->', u_up[:, i], cp.conj(v_dn[:, j]), (1 - f)) +
            cp.einsum('n,n,n->', cp.conj(v_up[:, i]), u_dn[:, j], f)
        )
        # Eqn 1: F_{ji,↑↓}^d
        F[j][i] = (
            cp.einsum('n,n,n->', u_up[:, j], cp.conj(v_dn[:, i]), (1 - f)) +
            cp.einsum('n,n,n->', cp.conj(v_up[:, j]), u_dn[:, i], f)
        )
        # Eqn 2: F_{ij,↑↑}^d
        Fuu[i][j] = (
            cp.einsum('n,n,n->', u_up[:, i], cp.conj(v_up[:, j]), (1 - f)) +
            cp.einsum('n,n,n->', cp.conj(v_up[:, i]), u_up[:, j], f)
        )
        # Eqn 2: F_{ji,↑↑}^d
        Fuu[j][i] = (
            cp.einsum('n,n,n->', u_up[:, j], cp.conj(v_up[:, i]), (1 - f)) +
            cp.einsum('n,n,n->', cp.conj(v_up[:, j]), u_up[:, i], f)
        )
        # Eqn 3: F_{ij,↓↓}^d
        Fdd[i][j] = (
            cp.einsum('n,n,n->', u_dn[:, i], cp.conj(v_dn[:, j]), (1 - f)) +
            cp.einsum('n,n,n->', cp.conj(v_dn[:, j]), u_dn[:, i], f)
        )
        # Eqn 3: F_{ji,↓↓}^d
        Fdd[j][i] = (
            cp.einsum('n,n,n->', u_dn[:, j], cp.conj(v_dn[:, i]), (1 - f)) +
            cp.einsum('n,n,n->', cp.conj(v_dn[:, i]), u_dn[:, j], f)
        )
    return F0, F, Fuu, Fdd


def arr_converged(new, old, atol, rtol,
                  nonzero_threshold=1e-6):
    abs = cp.linalg.norm(new - old) < atol
    rel = (
        cp.linalg.norm((new-old)/(new + nonzero_threshold)) < rtol
        or cp.linalg.norm(new) < nonzero_threshold)
    return abs and rel


def bdg_self_consistency_step(H,  # Hamiltonian
                              U=None, V=None, V_prime=None, T=0,  # Parameters
                              atol=1e-6, rtol=1e-4, mix=1.0):  # Tolerances
    F0, F, Fuu, Fdd = H.get_correlations()
    F0_new, F_new, Fuu_new, Fdd_new = correlators(H, T)
    F0_new = mix * F0_new + (1-mix) * F0
    F_new = mix * F_new + (1-mix) * F
    Fuu_new = mix * Fuu_new + (1-mix) * Fuu
    Fdd_new = mix * Fdd_new + (1-mix) * Fdd

    for i in range(H.lattice.num_sites):
        # indices for the 4x4 block of site i
        sl = slice(4*i, 4*(i+1))
        H.matrix[sl, sl][:2, 2:] = 1j * U[i] * F0_new[i] * s2
        H.matrix[sl, sl][2:, :2] = (1j * U[i] * F0_new[i] * s2).conj().T

    for i, j in H.lattice.edges:
        sli = slice(4*i, 4*(i+1))
        slj = slice(4*j, 4*(j+1))

        H.matrix[sli, slj][0, 3] = V[j] * F_new[i][j]
        H.matrix[sli, slj][1, 2] = V[j] * F_new[i][j]
        H.matrix[sli, slj][2, 1] = (V[j] * F_new[j][i]).conj().T
        H.matrix[sli, slj][3, 0] = (V[j] * F_new[j][i]).conj().T

        H.matrix[slj, sli][0, 2] = -V_prime[j] * Fuu_new[i][j]
        H.matrix[slj, sli][1, 3] = -V_prime[j] * Fdd_new[i][j]
        H.matrix[slj, sli][2, 0] = (-V_prime[j] * Fuu_new[j][i]).conj().T
        H.matrix[slj, sli][3, 1] = (-V_prime[j] * Fdd_new[j][i]).conj().T

    if (arr_converged(F0_new, F0, atol, rtol)
            and arr_converged(F_new, F, atol, rtol)
            and arr_converged(Fuu_new, Fuu, atol, rtol)
            and arr_converged(Fdd_new, Fdd, atol, rtol)):
        flag = True
    else:
        flag = False
    return flag, F0_new, F_new, Fuu_new, Fdd_new


def bdg_self_consistency(H, U=None, V=None, V_prime=None,
                         T=0, max_iter=100,
                         atol=1e-6, rtol=1e-4,
                         with_corr=False,
                         mix=1.0, verbose=False):
    if U is None:
        U = cp.zeros(H.lattice.num_sites)
    if V is None:
        V = cp.zeros(H.lattice.num_sites)
    if V_prime is None:
        V_prime = cp.zeros(H.lattice.num_sites)

    flag = False
    num_sites = H.lattice.num_sites
    F0 = cp.zeros_like(H.gap)
    F = cp.zeros((num_sites, num_sites), dtype=cp.complex128)
    Fuu = cp.zeros((num_sites, num_sites), dtype=cp.complex128)
    Fdd = cp.zeros((num_sites, num_sites), dtype=cp.complex128)

    for iteration in range(max_iter):
        flag, F0, F, Fuu, Fdd = bdg_self_consistency_step(H, U, V, V_prime,
                                                          T, atol, rtol, mix)
        if verbose:
            print(f"Iteration {iteration+1}:\
                \nAbs: {cp.linalg.norm(F - H.F)}\
                \nRel: {cp.linalg.norm((F - H.F)/(F+1e-12))}\
                \nIndex: {cp.argmax(cp.abs(F.flatten() - H.F.flatten()))}\
                \nCorrelation: \
                {H.F.flatten()[cp.argmax(cp.abs(F.flatten() - H.F.flatten()))]}\
                \nCorr_new: \
                {F.flatten()[cp.argmax(cp.abs(F.flatten() - H.F.flatten()))]}\
                \nCorrelation_dd: \
                    {H.Fdd.flatten()[cp.argmax(cp.abs(Fdd.flatten() - H.Fdd.flatten()))]}\
                \nCorr_dd_new: \
                    {Fdd.flatten()[cp.argmax(cp.abs(Fdd.flatten() - H.Fdd.flatten()))]}\
                \nCorrelation_uu: \
                    {H.Fuu.flatten()[cp.argmax(cp.abs(Fuu.flatten() - H.Fuu.flatten()))]}\
                \nCorr_uu_new: \
                    {Fuu.flatten()[cp.argmax(cp.abs(Fuu.flatten() - H.Fuu.flatten()))]}")
            print(f"F0 bool: {arr_converged(F0, H.F0, atol, rtol)}")
            print(f"F bool: {arr_converged(F, H.F, atol, rtol)}")
            print(f"Fuu bool: {arr_converged(Fuu, H.Fuu, atol, rtol)}")
            print(f"Fdd bool: {arr_converged(Fdd, H.Fdd, atol, rtol)}")

        H.set_F0(F0)
        H.set_F(F)
        H.set_Fuu(Fuu)
        H.set_Fdd(Fdd)

        if flag:
            print(f"Converged after {iteration+1} iterations.")
            break

    if not flag:
        print(f"Failed to converge after {iteration+1} iterations")
    if with_corr:
        return F0, F, Fuu, Fdd


def tc_search(H, Tmin, Tmax,
              U=0, V=0, V_prime=0,
              F0=1e-8, F=1e-8, Fuu=1e-8, Fdd=1e-8,
              tol=1e-6, max_iter=100):

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
