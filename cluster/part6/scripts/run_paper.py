import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.partial_sc import partial_sc
from utilsdir.self_consistency_ft.self_consistency_k import bdg_sc_full_k

V_prime_arr = np.concatenate([
    np.arange(0.5, 0.75 + 1e-12, 0.05),
    np.arange(0.75, 1 + 1e-12, 0.01),
    np.arange(1.0, 4.0 + 1e-12, 0.05),
])  # 93 vprimes total

t = 1.0
T = 1e-5 * t

widthX = 10
widthF = 1
widthP = 40

muD = 0.0 * t
muS = 1.2 * t
muP = 1.8 * t
muF = 1.4 * t

# Lattice parameters
Nx, Ny = 100, 100

# Tolerances and maximum iterations for self-consistency
atol = 1e-6
rtol = 1e-3
maxiter = 5000

X, Y = widthX+widthF+widthP+widthF+widthX, 200
lattice = Lattice(X, Y)

def calculate_xfpfx(V_prime0, muX, widthX, fixed_syms = [],
                    s_val=0, d_val=0, U0=0, V0=0):
    mu = np.zeros(lattice.X)
    mu[:widthX] = muX
    mu[widthX:widthX + widthF] = muF
    mu[widthX + widthF:widthX + widthF + widthP] = muP
    mu[widthX + widthF + widthP:widthX + widthF + widthP + widthF] = muF
    mu[widthX + widthF + widthP + widthF:widthX + widthF + widthP + widthF + widthX] = muX

    h0 = 0.9 * t
    h = np.zeros_like(mu)
    h[widthX:widthX + widthF] = h0
    h[widthX + widthF + widthP + widthF-1] = -h0

    V = V0 * np.ones(lattice.X)
    V[widthX:-widthX] = 0
    V[widthX-1] = V0/2
    V[-widthX] = V0/2

    U = U0 * np.ones(lattice.X)
    U[widthX:-widthX] = 0
    U[widthX-1] = U0/2
    U[-widthX] = U0/2

    V_prime = V_prime0 * np.ones(lattice.X)
    V_prime[:widthF+widthX] = 0
    V_prime[widthF+widthX+widthP:] = 0
    V_prime[widthF+widthX] = V_prime0/2
    V_prime[widthF+widthX+widthP-1] = V_prime0/2

    H = Hamiltonian(t, mu, lattice, U=U, V_prime=V_prime, V=V,
                    F0_init=s_val,
                    F_init=[d_val, d_val, -d_val, -d_val],
                    Fuu_init=[0.01, -0.01, 0.01j, -0.01j],
                    Fdd_init=[0.01, -0.01, 0.01j, -0.01j],
                    hx=h)

    fixed_sites = (V_prime == 0)
    partial_sc(H, temperature=T,
            fixed_sites=fixed_sites,
            fixed_syms=fixed_syms,
            atol=atol, rtol=rtol, maxiter=maxiter)
    F, Fuu, Fdd = H.get_correlations()
    xfpfx = np.real(Fuu.px-Fdd.px)[int(widthX+widthF+widthP/2)-1]/2
    return xfpfx, np.real(Fuu.px)[int(widthX+widthF+widthP/2)-1]

def main():
    idx = int(sys.argv[1])
    V_prime0 = V_prime_arr[idx]

    # Get the bulk p-wave order parameter for the given V_prime0 #
    out = bdg_sc_full_k(
        t, muP, temperature=T,
        V=V_prime0, Nx=100, Ny=100,
        atol=atol, rtol=rtol,
        maxiter=maxiter,
        F_init=[0.01, -0.01, 0.01j, -0.01j]
    )

    bulk = np.real(out.F_px)
    d_val = 50 * np.real(out.F_px)
    s_val = 10 * np.real(out.F_px)
    print("V':", V_prime0, "p-wave", np.real(out.F_px), "s-wave:", s_val, "d-wave:", d_val)

    # Calculate the order paramters for SFPFS for a given V_prime0 #
    _, nfpfn = calculate_xfpfx(V_prime0, muS, widthX)
    print("V':", V_prime0, "nfpfn:", nfpfn)

    # Calculate the order paramters for SFPFS for a given V_prime0 #
    sfpfs, _ = calculate_xfpfx(V_prime0, muS, widthX, fixed_syms=["F0"],
                            s_val=s_val, d_val=0, U0=1, V0=0)
    print("V':", V_prime0, "sfpfs:", sfpfs)

    # Calculate the order paramters for DFPFD for a given V_prime0 #
    dfpfd, _ = calculate_xfpfx(V_prime0, muD, widthX, 
                            fixed_syms=["F_xplus", "F_xmin", "F_yplus", "F_ymin"],
                            s_val=0, d_val=d_val, U0=0, V0=1)

    print("dfpfd:", dfpfd)
    
    os.makedirs("data/paper", exist_ok=True)
    np.savez(
        f"data/paper/temp_{idx:04d}.npz",
        idx=idx,
        V_prime=V_prime0,
        bulk=bulk,
        nfpfn= nfpfn,
        sfpfs=sfpfs,
        dfpfd=dfpfd,
    )

if __name__ == "__main__":
    main()
