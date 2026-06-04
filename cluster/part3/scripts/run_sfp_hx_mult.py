import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.partial_sc import partial_sc


U_arr = np.array([
    0,1,2,3,4,5,
    5.5,6,6.5,7,
    7.5,8,8.5
])

swave_arr = np.array([
    0, 0.07912, 0.12194, 0.143596,
    0.156518, 0.16489, 0.2, 0.225,
    0.25, 0.275, 0.3, 0.325, 0.35, 
])

temps = np.concatenate([
    np.arange(0.001, 0.009 + 1e-12, 0.001),
    np.arange(0.010, 0.080 + 1e-12, 0.01),
    np.arange(0.0805, 0.1100 + 1e-12, 0.0005),
    np.arange(0.115, 0.200 + 1e-12, 0.001),
])  # 163 temps total



def main():
    idx = int(sys.argv[1])
    idx_temp = idx % len(temps)
    idx_U = idx // len(temps)

    widthS = 10
    widthF = 1
    widthP = 40

    X, Y = widthS+widthF+widthP, 200
    lattice = Lattice(X, Y)

    t = 1

    muS = 1.2 * t
    muP = 1.8 * t
    muF = 1.4 * t
    mu = np.zeros(lattice.X)
    mu[:widthS] = muS
    mu[widthS:widthS + widthF] = muF
    mu[widthS + widthF:] = muP

    h0 = 0.9 * t
    h = np.zeros_like(mu)
    h[widthS:widthS + widthF] = h0

    V = None

    U0 = U_arr[idx_U]
    U = U0 * np.ones(lattice.X)
    U[widthS:] = 0
    U[widthS-1] = U0/2

    V_prime0 = 2 * t
    V_prime = V_prime0 * np.ones(lattice.X)
    V_prime[:widthF+widthS] = 0
    V_prime[widthF+widthS] = V_prime0/2

    temp = temps[idx_temp]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(t, mu, lattice,
                    U=U, V_prime=V_prime,
                    F0_init=swave_arr[idx_U],
                    Fuu_init=[0.01, -0.01, 0.01j, -0.01j],
                    Fdd_init=[0.01, -0.01, 0.01j, -0.01j],
                    hx=h)

    fixed_sites = U > 0
    fixed_syms = ["F0"]
    partial_sc(H, temperature=temp, 
               fixed_sites=fixed_sites, fixed_syms=fixed_syms, 
               atol=1e-6, rtol=1e-3, maxiter=20000, mixing=0.6)

    F0 = H.F0
    F, Fuu, Fdd = H.get_correlations()

    os.makedirs("data/SFP_hx_mult", exist_ok=True)
    np.savez(
        f"data/SFP_hx_mult/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthS/2)],
        F_swave=F.swave[int(widthS/2)],
        F_dwave=F.dwave[int(widthS/2)],
        F_px=F.px[int(widthS/2)],
        F_py=F.py[int(widthS/2)],
        Fuu_px=Fuu.px[int(widthS+widthF+widthP/2)-1],
        Fuu_py=Fuu.py[int(widthS+widthF+widthP/2)-1],
        Fdd_px=Fdd.px[int(widthS+widthF+widthP/2)-1],
        Fdd_py=Fdd.py[int(widthS+widthF+widthP/2)-1],
    )

    # save parameters for inspection
    if idx == 1:
        params = {
            "widthS": widthS,
            "widthF": widthF,
            "widthP": widthP,
            "X": X,
            "Y": Y,
            "muS": muS,
            "muF": muF,
            "muP": muP,
            "hx": h0,
            "U0": U0,
            "V0": 0,
            "V_prime": V_prime0,
        }

        param_file = f"data/SFP_hx_mult/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
