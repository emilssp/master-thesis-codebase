import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.partial_sc import partial_sc


def main():
    idx = int(sys.argv[1])

    widthS = 10
    widthF = 1
    widthP = 30

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

    hz0 = 0.9 * t
    hz = np.zeros_like(mu)
    hz[widthS:widthS + widthF] = hz0

    U0 = 4 * t
    U = U0 * np.ones(lattice.X)
    U[widthS:] = 0
    U[widthS-1] = U0/2

    V0 = 2 * t
    V = V0 * np.ones(lattice.X)
    V[:widthF+widthS] = 0
    V[widthF+widthS-1] = V0/2

    V_prime = None

    temps = np.concatenate([
        np.arange(0.001, 0.009 + 1e-12, 0.001),
        np.arange(0.010, 0.100 + 1e-12, 0.001),
        np.arange(0.1005, 0.1400 + 1e-12, 0.0005),
        np.arange(0.141, 0.200 + 1e-12, 0.001),
    ])  # 240 temps total
    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, V=V,
        F0_init=0.305, hz=hz,
        F_init=[0.01, -0.01, 0.01j, -0.01j],
    )

    fixed_sites = U > 0
    fixed_syms = ["F0"]
    partial_sc(H, temperature=temp, 
               fixed_sites=fixed_sites, fixed_syms=fixed_syms, 
               atol=1e-6, rtol=1e-3, maxiter=10000, mixing=0.6)

    F0 = H.F0
    F, Fuu, Fdd = H.get_correlations()

    os.makedirs("data/SFP_partial", exist_ok=True)
    np.savez(
        f"data/SFP_partial/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthS/2)-1],
        F_swave=F.swave[int(widthS+widthF+widthP/2)-1],
        F_dwave=F.dwave[int(widthS+widthF+widthP/2)-1],
        F_px=F.px[int(widthS+widthF+widthP/2)-1],
        F_py=F.py[int(widthS+widthF+widthP/2)-1],
    )

    print("Temperature:", temp)
    print("F_px:", F.px[int(widthS+widthF+widthP/2)-1])
    print("F_py:", F.py[int(widthS+widthF+widthP/2)-1])

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
            "hz0": hz0,
            "U0": U0,
            "V0": V0,
            "V_prime": 0,
        }

        param_file = f"data/SFP_partial/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
