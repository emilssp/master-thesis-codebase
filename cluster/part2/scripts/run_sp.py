import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    widthS = 10
    widthP = 30
    X, Y = widthS+widthP, 200
    lattice = Lattice(X, Y)

    t = 1
    muS = 1.2 * t
    muP = 1.8 * t
    mu = np.zeros(lattice.X)
    mu[:widthS:] = muS
    mu[widthS:] = muP

    U0 = 3.25 * t
    U = U0 * np.ones(lattice.X)
    U[widthS:] = 0
    U[widthS-1] = U0/2

    V0 = 2 * t
    V = V0 * np.ones(lattice.X)
    V[:widthS] = 0
    V[widthS-1] = V0/2

    V_prime = None

    temps = np.concatenate([
        np.arange(0.001, 0.009 + 1e-12, 0.001),
        np.arange(0.01, 0.20 + 1e-12, 0.001),
    ])  # 200 temps total

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, V=V,
        F0_init=0.3,
        F_init=[0.01, -0.01, 0.01j, -0.01j],
    )

    # Adjust to your actual bdg_sc output
    bdg_sc(
        H, maxiter=10000, temperature=temp,
        rtol=1e-3, atol=1e-6
    )

    F, Fuu, Fdd = H.get_correlations()
    F0 = H.F0
    
    os.makedirs("data/SP", exist_ok=True)
    np.savez(
        f"data/SP/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthS/2)],
        F_swave=F.swave[int(widthS+widthP/2)-1],
        F_dwave=F.dwave[int(widthS+widthP/2)-1],
        F_px=F.px[int(widthS+widthP/2)-1],
        F_py=F.py[int(widthS+widthP/2)-1],
    )

    # save parameters for inspection
    if idx == 1:
        params = {
            "widthS": widthS,
            "widthP": widthP,
            "X": X,
            "Y": Y,
            "muS": muS,
            "muP": muP,
            "U0": U0,
            "V0": V0,
            "V_prime": 0,
        }

        param_file = f"data/SP/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
