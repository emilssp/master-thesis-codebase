import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    widthS = 5
    widthP = 10
    X, Y = widthS+widthP, 200
    lattice = Lattice(X, Y)

    t = 1
    muS = 0.1 * t
    muP = 0.1 * t
    mu = np.zeros(lattice.X)
    mu[:widthS:] = muS
    mu[widthS:] = muP

    V0 = 1.5 * t

    V = V0 * np.ones(lattice.X)
    V[widthS:] = 0
    V[widthS-1] = V0/2

    V_prime = V0 * np.ones(lattice.X)
    V_prime[:widthS] = 0
    V_prime[widthS-1] = V0/2

    U = None

    temps = np.concatenate([
        np.arange(0.001, 0.009 + 1e-12, 0.001),
        np.arange(0.01, 0.30 + 1e-12, 0.01),
        np.arange(0.31, 0.41 + 1e-12, 0.001),
        np.arange(0.42, 0.60 + 1e-12, 0.002)
    ])  # 231 temps total

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, V=V,
        F_init=[0.1, 0.1, -0.1, -0.1],
        Fuu_init=[0.1, -0.1, 0.1j, -0.1j],
        Fdd_init=[0.1, -0.1, 0.1j, -0.1j],
    )

    # Adjust to your actual bdg_sc output
    F_swave, F_dwave, F_px, F_py = bdg_sc(
        H, maxiter=10000, temperature=temp,
        rtol=1e-3, atol=1e-6
    )
    F, Fuu, Fdd = H.get_correlations()
    F0 = H.F0
    os.makedirs("data/DP", exist_ok=True)
    np.savez(
        f"data/DP/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        F0=F0[int(widthS/2-1)],
        F_swave=F_swave[int(widthS/2-1)],
        F_dwave=F_dwave[int(widthS/2-1)],
        F_px=F_px[int(widthS/2-1)],
        F_py=F_py[int(widthS/2-1)],
        Fuu_px=Fuu.px[int(widthS+widthP/2)-1],
        Fuu_py=Fuu.py[int(widthS+widthP/2)-1],
        Fdd_px=Fdd.px[int(widthS+widthP/2)-1],
        Fdd_py=Fdd.py[int(widthS+widthP/2)-1],
    )

    # save parameters for inspection
    if idx == 1:
        params = {
            "widthN": widthS,
            "widthP": widthP,
            "X": X,
            "Y": Y,
            "muS": muS,
            "muP": muP,
            "U0": 0,
            "V0": V0,
            "V_prime": V0,
        }

        param_file = f"data/SP/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
