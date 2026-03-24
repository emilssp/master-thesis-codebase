import os
import sys
import numpy as np

from utils.hamiltonian import Lattice, Hamiltonian
from utils.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    widthN = 5
    widthP = 10

    X, Y = widthN+widthP, 200
    lattice = Lattice(X, Y)
    t = 1
    muN = 1.2 * t
    muP = 1.8 * t
    mu = np.zeros(lattice.X)
    mu[:widthN] = muN
    mu[widthN:] = muP

    U = None
    V_prime = None

    V0 = 1.5 * t
    V = V0 * np.ones(lattice.X)
    V[:widthN] = 0
    V[widthN] = V0/2

    temps = np.concatenate([
        np.arange(0.001, 0.009 + 1e-12, 0.001),
        np.arange(0.01, 0.20 + 1e-12, 0.01),
        np.arange(0.21, 0.31 + 1e-12, 0.001),
        np.arange(0.32, 0.40 + 1e-12, 0.002)
    ])

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, V=V,
        F_init=[0.1, -0.1, 0.1j, -0.1j]
    )

    # Adjust to your actual bdg_sc output
    F_swave, F_dwave, F_px, F_py = bdg_sc(
        H, maxiter=10000, temperature=temp,
        rtol=0.01, atol=1e-6
    )
    F0 = H.F0
    os.makedirs("data/NP", exist_ok=True)
    np.savez(
        f"data/NP/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        F0=F0[int(widthN/2)],
        F_swave=F_swave[int(widthN+widthP/2)],
        F_dwave=F_dwave[int(widthN+widthP/2)],
        F_px=F_px[int(widthN+widthP/2)],
        F_py=F_py[int(widthN+widthP/2)],
    )

    # save parameters for inspection
    if idx == 1:
        params = {
            "widthN": widthN,
            "widthP": widthP,
            "X": X,
            "Y": Y,
            "muS": muN,
            "muP": muP,
            "U0": 0,
            "V0": V0,
            "V_prime": 0,
        }

        param_file = f"data/NP/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
