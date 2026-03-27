import os
import sys
import numpy as np

from utils.Hamiltonian import Hamiltonian
from utils.Lattice import Lattice
from utils.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    widthS = 5
    widthF = 1
    widthP = 10

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

    U0 = 5.2 * t
    U = U0 * np.ones(lattice.X)
    U[widthS:] = 0
    U[widthS-1] = U0/2

    V0 = 1.5 * t
    V = V0 * np.ones(lattice.X)
    V[:widthF+widthS] = 0
    V[widthF+widthS-1] = V0/2

    V_prime = None

    temps = np.concatenate([
        np.arange(0.001, 0.009 + 1e-12, 0.001),
        np.arange(0.01, 0.20 + 1e-12, 0.01),
        np.arange(0.21, 0.31 + 1e-12, 0.005),
        np.arange(0.32, 0.41 + 1e-12, 0.001),
        np.arange(0.42, 0.60 + 1e-12, 0.002)
    ])
    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, V=V,
        F0_init=0.1, hz=hz,
        F_init=[0.1, -0.1, 0.1j, -0.1j],
    )

    # Adjust to your actual bdg_sc output
    F_swave, F_dwave, F_px, F_py = bdg_sc(
        H, maxiter=10000, temperature=temp,
        rtol=1e-3, atol=1e-6
    )
    F0 = H.F0
    os.makedirs("data/SFP", exist_ok=True)
    np.savez(
        f"data/SFP/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        F0=F0[int(widthS/2)-1],
        F_swave=F_swave[int(widthS+widthF+widthP/2)-1],
        F_dwave=F_dwave[int(widthS+widthF+widthP/2)-1],
        F_px=F_px[int(widthS+widthF+widthP/2)-1],
        F_py=F_py[int(widthS+widthF+widthP/2)-1],
    )

    print("Temperature:", temp)
    print("F_px:", F_px[int(widthS+widthF+widthP/2)-1])
    print("F_py:", F_py[int(widthS+widthF+widthP/2)-1])

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

        param_file = f"data/SFP/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
