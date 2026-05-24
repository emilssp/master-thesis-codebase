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
    muS = 1.2 * t
    muP = 1.8 * t
    mu = np.zeros(lattice.X)
    mu[:widthS:] = muS
    mu[widthS:] = muP

    U0 = 5.2 * t
    U = U0 * np.ones(lattice.X)
    U[widthS:] = 0
    U[widthS-1] = U0/2

    V0 = 1.5 * t
    V = V0 * np.ones(lattice.X)
    V[:widthS] = 0
    V[widthS-1] = V0/2

    V_prime = None

    temps = np.concatenate([
        np.array([0.001]),  # add a very low temp for testing
        np.arange(0.01, 0.20 + 1e-12, 0.01),
        np.arange(0.21, 0.60 + 1e-12, 0.002),
    ])  # 217 temps total

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, g_t=V,
        F0_init=0.5,
        F_init=[0.1, -0.1, 0.1j, -0.1j],
    )

    # Adjust to your actual bdg_sc output
    F_swave, F_dwave, F_px, F_py = bdg_sc(
        H, maxiter=30000, temperature=temp,
        rtol=1e-3, atol=1e-6
    )
    F0 = H.F0
    os.makedirs("data/SP", exist_ok=True)
    np.savez(
        f"data/SP/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthS/2)],
        F_swave=F_swave[int(widthS+widthP/2)-1],
        F_dwave=F_dwave[int(widthS+widthP/2)-1],
        F_px=F_px[int(widthS+widthP/2)-1],
        F_py=F_py[int(widthS+widthP/2)-1],
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
