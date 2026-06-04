import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    widthN = 10
    widthP = 40

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

    V0 = 2 * t
    V = V0 * np.ones(lattice.X)
    V[:widthN] = 0
    V[widthN-1] = V0/2

    H = Hamiltonian(t, mu, lattice,
                    U=U, V_prime=V_prime, V=V,
                    F_init=[0.01, -0.01, 0.01j, -0.01j])

    temps = np.concatenate([
        np.arange(0.001, 0.100 + 1e-12, 0.005),
        np.arange(0.102, 0.120+1e-12, 0.0001),
        np.arange(0.125, 0.150 + 1e-12, 0.005),
    ])  # 207 temps total

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    # Adjust to your actual bdg_sc output
    bdg_sc(H, maxiter=20000, temperature=temp, rtol=1e-3, atol=1e-6)
    F, Fuu, Fdd = H.get_correlations()
    F0 = H.F0

    os.makedirs("data/NP", exist_ok=True)
    np.savez(
        f"data/NP/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthN+widthP/2)-1],
        F_swave=F.swave[int(widthN+widthP/2)-1],
        F_dwave=F.dwave[int(widthN+widthP/2)-1],
        F_px=F.px[int(widthN+widthP/2)-1],
        F_py=F.py[int(widthN+widthP/2)-1],
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
