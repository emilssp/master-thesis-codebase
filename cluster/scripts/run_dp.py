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
    muS = 0.0 * t
    muP = 1.8 * t
    mu = np.zeros(lattice.X)
    mu[:widthS:] = muS
    mu[widthS:] = muP

    V0 = 5 * t
    V = np.zeros(lattice.X)
    V[:widthS] = V0
    V[widthS-1] = V0/2

    V_prime0 = 1.5 * t
    V_prime = np.zeros(lattice.X)
    V_prime[widthS:] = V_prime0
    V_prime[widthS-1] = V_prime0/2

    U = None

    temps = np.concatenate([
        np.array([0.001]),  # add a very low temp for testing
        np.arange(0.01, 0.20 + 1e-12, 0.01),
        np.arange(0.21, 0.60 + 1e-12, 0.002),
    ])  # 217 temps total

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, g_s=V,
        F_init=[0.3, 0.3, -0.3, -0.3],
        Fuu_init=[0.01, -0.01, 0.01j, -0.01j],
        Fdd_init=[0.01, -0.01, 0.01j, -0.01j],
    )

    # Adjust to your actual bdg_sc output
    bdg_sc(H, maxiter=50000, temperature=temp, rtol=1e-3, atol=1e-6)

    F0 = H.F0
    F, Fuu, Fdd = H.get_correlations()

    os.makedirs("data/DP", exist_ok=True)
    np.savez(
        f"data/DP/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthS/2)],
        F_swave=F.swave[int(widthS/2)],
        F_dwave=F.dwave[int(widthS/2)],
        F_px=F.px[int(widthS/2)],
        F_py=F.py[int(widthS/2)],
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
            "V_prime": V_prime0,
        }

        param_file = f"data/DP/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
