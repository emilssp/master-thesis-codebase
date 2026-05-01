import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.self_consistency import bdg_sc


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

    h0 = 0.9 * t
    h = np.zeros_like(mu)
    h[widthS:widthS + widthF] = h0

    V = None

    U0 = 5.2 * t
    U = U0 * np.ones(lattice.X)
    U[widthS:] = 0
    U[widthS-1] = U0/2

    V_prime0 = 1.5 * t
    V_prime = V_prime0 * np.ones(lattice.X)
    V_prime[:widthF+widthS] = 0
    V_prime[widthF+widthS] = V_prime0/2

    temps = np.concatenate([
        np.arange(0.001, 0.009 + 1e-12, 0.001),
        np.arange(0.01, 0.30 + 1e-12, 0.01),
        np.arange(0.31, 0.60 + 1e-12, 0.002)
    ])  # 185 temps total
    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(t, mu, lattice,
                    U=U, V_prime=V_prime, V=V,
                    F0_init=0.5,
                    Fuu_init=[0.1, -0.1, 0.1j, -0.1j],
                    Fdd_init=[0.1, -0.1, 0.1j, -0.1j],
                    hy=h)

    bdg_sc(H, atol=1e-6, rtol=1e-3, maxiter=10000, temperature=temp)

    F0 = H.F0
    F, Fuu, Fdd = H.get_correlations()

    os.makedirs("data/SFP_hy", exist_ok=True)
    np.savez(
        f"data/SFP_hy/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthS/2)+1],
        F_swave=F.swave[int(widthS/2)+1],
        F_dwave=F.dwave[int(widthS/2)+1],
        F_px=F.px[int(widthS/2)+1],
        F_py=F.py[int(widthS/2)+1],
        Fuu_px=Fuu.px[int(widthS+widthF+widthP/2)],
        Fuu_py=Fuu.py[int(widthS+widthF+widthP/2)],
        Fdd_px=Fdd.px[int(widthS+widthF+widthP/2)],
        Fdd_py=Fdd.py[int(widthS+widthF+widthP/2)],
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
            "hy": h0,
            "U0": U0,
            "V0": 0,
            "V_prime": V_prime0,
        }

        param_file = f"data/DFP_hy/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
