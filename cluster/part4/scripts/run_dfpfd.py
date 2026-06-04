import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.partial_sc import partial_sc


temps = np.concatenate([
    np.arange(0.001, 0.009 + 1e-12, 0.001),
    np.arange(0.010, 0.110 + 1e-12, 0.001),
    np.arange(0.1105, 0.1500 + 1e-12, 0.0005),
    np.arange(0.141, 0.200 + 1e-12, 0.001),
])  # 250 temps total


def main():
    idx = int(sys.argv[1])

    widthS = 10
    widthF = 1
    widthP = 40

    X, Y = widthS+widthF+widthP+widthF+widthS, 200
    lattice = Lattice(X, Y)

    t = 1

    muS = 0.0 * t
    muP = 1.8 * t
    muF = 1.4 * t

    mu = np.zeros(lattice.X)
    mu[:widthS] = muS
    mu[widthS:widthS + widthF] = muF
    mu[widthS + widthF:widthS + widthF + widthP] = muP
    mu[widthS + widthF + widthP:widthS + widthF + widthP + widthF] = muF
    mu[widthS + widthF + widthP + widthF:widthS + widthF + widthP + widthF + widthS] = muS

    h0 = 0.9 * t
    h = np.zeros_like(mu)
    h[widthS:widthS + widthF] = h0
    h[widthS + widthF + widthP + widthF-1] = -h0

    U = None

    V0 = 4.0 * t
    V = V0 * np.ones(lattice.X)
    V[widthS:-widthS] = 0
    V[widthS-1] = V0/2
    V[-widthS] = V0/2

    V_prime0 = 1.5 * t
    V_prime = V_prime0 * np.ones(lattice.X)
    V_prime[:widthF+widthS] = 0
    V_prime[widthF+widthS+widthP:] = 0
    V_prime[widthF+widthS] = V_prime0/2
    V_prime[widthF+widthS+widthP-1] = V_prime0/2

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(t, mu, lattice, U=U, V_prime=V_prime, V=V,
                F_init=[0.23, 0.23, -0.23, -0.23],
                Fuu_init=[0.01, -0.01, 0.01j, -0.01j],
                Fdd_init=[0.01, -0.01, 0.01j, -0.01j],
                hx=h)

    fixed_sites = (V_prime == 0)
    fixed_syms = ["F_xplus", "F_xmin", "F_yplus", "F_ymin"]
    partial_sc(H, temperature=temp,
            fixed_sites=fixed_sites,
            fixed_syms=fixed_syms,
            atol=1e-6, rtol=1e-3, maxiter=10000, 
            verbose=False)

    F0 = H.F0
    F, Fuu, Fdd = H.get_correlations()

    os.makedirs("data/DFPFD", exist_ok=True)
    np.savez(
        f"data/DFPFD/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthS+widthF+widthP/2)],
        F_swave=F.swave[int(widthS+widthF+widthP/2)],
        F_dwave=F.dwave[int(widthS+widthF+widthP/2)],
        F_px=F.px[int(widthS+widthF+widthP/2)],
        F_py=F.py[int(widthS+widthF+widthP/2)],
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
            "U0": 0,
            "V0": V0,
            "V_prime": V_prime0,
        }

        param_file = f"data/DFPFD/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
