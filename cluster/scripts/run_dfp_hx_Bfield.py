import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    widthD = 10
    widthF = 1
    widthP = 10

    X, Y = widthD+widthF+widthP, 200
    lattice = Lattice(X, Y)

    t = 1

    muD = 0.0 * t
    muP = 1.8 * t
    muF = 1.4 * t
    mu = np.zeros(lattice.X)
    mu[:widthD] = muD
    mu[widthD:widthD + widthF] = muF
    mu[widthD + widthF:] = muP

    hx0 = 0.9 * t
    hx = np.zeros_like(mu)
    hx[widthD:widthD + widthF] = hx0

    hz0 = 1.0 * t
    hz = np.zeros_like(mu)
    hz[:widthD] = hz0

    U = None

    V0 = 2.0 * t
    V = V0 * np.ones(lattice.X)
    V[widthD:] = 0
    V[widthD-1] = V0/2

    V_prime0 = 1.5 * t
    V_prime = V_prime0 * np.ones(lattice.X)
    V_prime[:widthF+widthD] = 0
    V_prime[widthF+widthD] = V_prime0/2

    temps = np.concatenate([
        np.array([0.001]),  # add a very low temp for testing
        np.arange(0.01, 0.20 + 1e-12, 0.01),
        np.arange(0.21, 0.60 + 1e-12, 0.002),
    ])  # 217 temps total

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    H = Hamiltonian(t, mu, lattice, U=U, V_prime=V_prime, V=V,
                    F_init=[0.5, 0.5, -0.5, -0.5],
                    Fuu_init=[0.1, -0.1, 0.1j, -0.1j],
                    Fdd_init=[0.1, -0.1, 0.1j, -0.1j],
                    hx=hx, hz=hz)

    bdg_sc(H, atol=1e-6, rtol=1e-3, maxiter=10000, temperature=temp)

    F0 = H.F0
    F, Fuu, Fdd = H.get_correlations()

    os.makedirs("data/DFP_hx_Bfield", exist_ok=True)
    np.savez(
        f"data/DFP_hx_Bfield/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(widthD/2)+1],
        F_swave=F.swave[int(widthD/2)+1],
        F_dwave=F.dwave[int(widthD/2)+1],
        F_px=F.px[int(widthD/2)+1],
        F_py=F.py[int(widthD/2)+1],
        Fuu_px=Fuu.px[int(widthD+widthF+widthP/2)+1],
        Fuu_py=Fuu.py[int(widthD+widthF+widthP/2)+1],
        Fdd_px=Fdd.px[int(widthD+widthF+widthP/2)+1],
        Fdd_py=Fdd.py[int(widthD+widthF+widthP/2)+1],
    )

    # save parameters for inspection
    if idx == 1:
        params = {
            "widthS": widthD,
            "widthF": widthF,
            "widthP": widthP,
            "X": X,
            "Y": Y,
            "muS": muD,
            "muF": muF,
            "muP": muP,
            "hx0": hx0,
            "hz0": hz0,
            "U0": 0,
            "V0": V0,
            "V_prime": V0,
        }

        param_file = f"data/DFP_hx_Bfield/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
