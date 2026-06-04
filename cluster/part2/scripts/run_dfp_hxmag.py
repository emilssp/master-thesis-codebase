import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.partial_sc import partial_sc

temps = np.concatenate([
    np.arange(0.001, 0.009 + 1e-12, 0.001),
    np.arange(0.010, 0.080 + 1e-12, 0.01),
    np.arange(0.0805, 0.1100 + 1e-12, 0.0005),
    np.arange(0.115, 0.300 + 1e-12, 0.001),
])  # 263 temps total
hmag_arr = np.linspace(0.5, 2, 10)

def main():
    idx = int(sys.argv[1])
    idx_temp = idx % len(temps)
    idx_hmag = idx // len(temps)

    widthD = 10
    widthF = 1
    widthP = 30

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

    h0 = hmag_arr[idx_hmag]
    h = np.zeros_like(mu)
    h[widthD:widthD + widthF] = h0

    U = None

    V0 = 7
    V = V0 * np.ones(lattice.X)
    V[widthD:] = 0
    V[widthD-1] = V0/2


    V_prime0 = 2 * t
    V_prime = V_prime0 * np.ones(lattice.X)
    V_prime[:widthF+widthD] = 0
    V_prime[widthF+widthD] = V_prime0/2

    temp = temps[idx_temp]
    print(f"Running job {idx} out of {len(temps)}")

    F_dwave = 0.3

    H = Hamiltonian(t, mu, lattice, U=U, V_prime=V_prime, V=V,
                    F_init=[F_dwave, F_dwave, -F_dwave, -F_dwave],
                    Fuu_init=[0.01, -0.01, 0.01j, -0.01j],
                    Fdd_init=[-0.01, 0.01, -0.01j, 0.01j],
                    hx=h)

    fixed_sites = V > 0
    fixed_syms = ["F0", "F_xplus", "F_xmin", "F_yplus", "F_ymin"]
    partial_sc(H, temperature=temp, 
            fixed_sites=fixed_sites, fixed_syms=fixed_syms, 
            atol=1e-6, rtol=1e-3, maxiter=10000, mixing=0.6)

    F0 = H.F0
    F, Fuu, Fdd = H.get_correlations()

    os.makedirs("data/DFP_hxmag", exist_ok=True)
    np.savez(
        f"data/DFP_hxmag/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        free_energy=H.free_energy(temperature=temp),
        hx = h0,
        F0=F0[int(widthD/2)],
        F_swave=F.swave[int(widthD/2)],
        F_dwave=F.dwave[int(widthD/2)],
        F_px=F.px[int(widthD/2)],
        F_py=F.py[int(widthD/2)],
        Fuu_px=Fuu.px[int(widthD+widthF+widthP/2)-1],
        Fuu_py=Fuu.py[int(widthD+widthF+widthP/2)-1],
        Fdd_px=Fdd.px[int(widthD+widthF+widthP/2)-1],
        Fdd_py=Fdd.py[int(widthD+widthF+widthP/2)-1],
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
            "hx": h0,
            "U0": 0,
            "V0": V0,
            "V_prime": V0,
        }

        param_file = f"data/DFP_hxmag/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
