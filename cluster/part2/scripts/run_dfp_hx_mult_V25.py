import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.partial_sc import partial_sc

V_arr = np.array([
    1,2,3,4,5,
    5.5,6,6.5,7,
    7.5,8,8.5,9,9.5
])

dwave_arr = np.array([
    0.07912, 0.12194, 0.143596, 0.156518, 0.16489,
    0.2, 0.225, 0.25, 0.275,
    0.3, 0.325, 0.35, 0.375, 0.4
])

temps = np.concatenate([
    np.arange(0.001, 0.009 + 1e-12, 0.001),
    np.arange(0.01, 0.10 + 1e-12, 0.01),
    np.arange(0.110, 0.300 + 1e-12, 0.001),
])  # 210 temps total

# total jobs = 14 * 210 = 2940

def main():
    idx = int(sys.argv[1])
    idx_temp = idx % len(temps)
    idx_V = idx // len(temps)

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

    h0 = 0.9 * t
    h = np.zeros_like(mu)
    h[widthD:widthD + widthF] = h0

    U = None

    V0 = V_arr[idx_V]
    V = V0 * np.ones(lattice.X)
    V[widthD:] = 0
    V[widthD-1] = V0/2

    F_dwave = dwave_arr[idx_V]

    V_prime0 = 2.5 * t
    V_prime = V_prime0 * np.ones(lattice.X)
    V_prime[:widthF+widthD] = 0
    V_prime[widthF+widthD] = V_prime0/2

    temp = temps[idx_temp]
    print(f"Running job {idx} out of {len(temps)}")

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

    os.makedirs("data/DFP_hx_mult_V25", exist_ok=True)
    np.savez(
        f"data/DFP_hx_mult_V25/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
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

        param_file = f"data/DFP_hx_mult_V25/params_{idx:04d}.txt"
        with open(param_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
