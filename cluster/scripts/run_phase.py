import os
import sys
import numpy as np

from utils.hamiltonian import Lattice, Hamiltonian
from utils.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    X, Y = 30, 100
    t = 1.0
    lattice = Lattice(X, Y)

    mu = 1.8 * t * np.ones(lattice.X)

    temps = np.concatenate([
        [0.001],
        np.arange(0.01, 0.25 + 1e-12, 0.01),
        np.arange(0.26, 0.37 + 1e-12, 0.005),
        np.arange(0.38, 0.40 + 1e-12, 0.01),
    ])

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    U = np.zeros(lattice.X)
    V_prime = np.zeros(lattice.X)
    V0 = 1.5 * t
    V = V0 * np.ones(lattice.X)

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, V=V,
        F0_init=0.0,
        F_init=[0.1, -0.1, 0.1j, -0.1j],
    )

    # Adjust to your actual bdg_sc output
    F_swave, F_dwave, F_px, F_py = bdg_sc(
        H, maxiter=10000, temperature=temp,
        rtol=0.01, atol=1e-4
    )
    F0 = H.F0
    os.makedirs("data/temps", exist_ok=True)
    np.savez(
        f"data/temps/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        F0=np.mean(F0),
        F_swave=np.mean(F_swave),
        F_dwave=np.mean(F_dwave),
        F_px=np.mean(F_px),
        F_py=np.mean(F_py),
    )


if __name__ == "__main__":
    main()
