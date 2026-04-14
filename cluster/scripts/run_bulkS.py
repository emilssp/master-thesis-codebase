import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    X, Y = 10, 200
    t = 1.0
    lattice = Lattice(X, Y)

    mu = 1.8 * t * np.ones(lattice.X)

    temps = np.concatenate([
        np.arange(0.001, 0.009 + 1e-12, 0.001),
        np.arange(0.01, 0.20 + 1e-12, 0.01),
        np.arange(0.21, 0.31 + 1e-12, 0.001),
        np.arange(0.32, 0.40 + 1e-12, 0.002)
    ])  # 171 temps total

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    U0 = 5.2 * t / 2
    U = U0 * np.ones(lattice.X)

    V_prime = None
    V = None

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, V=V,
        F0_init=0.1
    )

    # Adjust to your actual bdg_sc output
    F_swave, F_dwave, F_px, F_py = bdg_sc(
        H, maxiter=10000, temperature=temp,
        rtol=1e-3, atol=1e-6
    )
    F0 = H.F0
    os.makedirs("data/bulkS", exist_ok=True)
    np.savez(
        f"data/bulkS/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        F0=F0[5]  # smiddle of the swave region
    )


if __name__ == "__main__":
    main()
