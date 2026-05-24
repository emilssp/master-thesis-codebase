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

    mu = 0.5 * t * np.ones(lattice.X)

    temps = np.concatenate([
        np.arange(0.001, 0.009 + 1e-12, 0.001),
        np.arange(0.01, 0.40 + 1e-12, 0.01),
        np.arange(0.41, 0.61 + 1e-12, 0.001),
        np.arange(0.62, 1.00 + 1e-12, 0.005)
    ])  # 327 temps total
    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    U = np.zeros(lattice.X)
    V_prime = np.zeros(lattice.X)
    V0 = 1.5 * t  # add factor to check Kuboki
    V = V0 * np.ones(lattice.X)

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime, V=V,
        F_init=[0.1, 0.1, -0.1, -0.1],
    )

    # Adjust to your actual bdg_sc output
    F_swave, F_dwave, F_px, F_py = bdg_sc(
        H, maxiter=10000, temperature=temp,
        rtol=1e-3, atol=1e-6
    )
    F0 = H.F0
    os.makedirs("data/bulkD", exist_ok=True)
    np.savez(
        f"data/bulkD/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        F0=F0[int(X/2-1)],
        F_swave=F_swave[int(X/2-1)],
        F_dwave=F_dwave[int(X/2-1)],
        F_px=F_px[int(X/2-1)],
        F_py=F_py[int(X/2-1)],
    )


if __name__ == "__main__":
    main()
