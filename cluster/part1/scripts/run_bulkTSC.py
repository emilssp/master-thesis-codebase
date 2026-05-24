import os
import sys
import numpy as np

from utilsdir.Hamiltonian import Hamiltonian
from utilsdir.Lattice import Lattice
from utilsdir.self_consistency import bdg_sc


def main():
    idx = int(sys.argv[1])

    X, Y = 40, 200
    t = 1.0
    lattice = Lattice(X, Y)

    mu = 1.8 * t * np.ones(lattice.X)

    temps = np.concatenate([
        np.arange(0.001, 0.100 + 1e-12, 0.005),
        np.arange(0.102, 0.120+1e-12, 0.0001),
        np.arange(0.125, 0.150 + 1e-12, 0.005),
    ])  # 207 temps total

    temp = temps[idx]

    print(f"Running job {idx} out of {len(temps)}")

    U = None
    V0 = 2 * t
    V_prime = V0 * np.ones(lattice.X)

    H = Hamiltonian(
        t, mu, lattice,
        U=U, V_prime=V_prime,
        Fuu_init=[0.01, -0.01, 0.01j, -0.01j],
        Fdd_init=[0.01, -0.01, 0.01j, -0.01j],
    )

    # Adjust to your actual bdg_sc output
    bdg_sc(H, maxiter=20000, temperature=temp, rtol=1e-3, atol=1e-6)
    F, Fuu, Fdd = H.get_correlations()
    F0 = H.F0

    os.makedirs("data/bulkTSC", exist_ok=True)
    np.savez(
        f"data/bulkTSC/temp_{idx:04d}.npz",
        idx=idx,
        temp=temp,
        converged=H.converged,
        F0=F0[int(X/2)-1],
        F_swave=F.swave[int(X/2)-1],
        F_dwave=F.dwave[int(X/2)-1],
        F_px=F.px[int(X/2)-1],
        F_py=F.py[int(X/2)-1],
        Fuu_px=Fuu.px[int(X/2)-1],
        Fuu_py=Fuu.py[int(X/2)-1],
        Fdd_px=Fdd.px[int(X/2)-1],
        Fdd_py=Fdd.py[int(X/2)-1],
    )


if __name__ == "__main__":
    main()
