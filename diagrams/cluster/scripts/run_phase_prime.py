import sys
import os
import numpy as np
import pandas as pd

from self_consistency_k import bdg_sc_full_k
from phase_utils import stable_config, same_stable_dict
from phase_utils import flatten_df


initial_seeds_uu = np.array([
    [0, 0, 0, 0],

    [0.1, -0.1, 0.1, -0.1],
    [0.1, -0.1, 0.1j, -0.1j],

    [0, 0, 0, 0],
    [0, 0, 0, 0],

    [0.1, -0.1, 0.1, -0.1],
    [0.1, -0.1, 0.1j, -0.1j]
], dtype=np.complex128)

initial_seeds_dd = np.array([
    [0, 0, 0, 0],

    [0, 0, 0, 0],
    [0, 0, 0, 0],

    [0.1, -0.1, 0.1, -0.1],
    [0.1, -0.1, 0.1j, -0.1j],

    [0.1, -0.1, 0.1, -0.1],
    [0.1, -0.1, 0.1j, -0.1j],
], dtype=np.complex128)

seed_strings = [
    "normal state",
    "uu:px+py", "uu:px+ipy",
    "dd:px+py", "dd:px+ipy",
    "uu+dd:px+py", "uu+dd:px+ipy",
]

mu_arr = np.linspace(-4.0, 4.0, 48)
T_arr = np.linspace(0.001, 0.1, 25)

free_tol = 0.005

t = 1
V = 3.0
Nx, Ny = 100, 100

atol = 1e-8
rtol = 1e-5
maxiter = 3000

rows = []


def main():
    idx = int(sys.argv[1])

    print(f"Running job {idx+1} out of {len(mu_arr)}")

    records = []

    mu = mu_arr[idx]

    for T in T_arr:
        best_free = np.inf
        configs = []
        print("====================================")
        print(f"mu={mu:.2f}, T={T:.4f}")
        print("====================================")
        for seed_uu, seed_dd, seed_str in zip(initial_seeds_uu,
                                              initial_seeds_dd, seed_strings):
            print(f"  Seed: {seed_str}")
            print(f"  Seed uu: {seed_uu}, Seed dd: {seed_dd}")
            out = bdg_sc_full_k(
                t, mu, temperature=T,
                V_prime=V, Nx=Nx, Ny=Ny,
                atol=atol, rtol=rtol,
                maxiter=maxiter,
                Fuu_init=seed_uu,
                Fdd_init=seed_dd
            )

            rows.append({
                "mu": mu,
                "temp": T,
                "seed": seed_str,
                "free_energy": out.free_energy,
                "Fuu_px": out.Fuu_px,
                "Fuu_py": out.Fuu_py,
                "Fdd_px": out.Fdd_px,
                "Fdd_py": out.Fdd_py,
            })

            print(f"F_onsite: {out.F_onsite}")
            print(f"F_swave: {out.F_swave}, F_dwave: {out.F_dwave}")
            print(f"F_px: {out.F_px}, F_py: {out.F_py}")
            print(f"Free_energy = {out.free_energy}")

            stable = stable_config(out, atol=atol)

            print("------------------------------------------")
            if (out.free_energy < best_free and
                    np.abs(out.free_energy - best_free) > free_tol):

                best_free = out.free_energy
                configs = [{
                    "stable": stable,
                    "free": out.free_energy,
                }]

            elif np.abs(out.free_energy - best_free) <= free_tol:
                if len(stable) != 0 and not any(
                    same_stable_dict(c["stable"], stable, atol=atol, rtol=rtol)
                    for c in configs
                ):
                    configs.append({
                        "stable": stable,
                        "free": out.free_energy,
                    })

            print(configs)
            print("------------------------------------------")

        records.append({
            "mu": mu,
            "T": T,
            "best_free": best_free,
            "configs": configs
        })

    # Create DataFrames and save results
    df_raw = pd.DataFrame(rows)
    df = pd.DataFrame(records)
    df_flat = flatten_df(df)

    file_idx = idx
    os.makedirs("data/muT_V15_prime", exist_ok=True)
    os.makedirs("data/muT_V15_prime/raw", exist_ok=True)

    df_flat.to_json(f"data/muT_V15_prime/results_{file_idx:04d}.json")
    df_raw.to_json(f"data/muT_V15_prime/raw/results_{file_idx:04d}.json")


if __name__ == "__main__":
    main()
