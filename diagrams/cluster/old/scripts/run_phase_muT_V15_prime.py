import sys
import os
import numpy as np
import pandas as pd

from self_consistency_k import bdg_sc_full_k
from phase_utils import stable_config, same_stable_dict
from phase_utils import flatten_df


initial_seeds = np.array([
    [0, 0, 0, 0],  # normal state
    [0.1, -0.1, 0, 0],  # px
    [0, 0, 0.1, -0.1],  # py
    [0.1, -0.1, 0.1, -0.1],  # px + py
    [0.1, -0.1, 0.1j, -0.1j],  # px + i*py
], dtype=np.complex128)

seed_strings = [
    "normal state",
    "px", "py", "px+py", "px+i*py",
]

mu_arr = np.linspace(0.01, 4.0, 50)
T_arr = np.linspace(0.001, 0.15, 50)

free_tol = 0.001

t = 1
V_prime = 1.5
Nx, Ny = 35, 35

atol = 1e-6
rtol = 1e-4
maxiter = 5000


def main():
    idx = int(sys.argv[1])
    print(f"Running job {idx} out of {len(mu_arr)}")

    records = []

    mu = mu_arr[idx]
    for T in T_arr:
        best_free = np.inf
        configs = []
        print("====================================")
        print(f"mu={mu:.2f}, T={T:.4f}")
        print("====================================")
        for seed1, seed_str1 in zip(initial_seeds, seed_strings):
            for seed2, seed_str2 in zip(initial_seeds, seed_strings):
                print(f"  Seed-up: {seed_str1}")
                print(f"  Seed-down: {seed_str2}")

                out = bdg_sc_full_k(
                    t, mu, Nx, Ny, V_prime=V_prime,
                    temperature=T, maxiter=maxiter,
                    atol=atol, rtol=rtol,
                    Fuu_init=seed1,
                    Fdd_init=seed2
                )
                print(f"Fuu_px: {out.Fuu_px}, Fuu_py: {out.Fuu_py}")
                print(f"Fdd_px: {out.Fdd_px}, Fdd_py: {out.Fdd_py}")
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
                        same_stable_dict(c["stable"], stable,
                                         atol=atol, rtol=rtol)
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

    # Create DataFrame
    df = pd.DataFrame(records)
    df_flat = flatten_df(df)

    os.makedirs("data/muT_V15_prime", exist_ok=True)
    df_flat.to_json(f"data/muT_V15_prime/results_{idx:04d}.json")


if __name__ == "__main__":
    main()
