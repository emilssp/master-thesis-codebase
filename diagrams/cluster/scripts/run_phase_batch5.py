import sys
import os
import numpy as np
import pandas as pd

from self_consistency_k import bdg_sc_full_k
from phase_utils import stable_config, same_stable_dict
from phase_utils import flatten_df


initial_seeds = np.array([
    [0, 0, 0, 0],  # normal state
    [0.01, -0.01, 0.01j, -0.01j],  # px + i*py
    [0.01, 0.01, 0.01, 0.01],  # s-wave
    [0.01+0.01, 0.01-0.01, 0.01, 0.01],  # s-wave + px
    [0.01, 0.01, 0.01+0.01, 0.01-0.01]  # s-wave + py
], dtype=np.complex128)

seed_strings = [
    "normal state", "px+i*py",
    "s-wave", "s-wave+px", "s-wave+py"
]

mu_arr = np.linspace(2.31, 4.0, 25)
T_arr = np.linspace(0.001, 0.2, 30)

free_tol = 0.1

t = 1
V = 2
Nx, Ny = 100, 100

atol = 1e-6
rtol = 1e-3
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
        for seed, seed_str in zip(initial_seeds, seed_strings):
            print(f"  Seed: {seed_str}")

            out = bdg_sc_full_k(
                t, mu, temperature=T,
                V=V, Nx=Nx, Ny=Ny,
                atol=atol, rtol=rtol,
                maxiter=maxiter,
                F_init=seed
            )
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

    # Create DataFrame
    df = pd.DataFrame(records)
    df_flat = flatten_df(df)

    file_idx = idx+500
    os.makedirs("data/muT_V2", exist_ok=True)
    df_flat.to_json(f"data/muT_V2/results_{file_idx:04d}.json")


if __name__ == "__main__":
    main()
