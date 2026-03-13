#!/usr/bin/env python3

from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from optimize import *

filename = "rkky_triangle_chiral_20251202_081203"
df = load_proc(f"{filename}_proc")

res3 = {}

def optimize_for_dist(dist, df):
    print(dist)
    params = df[df["dist"] == dist]
    F0 = params["F0"].iloc[0]
    mu = params[["mu_x", "mu_y", "mu_z"]].to_numpy(dtype=float)
    J = params[["J12_x",  "J12_y",  "J12_z"]].to_numpy(dtype=float)
    D = params[["D12_x",  "D12_y",  "D12_z"]].to_numpy(dtype=float)
    temp = find_ground_state_3(F0, mu, J, D,
                               step_deg=30,
                               energy_tol=1e-6,
                               spin_tol=1e-6,
                               verbose=False,
                               all_axes=True)
    return dist, temp


dists = list(range(1, 14))
res3 = {}

with ProcessPoolExecutor(max_workers=14) as executor:
    futures = {executor.submit(optimize_for_dist, dist, df):
               dist for dist in dists}

    # collect as they finish
    for future in as_completed(futures):
        dist, temp = future.result()
        print(f"Finished dist = {dist}")
        res3[dist] = temp

with open(f"data/{filename}_3d.pkl", "wb") as f:
    pickle.dump(res3, f)
