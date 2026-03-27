import glob
import numpy as np

files = sorted(glob.glob("data/bulkS/temp_*.npz"))
print("Start merging.")
rows = []
for f in files:
    d = np.load(f)
    rows.append((
        int(d["idx"]),
        float(d["temp"]),
        d["F0"].item(),
    ))

rows.sort(key=lambda x: x[0])

temps = np.array([r[1] for r in rows])
F0_arr = np.array([r[2] for r in rows])

np.savez(
    "data/bulkS.npz",
    temps=temps,
    F0=F0_arr,
)
