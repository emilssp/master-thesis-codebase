import glob
import numpy as np

files = sorted(glob.glob("data/SFP_hx_strict/temp_*.npz"))
print("Start merging.")
rows = []
for f in files:
    d = np.load(f)
    rows.append((
        int(d["idx"]),
        float(d["temp"]),
        bool(d["converged"]),
        d["F0"].item(),
        d["F_swave"].item(),
        d["F_dwave"].item(),
        d["F_px"].item(),
        d["F_py"].item(),
        d["Fuu_px"].item(),
        d["Fuu_py"].item(),
        d["Fdd_px"].item(),
        d["Fdd_py"].item(),
    ))

rows.sort(key=lambda x: x[0])

temps = np.array([r[1] for r in rows])
converged = np.array([r[2] for r in rows])
F0_arr = np.array([r[3] for r in rows])
F_swave_arr = np.array([r[4] for r in rows])
F_dwave_arr = np.array([r[5] for r in rows])
F_px_arr = np.array([r[6] for r in rows])
F_py_arr = np.array([r[7] for r in rows])
Fuu_px_arr = np.array([r[8] for r in rows])
Fuu_py_arr = np.array([r[9] for r in rows])
Fdd_px_arr = np.array([r[10] for r in rows])
Fdd_py_arr = np.array([r[11] for r in rows])

np.savez(
    "data/SFP_hx_strict.npz",
    temps=temps,
    converged=converged,
    F0=F0_arr,
    F_swave=F_swave_arr,
    F_dwave=F_dwave_arr,
    F_px=F_px_arr,
    F_py=F_py_arr,
    Fuu_px=Fuu_px_arr,
    Fuu_py=Fuu_py_arr,
    Fdd_px=Fdd_px_arr,
    Fdd_py=Fdd_py_arr,
)
