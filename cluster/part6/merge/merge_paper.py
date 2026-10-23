import glob
import numpy as np

files = sorted(glob.glob("data/paper/temp_*.npz"))
print("Start merging.")
rows = []
for f in files:
    print(f"Loading {f}")
    try:
        d = np.load(f)
        rows.append((
            int(d["idx"]),
            d["V_prime"],
            d["bulk"],
            d["nfpfn"],
            d["sfpfs"],
            d["dfpfd"]
        ))
    except Exception as e:
        print(f"Error loading {f}: {e}")

rows.sort(key=lambda x: x[0])
V_prime_arr = np.array([r[1] for r in rows])
bulk = np.array([r[2] for r in rows])
nfpfn_arr = np.array([r[3] for r in rows])
sfpfs_arr = np.array([r[4] for r in rows])
dfpfd_arr = np.array([r[5] for r in rows])

np.savez(
    "data/paper_data.npz",
    V_prime_arr=V_prime_arr,
    bulk_arr=bulk,
    nfpfn_arr=nfpfn_arr,
    sfpfs_arr=sfpfs_arr,
    dfpfd_arr=dfpfd_arr
)

