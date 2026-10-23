import numpy as np
from utilsdir.self_consistency_ft.self_consistency_k import bdg_sc_full_k


initial_seed = np.array([0.001, -0.001, 0.001j, -0.001j])
T_arr = np.concatenate([
    np.arange(0.00001, 0.0001, 0.00001),
    np.arange(0.0001, 0.003, 0.00005),
    np.arange(0.004, 0.10 + 1e-12, 0.001),
    np.arange(0.101, 0.140 + 1e-12, 0.001),
])  # 204 temps total

px_arr = np.zeros_like(T_arr)
py_arr = np.zeros_like(T_arr)

t = 1
V = 0.92
mu = 1.8
Nx, Ny = 100, 100

atol = 1e-6
rtol = 1e-3
maxiter = 5000


def main():
    for i, T in enumerate(T_arr):
        print("====================================")
        print(f"mu={mu:.2f}, T={T:.4f}")
        print("====================================")

        out = bdg_sc_full_k(
            t, mu, temperature=T,
            V=V, Nx=Nx, Ny=Ny,
            atol=atol, rtol=rtol,
            maxiter=maxiter,
            F_init=initial_seed
        )
        print(f"F_onsite: {out.F_onsite}")
        print(f"F_swave: {out.F_swave}, F_dwave: {out.F_dwave}")
        print(f"F_px: {out.F_px}, F_py: {out.F_py}")
        print(f"Free_energy = {out.free_energy}")

        print("------------------------------------------")
        px_arr[i] = out.F_px
        py_arr[i] = out.F_py

    np.savez(
    "data/bulk_092.npz",
    T_arr=T_arr,
    F_px=px_arr,
    F_py=py_arr
)

if __name__ == "__main__":
    main()
