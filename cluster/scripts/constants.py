import cupy as cp

s0 = cp.eye(2, dtype=cp.complex128)
s1 = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)
s2 = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
s3 = cp.array([[1, 0], [0, -1]], dtype=cp.complex128)

s = cp.stack([s1, s2, s3])

e_x = cp.array([[1], [0], [0]])
e_y = cp.array([[0], [1], [0]])
e_z = cp.array([[0], [0], [1]])

spins = {
    "x+": +s1,
    "y+": +s2,
    "z+": +s3,
    "x-": -s1,
    "y-": -s2,
    "z-": -s3,
}
