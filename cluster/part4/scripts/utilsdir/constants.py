import numpy as np

PI = np.pi
s0 = np.eye(2, dtype=np.complex128)
s1 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
s2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
s3 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

s = np.stack([s1, s2, s3])

e_x = np.array([[1], [0], [0]])
e_y = np.array([[0], [1], [0]])
e_z = np.array([[0], [0], [1]])

spins = {
    "x+": +s1,
    "y+": +s2,
    "z+": +s3,
    "x-": -s1,
    "y-": -s2,
    "z-": -s3,
}
