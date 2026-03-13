import numpy as np
# import scipy.linalg as la
import matplotlib.pyplot as plt

from .utils import sl


class Lattice:
    def __init__(self, X, Y=None,
                 pbc_x=False, pbc_y=True):
        self.X = X
        Y = X if Y is None else Y
        self.Y = Y
        self.pbc_x = pbc_x
        self.pbc_y = pbc_y
        self.num_sites = X * Y

        edges_x = []
        edges_y = []
        for y in range(Y):
            for x in range(X):
                i = y * X + x  # flat index for (x,y)

                # right neighbor (x+1, y)
                if x < X - 1:
                    edges_x.append((i, i + 1))
                elif pbc_x and X > 2:
                    edges_x.append((i, y * X))

                # down neighbor (x, y+1)
                if y < Y - 1:
                    edges_y.append((i, (y + 1) * X + x))
                elif pbc_y and Y > 2:
                    edges_y.append((i, x))
        self.edges_y = edges_y
        self.edges_x = edges_x

    def get_coords(self, i):
        y = i // self.X
        x = i % self.X
        return np.asarray([x, y], dtype=int)

    def get_disp(self, i, j):
        ri = self.get_coords(i)  # [xi, yi]
        rj = self.get_coords(j)  # [xj, yj]
        dr = np.subtract(rj, ri)  # [dx, dy]

        Lx, Ly = self.X, self.Y
        if self.pbc_x:
            if dr[0] > Lx // 2:
                dr[0] -= Lx
            elif dr[0] < -Lx // 2:
                dr[0] += Lx

        if self.pbc_y:
            if dr[1] > Ly // 2:
                dr[1] -= Ly
            elif dr[1] < -Ly // 2:
                dr[1] += Ly

        return dr

    def get_dir(self, i, j):
        dr = self.get_disp(i, j)
        if dr[0] == 1 and dr[1] == 0:
            return "+x"
        elif dr[0] == -1 and dr[1] == 0:
            return "-x"
        elif dr[0] == 0 and dr[1] == 1:
            return "+y"
        elif dr[0] == 0 and dr[1] == -1:
            return "-y"

    def get_edge_and_bulk_indices(X, Y, edge_width=1):
        edge_indices = []
        bulk_indices = []

        for y in range(Y):
            for x in range(X):
                i = y * X + x  # flattened index

                if (x < edge_width or x >= X - edge_width or
                        y < edge_width or y >= Y - edge_width):
                    edge_indices.append(i)
                else:
                    bulk_indices.append(i)

        return edge_indices, bulk_indices

    def plot(self, highlight=None, path=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        coords = {}

        for y in range(self.Y):
            for x in range(self.X):
                i = y * self.X + x
                coords[i] = (x, -y)

        for i, j in self.edges_x + self.edges_y:
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            ax.plot([x1, x2], [y1, y2], 'k-', lw=1)

        for i, (xx, yy) in coords.items():
            ax.plot(xx, yy, 'ro')
            ax.text(xx, yy, str(i), fontsize=10, ha='center', va='center',
                    color="white", bbox=dict(facecolor="black",
                                             edgecolor="none",
                                             boxstyle="circle,pad=0.25"))
            if highlight is not None and i in highlight:
                ax.plot(xx, yy, 'ro')
                ax.text(xx, yy, str(i), fontsize=10, ha='center', va='center',
                        color="black", bbox=dict(facecolor="red",
                                                 edgecolor="none",
                                                 boxstyle="circle,pad=0.25"))

        ax.set_aspect("equal")
        ax.axis("off")
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        plt.show()


class Hamiltonian:
    def __init__(self, t, mu, lattice,
                 U=None, V=None, V_prime=None,
                 F0_init=0, F_init=np.zeros(4),
                 Fuu_init=np.zeros(4),
                 Fdd_init=np.zeros(4)):
        '''
        Sets the hamiltonian parameters and initial conditions
        '''
        self.t = t
        self.mu = mu
        self.lattice = lattice
        Nx = lattice.X
        self.dim = 4 * Nx

        if U is None:
            U = np.zeros(Nx)
        if V is None:
            V = np.zeros(Nx)
        if V_prime is None:
            V_prime = np.zeros(Nx)
        self.U = U
        self.V = V
        self.V_prime = V_prime

        self.F0 = np.zeros(Nx, dtype=np.complex128)
        self.F0[np.where(U != 0)] = F0_init
        self.F_xplus = np.zeros(Nx-1, dtype=np.complex128)
        self.F_xplus[np.where(V[:-1] != 0)] = F_init[0]
        self.F_xmin = np.zeros(Nx-1, dtype=np.complex128)
        self.F_xmin[np.where(V[1:] != 0)] = F_init[1]
        self.F_yplus = np.zeros(Nx, dtype=np.complex128)
        self.F_yplus[np.where(V != 0)] = F_init[2]
        self.F_ymin = np.zeros(Nx, dtype=np.complex128)
        self.F_ymin[np.where(V != 0)] = F_init[3]

        self.Fuu_xplus = np.zeros(Nx-1, dtype=np.complex128)
        self.Fuu_xplus[np.where(V_prime[:-1] != 0)] = Fuu_init[0]
        self.Fuu_xmin = np.zeros(Nx-1, dtype=np.complex128)
        self.Fuu_xmin[np.where(V_prime[1:] != 0)] = Fuu_init[1]
        self.Fuu_yplus = np.zeros(Nx, dtype=np.complex128)
        self.Fuu_yplus[np.where(V_prime != 0)] = Fuu_init[2]
        self.Fuu_ymin = np.zeros(Nx, dtype=np.complex128)
        self.Fuu_ymin[np.where(V_prime != 0)] = Fuu_init[3]

        self.Fdd_xplus = np.zeros(Nx-1, dtype=np.complex128)
        self.Fdd_xplus[np.where(V_prime[:-1] != 0)] = Fdd_init[0]
        self.Fdd_xmin = np.zeros(Nx-1, dtype=np.complex128)
        self.Fdd_xmin[np.where(V_prime[1:] != 0)] = Fdd_init[1]
        self.Fdd_yplus = np.zeros(Nx, dtype=np.complex128)
        self.Fdd_yplus[np.where(V_prime != 0)] = Fdd_init[2]
        self.Fdd_ymin = np.zeros(Nx, dtype=np.complex128)
        self.Fdd_ymin[np.where(V_prime != 0)] = Fdd_init[3]

        # self.set_kindep(gap0, gap1, gap2, gap1_uu, gap2_uu, gap1_dd, gap2_dd)

    def get_dim(self):
        return self.dim

    def set_kindep(self, gap0, gap1, gap2, gap1_uu, gap2_uu, gap1_dd, gap2_dd):

        self.matrix = np.zeros((self.dim, self.dim), dtype=np.complex128)
        t = self.t

        # set in diagonal elements (mu + 2cos(0))
        for i in range(self.lattice.X):
            diag = - self.mu[i]  # - 2.0 * t *cos(0)
            sl = slice(4 * i, 4 * i + 4)
            self.matrix[sl, sl] += np.diag(np.array(
                [diag, diag, -diag, -diag]
            ))
            # set in BCS gap
            block = np.zeros((4, 4), dtype=np.complex128)
            block[0, 3] = gap0[i]
            block[1, 2] = -gap0[i]
            block[2:, :2] = block[:2, 2:].conj().T
            self.matrix[sl, sl] += block

        # set in hopping block in x direction
        for i in range(self.lattice.X-1):
            sli = slice(4 * i, 4 * i + 4)
            slj = slice(4 * (i + 1), 4 * (i + 1) + 4)
            self.matrix[sli, slj] += np.diag(np.array(
                [-t, -t, t, t]
            ))
            self.matrix[slj, sli] += np.diag(np.array(
                [-t, -t, t, t]
            ))

            # # x+
            upper = np.zeros((4, 4), dtype=np.complex128)
            upper[0, 2] = gap2_uu[i]
            upper[0, 3] = gap2[i]
            upper[1, 2] = -gap1[i]
            upper[1, 3] = gap2_dd[i]

            # # x-
            lower = np.zeros((4, 4), dtype=np.complex128)
            lower[0, 2] = gap1_uu[i]
            lower[0, 3] = gap1[i]
            lower[1, 2] = -gap2[i]
            lower[1, 3] = gap1_dd[i]

            lower[2:, :2] = upper[:2, 2:].T.conj()
            upper[2:, :2] = lower[:2, 2:].T.conj()

            self.matrix[sli, slj] += upper
            self.matrix[slj, sli] += lower

    def build_H_k0(self):
        H = np.zeros((self.dim, self.dim), dtype=np.complex128)

        gap1 = self.V * (self.F_yplus + self.F_ymin)
        gap2 = -self.V * (self.F_yplus + self.F_ymin)
        # gap1_uu = self.V_prime[1:] * self.Fuu_xmin
        # gap2_uu = self.V_prime[:-1] * self.Fuu_xplus
        # gap1_dd = self.V_prime[1:] * self.Fuu_xmin
        # gap2_dd = self.V_prime[:-1] * self.Fuu_xplus

        for i in range(self.lattice.X):
            H[sl(i, 0), sl(i, 0)] = -2 * self.t
            H[sl(i, 1), sl(i, 1)] = -2 * self.t
            H[sl(i, 2), sl(i, 2)] = 2 * self.t
            H[sl(i, 3), sl(i, 3)] = 2 * self.t

            H[sl(i, 0), sl(i, 3)] = gap1[i]
            H[sl(i, 1), sl(i, 2)] = gap2[i]
            H[sl(i, 2), sl(i, 1)] = np.conj(gap2[i])
            H[sl(i, 3), sl(i, 0)] = np.conj(gap1[i])
        return H

    def build_H_kindep(self):
        gap0 = self.U * self.F0
        gap1 = self.V[1:] * self.F_xmin
        gap2 = self.V[:-1] * self.F_xplus

        H = np.zeros((self.dim, self.dim), dtype=np.complex128)

        for i in range(self.lattice.X):
            H[sl(i, 0), sl(i, 0)] = -self.mu[i]
            H[sl(i, 1), sl(i, 1)] = -self.mu[i]
            H[sl(i, 2), sl(i, 2)] = self.mu[i]
            H[sl(i, 3), sl(i, 3)] = self.mu[i]

            H[sl(i, 0), sl(i, 3)] = gap0[i]
            H[sl(i, 1), sl(i, 2)] = -gap0[i]
            H[sl(i, 2), sl(i, 1)] = np.conj(-gap0[i])
            H[sl(i, 3), sl(i, 0)] = np.conj(gap0[i])

        for i in range(self.lattice.X - 1):
            for c, hop in enumerate([-self.t, -self.t, self.t, self.t]):
                H[sl(i, c),   sl(i + 1, c)] = hop
                H[sl(i + 1, c), sl(i, c)] = hop

            # upper
            H[sl(i, 0), sl(i+1, 3)] = gap2[i]
            H[sl(i, 1), sl(i+1, 2)] = -gap1[i]
            H[sl(i, 2), sl(i+1, 1)] = -np.conj(gap2[i])
            H[sl(i, 3), sl(i+1, 0)] = np.conj(gap1[i])

            # lower
            H[sl(i+1, 0), sl(i, 3)] = gap1[i]
            H[sl(i+1, 1), sl(i, 2)] = -gap2[i]
            H[sl(i+1, 2), sl(i, 1)] = -np.conj(gap1[i])
            H[sl(i+1, 3), sl(i, 0)] = np.conj(gap2[i])

        return H

    def build_H_k(self, k):
        H2 = np.zeros((self.dim, self.dim), dtype=np.complex128)

        epsilon = -2 * self.t * np.cos(k)
        ep = np.exp(1j * k)
        em = np.exp(-1j * k)

        # y-pairing with phase
        gap1 = self.V * (self.F_yplus * em + self.F_ymin * ep)
        gap2 = -self.V * (self.F_yplus * ep + self.F_ymin * em)

        for i in range(self.lattice.X):
            # dispersion
            H2[sl(i, 0), sl(i, 0)] = epsilon
            H2[sl(i, 1), sl(i, 1)] = epsilon
            H2[sl(i, 2), sl(i, 2)] = -epsilon
            H2[sl(i, 3), sl(i, 3)] = -epsilon

            H2[sl(i, 0), sl(i, 3)] = gap1[i]
            H2[sl(i, 1), sl(i, 2)] = gap2[i]
            H2[sl(i, 2), sl(i, 1)] = np.conj(gap2[i])
            H2[sl(i, 3), sl(i, 0)] = np.conj(gap1[i])

        return H2

    def set_correlations(self, corr):
        self.F0 = corr[0].copy()

        self.F_xplus = corr[1].copy()
        self.F_xmin = corr[2].copy()
        self.F_yplus = corr[3].copy()
        self.F_ymin = corr[4].copy()

        # self.Fuu_xplus = corr[5].copy()
        # self.Fuu_xmin = corr[6].copy()
        # self.Fuu_yplus = corr[7].copy()
        # self.Fuu_ymin = corr[8].copy()

        # self.Fdd_xplus = corr[9].copy()
        # self.Fdd_xmin = corr[10].copy()
        # self.Fdd_yplus = corr[11].copy()
        # self.Fdd_ymin = corr[12].copy()
