# flake8: noqa: E501

import cupy as cp
import numpy as np
from .constants import *
import matplotlib.pyplot as plt

class SquareLattice:
    def __init__(self, X, Y=None,
                 pbc=False, pbc_x=False, pbc_y=False):
        self.X = X
        Y = X if Y is None else Y
        self.Y = Y
        self.pbc = pbc
        if pbc:
            self.pbc_x=True
            self.pbc_y=True
        else:
            self.pbc_x=pbc_x
            self.pbc_y=pbc_y
        self.num_sites = X * Y
        self.edges = []

        edges_ij = []
        edges_ji = []
        for y in range(Y):
            for x in range(X):
                i = y * X + x  # flat index for (x,y)

                # right neighbor (x+1, y)
                if x < X - 1:
                    edges_ij.append((i, i + 1))
                elif pbc_x and X > 2:
                    edges_ij.append((i, y * X))
                if x > 0:
                    j = i - 1  # (x-1, y)
                    edges_ij.append((i, j))
                elif pbc_x and X > 2:
                    # wrap to (X-1, y)
                    j = y * X + (X - 1)
                    edges_ij.append((i, j))
                # down neighbor (x, y+1)
                if y < Y - 1:
                    edges_ij.append((i, (y + 1) * X + x))
                elif pbc_y and Y > 2:
                    edges_ij.append((i, x))
                if y > 0:
                    j = (y - 1) * X + x  # (x, y-1)
                    edges_ij.append((i, j))
                elif pbc_y and Y > 2:
                    # wrap to (x, Y-1)
                    j = (Y - 1) * X + x
                    edges_ij.append((i, j))
        self.edges = edges_ij + edges_ji

    def get_coords(self, i):
        y = i // self.X
        x = i % self.X
        return cp.asarray([x, y], dtype=int)

    def get_disp(self, i, j):
        ri = self.get_coords(i)  # [xi, yi]
        rj = self.get_coords(j)  # [xj, yj]
        dr = cp.subtract(rj, ri) # [dx, dy]

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
    
    def get_edge_and_bulk_indices(X, Y):
        edge_indices = []
        bulk_indices = []

        for y in range(Y):
            for x in range(X):
                i = y * X + x  # flattened index

                if x == 0 or x == X - 1 or y == 0 or y == Y - 1:
                    edge_indices.append(i)
                elif (X // 4 <= x < 3 * X // 4) and (Y // 4 <= y < 3 * Y // 4):
                    bulk_indices.append(i)

        return edge_indices, bulk_indices

    def plot(self, highlight=None, path=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        coords = {}

        for y in range(self.Y):
            for x in range(self.X):
                i = y * self.X + x
                coords[i] = (x, -y)

        for i, j in self.edges:
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


class TriangularLattice:
    def __init__(self, L, pbc=False):
        self.L = L               # number of rows
        self.pbc = False
        self.num_sites = L * (L + 1) // 2

        self.row_offset = [y * (y + 1) // 2 for y in range(L)]

        self._x_of_site = [None] * self.num_sites
        self._y_of_site = [None] * self.num_sites

        edges_ij = []

        for y in range(L):
            row_len = y + 1
            base_i = self.row_offset[y]

            for x in range(row_len):
                i = base_i + x
                self._x_of_site[i] = x
                self._y_of_site[i] = y
                nbrs = [
                    (x - 1, y),     # left
                    (x + 1, y),     # right
                    (x,     y - 1), # up
                    (x - 1, y - 1), # up-left
                    (x,     y + 1), # down
                    (x + 1, y + 1)  # down-right
                ]

                for nx, ny in nbrs:
                    if 0 <= ny < L and 0 <= nx <= ny:
                        j = self.row_offset[ny] + nx
                        if j != i:
                            edges_ij.append((i, j))

        self.edges = edges_ij

    def get_coords(self, i):
        x = self._x_of_site[i]
        y = self._y_of_site[i]
        return cp.asarray([x, y], dtype=int)
    
    def get_coords_cart(self, i):
        x = self._x_of_site[i]
        y = self._y_of_site[i]
        X = x - 0.5 * y
        Y = - (cp.sqrt(cp.array(3.0)) / 2.0) * y
        return cp.asarray([X, Y], dtype=float)
    
    def get_disp(self, i, j):
        xi = self._x_of_site[i]
        yi = self._y_of_site[i]
        xj = self._x_of_site[j]
        yj = self._y_of_site[j]

        dx_int = xj - xi
        dy_int = yj - yi

        dx = cp.asarray(dx_int, dtype=cp.float64)
        dy = cp.asarray(dy_int, dtype=cp.float64)

        sqrt3 = cp.sqrt(cp.array(3.0, dtype=cp.float64))

        dX = dx - 0.5 * dy
        dY = - (sqrt3 / 2.0) * dy

        return cp.array([dX, dY], dtype=cp.float64)

    def get_dist(self, i, j):
        dr = self.get_disp(i, j)
        return cp.linalg.norm(dr)

    def get_index(self, loc):
        x,y=loc
        xs = cp.asarray(self._x_of_site, dtype=int)
        ys = cp.asarray(self._y_of_site, dtype=int)

        x = int(x)
        y = int(y)

        mask = (xs == x) & (ys == y)

        idx = cp.nonzero(mask)[0]

        if idx.size == 0:
            raise ValueError(f"No site at ({x}, {y})")

        return int(idx.item())

    def plot(self,
            highlight=None,
            path=None,
            show_labels=True,
            show_all_labels=False,
            show_edges=True,
            node_size=10,
            highlight_size=60):
        """
        show_labels: if False, no labels at all (fastest).
        show_all_labels: if True, label every node (your original behaviour).
                        Ignored if show_labels=False.
        highlight: list of indices to emphasize.
        """

        fig, ax = plt.subplots(figsize=(6, 6))
        coords = {}

        for y in range(self.L):
            row_len = y + 1
            base_i = self.row_offset[y]
            for x in range(row_len):
                i = base_i + x
                X = x - 0.5 * y
                Y = - (np.sqrt(3) / 2.0) * y
                coords[i] = (X, Y)

        if show_edges:
            for i, j in self.edges:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                ax.plot([x1, x2], [y1, y2], 'k-', lw=0.5)

        highlight_set = set(highlight or [])

        xs_all = []
        ys_all = []
        xs_hi = []
        ys_hi = []
        for i, (xx, yy) in coords.items():
            if i in highlight_set:
                xs_hi.append(xx)
                ys_hi.append(yy)
            else:
                xs_all.append(xx)
                ys_all.append(yy)

        ax.scatter(xs_all, ys_all, s=node_size, color='blue', zorder=4)
        if xs_hi:
            ax.scatter(xs_hi, ys_hi, s=highlight_size, color='red', zorder=3)

        if show_labels:
            if show_all_labels:
                for i, (xx, yy) in coords.items():
                    if i in highlight_set:
                        bbox = dict(facecolor="red", edgecolor="none",
                                    boxstyle="circle,pad=0.25")
                        color = "black"
                    else:
                        bbox = dict(facecolor="black", edgecolor="none",
                                    boxstyle="circle,pad=0.25")
                        color = "white"
                    ax.text(xx, yy, str(i), fontsize=6,
                            ha='center', va='center',
                            color=color, bbox=bbox)
            else:
                for i in highlight_set:
                    xx, yy = coords[i]
                    ax.text(xx, yy, str(i), fontsize=8,
                            ha='center', va='center',
                            color="black",
                            bbox=dict(facecolor="red",
                                    edgecolor="none",
                                    boxstyle="circle,pad=0.25"))

        ax.set_aspect("equal")
        ax.axis("off")
        if path is not None:
            plt.savefig(path, bbox_inches='tight', dpi=150)
        return fig, ax

class Lattice:
    """
    Unified lattice wrapper.
    Lattice(L, ...) -> Triangular cluster with side length L, no pbc 
    Lattice(X, Y, pbc=False, ...) -> Square/rectangular lattice X times Y
    Any additional kwargs are passed to the underlying lattice classes.
    """

    def __init__(self, *shape, pbc=False, **kwargs):

        if len(shape) == 1:
            # Triangular cluster of side length L
            L = shape[0]
            self.kind = "triangular"
            self.backend = TriangularLattice(L, **kwargs)

        elif len(shape) == 2:
            # Square/rectangular lattice X Y
            X, Y = shape
            self.kind = "square"
            self.backend = SquareLattice(X, Y, pbc=pbc, **kwargs)

        else:
            raise ValueError(
                f"Lattice expects 1 or 2 shape arguments, got {len(shape)}: {shape}"
            )

        self.pbc = pbc
        self.num_sites = self.backend.num_sites
        self.edges = self.backend.edges

    def __getattr__(self, name):
        return getattr(self.backend, name)


def delta(x, eta=1e-6): # Lorentzian approximation of delta function
    return eta / (cp.pi * (eta**2 + x**2))

def lorentzian(x, eta=1e-6):
    return eta / (cp.pi * (x**2 + eta**2))

def fermi_dirac(energy, T=1e-6):
    if T == 0:
        return cp.zeros_like(energy)
    else:
        return 1.0 / (cp.exp((energy) / T) + 1.0)


def unconventional_gap(disp, gap_p=cp.zeros(2), gap_p_uu=cp.zeros(2),
                       gap_p_dd=cp.zeros(2), gap_d=0, gap_s=0): 
    dz = cp.dot(disp, gap_p)
    dx = 0.5 * (cp.dot(disp, gap_p_uu) - cp.dot(disp, gap_p_dd))
    dy = -0.5j * (cp.dot(disp, gap_p_uu) + cp.dot(disp, gap_p_dd))

    gap_d = cp.array([gap_d,-gap_d])
    psi = gap_s + cp.dot(cp.abs(disp), gap_d)

    return cp.dot(1j*s2, (psi * s0 + dx * s1 + dy*s2 + dz*s3 )) 
    

class Hamiltonian:
    def __init__(self, lattice):
        self.lattice = lattice
        N = lattice.num_sites
        self.matrix = cp.zeros((4*N, 4*N),dtype=cp.complex128)
        self.gap = cp.zeros(N, dtype=cp.complex128)  # gap at each site
        self.gap_d = cp.zeros(N, dtype=cp.complex128)
        self.gap_s = cp.zeros(N, dtype=cp.complex128) 
        self.gap_px = cp.zeros(N, dtype=cp.complex128)
        self.gap_py = cp.zeros(N, dtype=cp.complex128)
        self.gap_px_uu = cp.zeros(N, dtype=cp.complex128)
        self.gap_py_uu = cp.zeros(N, dtype=cp.complex128)
        self.gap_px_dd = cp.zeros(N, dtype=cp.complex128)
        self.gap_py_dd = cp.zeros(N, dtype=cp.complex128)

    def set_block(self, i, j, block):
        self.matrix[4*i:4*(i+1), 4*j:4*(j+1)] = block

    def set_gap(self, i=None, gap=None):
        if gap is None:
            raise ValueError("Gap value must be provided.")
        if i is None:
            self.gap[:] = gap
        else:
            self.gap[i] = gap

    def diagonalize(self, drop_matrix=False):
        if self.matrix is None:
            raise RuntimeError("Hamiltonian matrix not built yet. "
                               "Call build() first.")
        eigenvalues, eigenvectors = cp.linalg.eigh(self.matrix)

        if drop_matrix:
            self.matrix = None
        return eigenvalues, eigenvectors

    def dos(self, energies, eta, idx=None, drop_matrix=False):
        evals, evecs = self.diagonalize(drop_matrix=drop_matrix)
        dos_values = cp.zeros_like(energies)

        N = self.lattice.num_sites
        M = evals.size  # should be 4*N
        for k in range(M):
            E = evals[k]
            if E < 0:
                continue  # skip negative ones

            col = evecs[:, k]

            if idx is None:
                # (4N,) -> (N,4)
                col_site = col.reshape(N, 4)
                u_up = col_site[:, 0]
                u_dn = col_site[:, 1]
                v_up = col_site[:, 2]
                v_dn = col_site[:, 3]

                w_pos = cp.sum(cp.abs(u_up)**2 + cp.abs(u_dn)**2)
                w_neg = cp.sum(cp.abs(v_up)**2 + cp.abs(v_dn)**2)
            else:
                base = 4 * idx
                u_up = col[base + 0]
                u_dn = col[base + 1]
                v_up = col[base + 2]
                v_dn = col[base + 3]

                w_pos = cp.sum(cp.abs(u_up)**2 + cp.abs(u_dn)**2)
                w_neg = cp.sum(cp.abs(v_up)**2 + cp.abs(v_dn)**2)

            dos_values += w_pos * lorentzian(energies - E, eta=eta)
            dos_values += w_neg * lorentzian(energies + E, eta=eta)

        del evecs, evals
        cp.get_default_memory_pool().free_all_blocks()

        return dos_values / energies.size

    def ldos(self, energies, eta, asnumpy=True):
        X = self.lattice.X
        Y = self.lattice.Y
        N = X * Y

        i = cp.arange(N)          # 0, 1, ..., N-1
        x = i % X                 # column (0 ... X-1)
        y = i // X                # row    (0 ... Y-1)

        edge_mask = ((x == 0) | (x == X - 1) | (y == 0) | (y == Y - 1) |
                    (x == 1) | (x == X - 2) | (y == 1) | (y == Y - 2) |
                    (x == 2) | (x == X - 3) | (y == 2) | (y == Y - 3) )
        edge_idx = i[edge_mask]
        bulk_idx = i[~edge_mask]
        if asnumpy:
            total_dos = self.dos(energies, eta=eta,drop_matrix=False)
            np_total_dos = cp.asnumpy(total_dos)
            del total_dos
            ldos_edge = self.dos(energies,eta=eta,idx=edge_idx,drop_matrix=False)
            np_ldos_edge = cp.asnumpy(ldos_edge)
            del ldos_edge
            ldos_bulk = self.dos(energies,eta=eta,idx=bulk_idx,drop_matrix=False)
            np_ldos_bulk = cp.asnumpy(ldos_bulk)
            del ldos_bulk
            return np_total_dos, np_ldos_edge, np_ldos_bulk
        else:
            total_dos = self.dos(energies, eta=eta,drop_matrix=False)
            ldos_edge = self.dos(energies,eta=eta,idx=edge_idx,drop_matrix=False)
            ldos_bulk = self.dos(energies,eta=eta,idx=bulk_idx,drop_matrix=False)
            return total_dos, ldos_edge, ldos_bulk

    def free_energy(self, temperature=0, drop_matrix=False):
        if self.matrix is None:
            raise RuntimeError("Hamiltonian matrix not built yet. "
                               "Call build() first.")
        eps = cp.linalg.eigvalsh(self.matrix)
        eps = eps[eps > 0]
        
        if drop_matrix:
            self.matrix = None

        U = -(1 / 2) * cp.sum(eps)
        if temperature == 0:
            S = 0
        elif temperature > 0:
            S = cp.sum(cp.log(1 + cp.exp(-eps / temperature)))

        F = U - temperature * S

        return F
    
    def ldos_per_spin(self, energies, eta, idx=None):
        evals, evecs = self.diagonalize(drop_matrix=False)
        dos_up_values = cp.zeros_like(energies)
        dos_dn_values = cp.zeros_like(energies)

        N = self.lattice.num_sites
        M = evals.size  # should be 4*N
        for k in range(M):
            E = evals[k]
            if E < 0:
                continue  # skip negative ones without creating a masked copy

            col = evecs[:, k]

            if idx is None:
                col_site = col.reshape(N, 4)
                u_up = col_site[:, 0]
                u_dn = col_site[:, 1]
                v_up = col_site[:, 2]
                v_dn = col_site[:, 3]

            else:
                base = 4 * idx
                u_up = col[base + 0]
                u_dn = col[base + 1]
                v_up = col[base + 2]
                v_dn = col[base + 3]

            dos_up_values += cp.sum(cp.abs(u_up)**2) * lorentzian(energies - E, eta=eta)
            dos_up_values += cp.sum(cp.abs(v_up)**2) * lorentzian(energies + E, eta=eta)

            dos_dn_values += cp.sum(cp.abs(u_dn)**2) * lorentzian(energies - E, eta=eta)
            dos_dn_values += cp.sum(cp.abs(v_dn)**2) * lorentzian(energies + E, eta=eta)
        return dos_up_values / energies.size, dos_dn_values / energies.size
    
    def get_corr(self, T):
        gap = self.gap
        eigval, eigvec = self.diagonalize(drop_matrix=False)
        eigvec = eigvec[:, eigval >= 0]
        eigval = eigval[eigval >= 0]
        eigvec = eigvec.T.reshape((eigval.size, -1, 4))
        corr = cp.zeros_like(gap)
        for n in range(eigval.size):
            En = eigval[n]
            u_n = eigvec[n, :, 0:2]  # u spinors
            v_n = eigvec[n, :, 2:4]  # v spinors
            f_En = fermi_dirac(En, T)

            corr += (u_n[:, 1] * cp.conj(v_n[:, 0]) * f_En +
                    u_n[:, 0] * cp.conj(v_n[:, 1]) * (1 - f_En))
        return corr

def rotate120(loc, X):
    L = X - 1
    x, y = int(loc[0]), int(loc[1])
    return cp.array([y - x, L - x])

def find_equilateral_triplet(X, pad, dist):
    L = X - 1

    if dist < 1 or dist > L - 3*pad:
        raise ValueError(f"No 120°-symmetric triple with dist={dist} "
                         f"for X={X}, pad={pad} (max dist is {L - 3*pad}).")

    for i in range(pad, L - 2*pad + 1):
        # j >= pad, k = L - i - j >= pad  ->  j <= L - i - pad
        for j in range(pad, L - i - pad + 1):
            k = L - i - j
            if k < pad:
                continue
            vals = sorted((i, j, k))
            if vals[-1] - vals[0] == dist:
                # convert (i,j,k) -> (x,y) using x=i, j=y-x
                x1, y1 = i, i + j
                loc1 = cp.array([x1, y1])
                loc2 = rotate120(loc1, X)
                loc3 = rotate120(loc2, X)
                return loc1, loc2, loc3

    raise ValueError(f"No configuration found for dist={dist}.")