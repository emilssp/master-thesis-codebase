import numpy as np
from scipy.special import expit

from .constants import PI


def sl(i_site: int, comp: int) -> int:
    """Global index for site i_site (0..Nx-1) and component comp (0..3)."""
    return 4 * i_site + comp


def delta(x, eta=1e-6):  # Lorentzian approximation of delta function
    return eta / (PI * (eta**2 + x**2))


def lorentzian(x, eta=1e-6):
    return eta / (PI * (x**2 + eta**2))


def fermi_dirac(energy, T=1e-6):
    x = energy / T
    return expit(-x)


def is_hermitian(matrix, atol=1e-8, rtol=1e-6):
    is_hermitian = np.allclose(matrix, matrix.conj().T, rtol=rtol, atol=atol)
    return is_hermitian


def is_converged(corr, corr_new, atol, rtol, eps=1e-10):
    c1 = True
    c2 = True
    for new, old in zip(corr, corr_new):
        c1 = c1 and (np.linalg.norm(new - old) < atol)
        c2 = c2 and (np.linalg.norm((new - old)/(old + eps)) < rtol)

    return c1 and c2


def free_energy_nsc(H, U=0, V=0, V_prime=0, Delta0=0,
                    Delta_s=0,  Delta_d=0,  Delta_px=0, Delta_py=0,
                    Delta_uu_px=0, Delta_uu_py=0,
                    Delta_dd_px=0, Delta_dd_py=0):
    temperature = H.temperature
    eps = np.linalg.eigvalsh(H.matrix)
    eps = eps[eps > 0]
    E_S = 0
    E_S -= (np.abs(Delta0)**2)/U if not U == 0 else 0
    E_S -= (np.abs(Delta_s)**2)/V if not V == 0 else 0
    E_S -= (np.abs(Delta_d)**2)/V if not V == 0 else 0
    E_S -= (np.abs(Delta_px)**2)/V if not V == 0 else 0
    E_S -= (np.abs(Delta_py)**2)/V if not V == 0 else 0

    E_S -= (np.abs(Delta_dd_px)**2)/V_prime if not V_prime == 0 else 0
    E_S -= (np.abs(Delta_dd_py)**2)/V_prime if not V_prime == 0 else 0
    E_S -= (np.abs(Delta_uu_px)**2)/V_prime if not V_prime == 0 else 0
    E_S -= (np.abs(Delta_uu_py)**2)/V_prime if not V_prime == 0 else 0

    internal_energy = -(1 / 2) * np.sum(eps)
    if temperature == 0:
        S = 0
    elif temperature > 0:
        S = np.sum(np.log(1 + np.exp(-eps / temperature)))

    F = internal_energy - temperature * S + E_S

    return F
