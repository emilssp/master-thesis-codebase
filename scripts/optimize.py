import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import minimize


def free_energy(S1, S2, F0, mu, J, D):
    S1 = np.asarray(S1)
    S2 = np.asarray(S2)
    mu = np.asarray(mu)
    J = np.asarray(J)
    D = np.asarray(D)

    term_mu = np.dot(mu, S1 + S2)
    term_J = np.dot(J,  S1 * S2)
    term_D = np.dot(D,  np.cross(S1, S2))

    return F0 + term_mu + term_J + term_D


def penalized_objective(x, F0, mu, J, D, lam):
    S1 = x[:3]
    S2 = x[3:]
    F_val = free_energy(S1, S2, F0, mu, J, D)
    penalty = lam * (
        (np.dot(S1, S1) - 1.0)**2 +
        (np.dot(S2, S2) - 1.0)**2
    )
    return F_val + penalty


def free_energy_3(S1, S2, S3, F0, mu, J, D):
    S1 = np.asarray(S1)
    S2 = np.asarray(S2)
    S3 = np.asarray(S3)
    mu = np.asarray(mu)
    J = np.asarray(J)
    D = np.asarray(D)

    term_mu = np.dot(mu, S1 + S2 + S3)
    term_J = np.dot(J, S1 * S2 + S1 * S3 + S2 * S3)
    term_D = np.dot(D,
                    np.cross(S1, S2)
                    + np.cross(S1, S3)
                    + np.cross(S2, S3))
    return F0 + term_mu + term_J + term_D


def penalized_objective_3(x, F0, mu, J, D, lam):
    S1 = x[0:3]
    S2 = x[3:6]
    S3 = x[6:9]

    F_val = free_energy_3(S1, S2, S3, F0, mu, J, D)
    penalty = lam * (
        (np.dot(S1, S1) - 1.0)**2 +
        (np.dot(S2, S2) - 1.0)**2 +
        (np.dot(S3, S3) - 1.0)**2
    )
    return F_val + penalty


def xz_spin(theta_deg):
    theta = np.deg2rad(theta_deg)
    return np.array([np.cos(theta), 0.0, np.sin(theta)])


def generate_initial_guesses_xz(step_deg=45):
    angles = np.arange(0, 360 + step_deg, step_deg)
    guesses = []
    for theta1, theta2 in product(angles, repeat=2):
        S1_0 = xz_spin(theta1)
        S2_0 = xz_spin(theta2)
        x0 = np.concatenate([S1_0, S2_0])
        guesses.append(x0)
    return guesses


def generate_initial_guesses_xz_3(step_deg=45):
    angles = np.arange(0, 360 + step_deg, step_deg)
    guesses = []
    for theta1, theta2, theta3 in product(angles, repeat=3):
        S1_0 = xz_spin(theta1)
        S2_0 = xz_spin(theta2)
        S3_0 = xz_spin(theta3)
        x0 = np.concatenate([S1_0, S2_0, S3_0])
        guesses.append(x0)
    return guesses


def _normalize_or_fallback(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        r = np.random.randn(3)
        return r / np.linalg.norm(r)
    return v / n


def _spins_close_pair(S1a, S2a, S1b, S2b, spin_tol):
    d1 = np.linalg.norm(S1a - S1b)
    d2 = np.linalg.norm(S2a - S2b)
    return (d1 < spin_tol) and (d2 < spin_tol)


def _spins_close_triplet(S1a, S2a, S3a, S1b, S2b, S3b, spin_tol):
    d1 = np.linalg.norm(S1a - S1b)
    d2 = np.linalg.norm(S2a - S2b)
    d3 = np.linalg.norm(S3a - S3b)
    return (d1 < spin_tol) and (d2 < spin_tol) and (d3 < spin_tol)


def _extract_free_2(x_full, S1_fix, S2_fix):
    """Take full 6-vector and extract only free components into y."""
    parts = []
    if S1_fix is None:
        parts.append(x_full[0:3])
    if S2_fix is None:
        parts.append(x_full[3:6])
    if parts:
        return np.concatenate(parts)
    else:
        return np.array([])


def _reconstruct_full_2(y, S1_fix, S2_fix):
    # Rebuild full 6-vector from free variables y and fixed spins.
    full = np.zeros(6, dtype=float)
    i = 0
    if S1_fix is None:
        full[0:3] = y[i:i+3]
        i += 3
    else:
        full[0:3] = S1_fix
    if S2_fix is None:
        full[3:6] = y[i:i+3]
    else:
        full[3:6] = S2_fix
    return full


def _penalized_objective_2_partial(y, F0, mu, J, D, lam, S1_fix, S2_fix):
    # Objective over free subset; fixed spins are inserted as constants.
    x_full = _reconstruct_full_2(y, S1_fix, S2_fix)
    return penalized_objective(x_full, F0, mu, J, D, lam)


def _extract_free_3(x_full, S1_fix, S2_fix, S3_fix):
    parts = []
    if S1_fix is None:
        parts.append(x_full[0:3])
    if S2_fix is None:
        parts.append(x_full[3:6])
    if S3_fix is None:
        parts.append(x_full[6:9])
    if parts:
        return np.concatenate(parts)
    else:
        return np.array([])


def _reconstruct_full_3(y, S1_fix, S2_fix, S3_fix):
    full = np.zeros(9, dtype=float)
    i = 0
    if S1_fix is None:
        full[0:3] = y[i:i+3]
        i += 3
    else:
        full[0:3] = S1_fix
    if S2_fix is None:
        full[3:6] = y[i:i+3]
        i += 3
    else:
        full[3:6] = S2_fix
    if S3_fix is None:
        full[6:9] = y[i:i+3]
    else:
        full[6:9] = S3_fix
    return full


def _penalized_objective_3_partial(y, F0, mu, J, D, lam,
                                   S1_fix, S2_fix, S3_fix):
    x_full = _reconstruct_full_3(y, S1_fix, S2_fix, S3_fix)
    return penalized_objective_3(x_full, F0, mu, J, D, lam)


def find_ground_state(F0, mu, J, D,
                      lambdas=None,
                      step_deg=45,
                      optimizer_method="Powell",
                      energy_tol=1e-8,
                      spin_tol=1e-6,
                      verbose=False,
                      S1=None,
                      S2=None):
    mu = np.asarray(mu, dtype=float)
    J = np.asarray(J,  dtype=float)
    D = np.asarray(D,  dtype=float)

    # Normalize fixed spins (if provided)
    S1_fix = _normalize_or_fallback(S1) if S1 is not None else None
    S2_fix = _normalize_or_fallback(S2) if S2 is not None else None

    if lambdas is None:
        lambdas = [10.0**k for k in range(0, 8)]  # 1e2,...,1e6

    # Case: all spins fixed → no optimization needed
    if (S1_fix is not None) and (S2_fix is not None):
        S1_final = S1_fix
        S2_final = S2_fix
        F_val = free_energy(S1_final, S2_final, F0, mu, J, D)
        primary = {
            "S1": S1_final,
            "S2": S2_final,
            "F":  F_val,
            "x_opt": np.concatenate([S1_final, S2_final]),
            "success": True,
        }
        return {
            "S1": primary["S1"],
            "S2": primary["S2"],
            "F_min": primary["F"],
            "x_opt": primary["x_opt"],
            "success": primary["success"],
            "minima": [primary],
            "degenerate": False,
            "F_ref": F_val,
        }

    initial_guesses = generate_initial_guesses_xz(step_deg=step_deg)

    best_F = np.inf
    minima = []  # list of dicts with S1, S2, F, x, success

    for idx, x0_full in enumerate(initial_guesses):
        if verbose:
            print(f"[2-spin] Initial guess {idx+1}/{len(initial_guesses)}")

        x_curr_full = np.copy(x0_full)
        success = True

        for lam in lambdas:
            y0 = _extract_free_2(x_curr_full, S1_fix, S2_fix)
            res = minimize(
                _penalized_objective_2_partial,
                y0,
                args=(F0, mu, J, D, lam, S1_fix, S2_fix),
                method=optimizer_method,
            )
            if not res.success:
                success = False
            y_opt = res.x
            x_curr_full = _reconstruct_full_2(y_opt, S1_fix, S2_fix)

        S1_raw = x_curr_full[:3] if S1_fix is None else S1_fix
        S2_raw = x_curr_full[3:] if S2_fix is None else S2_fix

        S1_final = _normalize_or_fallback(S1_raw)
        S2_final = _normalize_or_fallback(S2_raw)

        F_val = free_energy(S1_final, S2_final, F0, mu, J, D)

        if F_val < best_F - energy_tol:
            best_F = F_val
            minima = [{
                "S1": S1_final,
                "S2": S2_final,
                "F":  F_val,
                "x_opt": np.concatenate([S1_final, S2_final]),
                "success": success,
            }]
        elif abs(F_val - best_F) <= energy_tol:
            is_new = True
            for m in minima:
                if _spins_close_pair(S1_final, S2_final,
                                     m["S1"], m["S2"], spin_tol):
                    is_new = False
                    break
            if is_new:
                minima.append({
                    "S1": S1_final,
                    "S2": S2_final,
                    "F":  F_val,
                    "x_opt": np.concatenate([S1_final, S2_final]),
                    "success": success,
                })

    primary = minima[0] if minima else {
        "S1": None, "S2": None, "F": np.inf, "x_opt": None, "success": False
    }
    return {
        "S1": primary["S1"],
        "S2": primary["S2"],
        "F_min": primary["F"],
        "x_opt": primary["x_opt"],
        "success": primary["success"],
        "minima": minima,
        "degenerate": len(minima) > 1,
        "F_ref": best_F,
    }


def spherical_spin(theta_deg, phi_deg):
    """
    Unit vector on S^2 given by spherical angles:
    - theta: polar angle from +z, in [0, pi]
    - phi:   azimuthal angle from +x towards +y, in [0, 2*pi)
    """
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    return np.array([
        np.sin(theta) * np.cos(phi),  # Sx
        np.sin(theta) * np.sin(phi),  # Sy
        np.cos(theta)                 # Sz
    ], dtype=float)


def generate_initial_guesses_3d_3(step_deg=45):
    """
    Generate initial guesses for 3 spins on the full sphere.

    For each spin, we sample:
    - theta in [0, 180] with spacing step_deg
    - phi   in [0, 360) with spacing step_deg

    Total guesses = (N_theta * N_phi)^3.
    """
    thetas = np.arange(0, 180 + step_deg, step_deg)  # include theta=0,180
    phis = np.arange(0, 360, step_deg)             # 0 <= phi < 360

    guesses = []
    # 6 angles: (θ1, φ1, θ2, φ2, θ3, φ3)
    for theta1, phi1, theta2, phi2, theta3, phi3 in product(
        thetas, phis, thetas, phis, thetas, phis
    ):
        S1_0 = spherical_spin(theta1, phi1)
        S2_0 = spherical_spin(theta2, phi2)
        S3_0 = spherical_spin(theta3, phi3)
        x0 = np.concatenate([S1_0, S2_0, S3_0])
        guesses.append(x0)

    return guesses


# ############ New 3D #####################################################S

def find_ground_state_3(F0, mu, J, D,
                        lambdas=None,
                        step_deg=30,
                        optimizer_method="Powell",
                        energy_tol=1e-8,
                        spin_tol=1e-6,
                        verbose=False,
                        S1=None,
                        S2=None,
                        S3=None,
                        all_axes=True):
    mu = np.asarray(mu, dtype=float)
    J = np.asarray(J,  dtype=float)
    D = np.asarray(D,  dtype=float)

    S1_fix = _normalize_or_fallback(S1) if S1 is not None else None
    S2_fix = _normalize_or_fallback(S2) if S2 is not None else None
    S3_fix = _normalize_or_fallback(S3) if S3 is not None else None

    if lambdas is None:
        lambdas = [10.0**k for k in range(2, 7)]

    if (S1_fix is not None) and (S2_fix is not None) and (S3_fix is not None):
        S1_final = S1_fix
        S2_final = S2_fix
        S3_final = S3_fix
        F_val = free_energy_3(S1_final, S2_final, S3_final, F0, mu, J, D)
        primary = {
            "S1": S1_final,
            "S2": S2_final,
            "S3": S3_final,
            "F":  F_val,
            "x_opt": np.concatenate([S1_final, S2_final, S3_final]),
            "success": True,
        }
        return {
            "S1": primary["S1"],
            "S2": primary["S2"],
            "S3": primary["S3"],
            "F_min": primary["F"],
            "x_opt": primary["x_opt"],
            "success": primary["success"],
            "minima": [primary],
            "degenerate": False,
            "F_ref": F_val,
        }

    if all_axes:
        initial_guesses = generate_initial_guesses_3d_3(step_deg=step_deg)
    else:
        initial_guesses = generate_initial_guesses_xz_3(step_deg=step_deg)

    best_F = np.inf
    minima = []

    for idx, x0_full in enumerate(initial_guesses):
        if verbose:
            print(f"[3-spin] Initial guess {idx+1}/{len(initial_guesses)}")

        x_curr_full = np.copy(x0_full)
        success = True

        for lam in lambdas:
            y0 = _extract_free_3(x_curr_full, S1_fix, S2_fix, S3_fix)
            res = minimize(
                _penalized_objective_3_partial,
                y0,
                args=(F0, mu, J, D, lam, S1_fix, S2_fix, S3_fix),
                method=optimizer_method,
            )
            if not res.success:
                success = False
            y_opt = res.x
            x_curr_full = _reconstruct_full_3(y_opt, S1_fix, S2_fix, S3_fix)

        S1_raw = x_curr_full[0:3] if S1_fix is None else S1_fix
        S2_raw = x_curr_full[3:6] if S2_fix is None else S2_fix
        S3_raw = x_curr_full[6:9] if S3_fix is None else S3_fix

        S1_final = _normalize_or_fallback(S1_raw)
        S2_final = _normalize_or_fallback(S2_raw)
        S3_final = _normalize_or_fallback(S3_raw)

        F_val = free_energy_3(S1_final, S2_final, S3_final, F0, mu, J, D)

        if F_val < best_F - energy_tol:
            best_F = F_val
            minima = [{
                "S1": S1_final,
                "S2": S2_final,
                "S3": S3_final,
                "F":  F_val,
                "x_opt": np.concatenate([S1_final, S2_final, S3_final]),
                "success": success,
            }]
        elif abs(F_val - best_F) <= energy_tol:
            is_new = True
            for m in minima:
                if _spins_close_triplet(S1_final, S2_final, S3_final,
                                        m["S1"], m["S2"], m["S3"],
                                        spin_tol):
                    is_new = False
                    break
            if is_new:
                minima.append({
                    "S1": S1_final,
                    "S2": S2_final,
                    "S3": S3_final,
                    "F":  F_val,
                    "x_opt": np.concatenate([S1_final, S2_final, S3_final]),
                    "success": success,
                })

    primary = minima[0] if minima else {
        "S1": None, "S2": None, "S3": None,
        "F": np.inf, "x_opt": None, "success": False
    }

    return {
        "S1": primary["S1"],
        "S2": primary["S2"],
        "S3": primary["S3"],
        "F_min": primary["F"],
        "x_opt": primary["x_opt"],
        "success": primary["success"],
        "minima": minima,               # all degenerate minima
        "degenerate": len(minima) > 1,  # flag
        "F_ref": best_F,
    }


def read_free_energy(df: pd.DataFrame, s1: int = None,
                     s2: int = None, s3: int = None):
    spins = {
        +1: "x+",
        +2: "y+",
        +3: "z+",
        -1: "x-",
        -2: "y-",
        -3: "z-",
    }

    if s1 is None and s2 is not None and s3 is not None:
        F = 0
        s2_ = spins[s2]
        s3_ = spins[s3]
        for s1 in [+1, +2, +3, -1, -2, -3]:
            s1_ = spins[s1]
            filtered_df = df[(df["spin1"] == s1_) &
                             (df["spin2"] == s2_) &
                             (df["spin3"] == s3_)]
            F += filtered_df.free_energy.iloc[0]/6
        return float(F)
    elif s2 is None and s1 is not None and s3 is not None:
        F = 0
        s1_ = spins[s1]
        s3_ = spins[s3]
        for s2 in [+1, +2, +3, -1, -2, -3]:
            s2_ = spins[s2]
            filtered_df = df[(df["spin1"] == s1_) &
                             (df["spin2"] == s2_) &
                             (df["spin3"] == s3_)]
            F += filtered_df.free_energy.iloc[0]/6
        return float(F)
    elif s3 is None and s1 is not None and s2 is not None:
        F = 0
        s1_ = spins[s1]
        s2_ = spins[s2]
        for s3 in [+1, +2, +3, -1, -2, -3]:
            s3_ = spins[s3]
            filtered_df = df[(df["spin1"] == s1_) &
                             (df["spin2"] == s2_) &
                             (df["spin3"] == s3_)]
            F += filtered_df.free_energy.iloc[0]/6
        return float(F)
    elif s1 is not None and s2 is not None and s3 is not None:
        s1_ = spins[s1]
        s2_ = spins[s2]
        s3_ = spins[s3]

        filtered_df = df[(df["spin1"] == s1_) &
                         (df["spin2"] == s2_) &
                         (df["spin3"] == s3_)]
        if not filtered_df.empty:
            return float(filtered_df.free_energy.iloc[0])

    return np.nan


def spin(theta):
    return np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=float)


def load_proc(filename):
    names = ["dist", "F0", "mu_x", "mu_y", "mu_z",
             "J12_x", "J12_y", "J12_z",
             "J13_x", "J13_y", "J13_z",
             "J23_x", "J23_y", "J23_z",
             "D12_x", "D12_y", "D12_z",
             "D13_x", "D13_y", "D13_z",
             "D23_x", "D23_y", "D23_z"]
    df = pd.read_csv(f"./data/processed/{filename}.csv",
                     skipinitialspace=True, names=names, header=0)
    df = df.sort_values(by=names)
    return df


def free_energy_dist(df, dist, s1, s2, s3):
    params = df[df["dist"] == dist]
    F0 = params["F0"].iloc[0]

    mu = params[["mu_x", "mu_y", "mu_z"]].to_numpy(dtype=float)
    J12 = params[["J12_x", "J12_y", "J12_z"]].to_numpy(dtype=float)
    J13 = params[["J13_x", "J13_y", "J13_z"]].to_numpy(dtype=float)
    J23 = params[["J23_x", "J23_y", "J23_z"]].to_numpy(dtype=float)
    D12 = params[["D12_x", "D12_y", "D12_z"]].to_numpy(dtype=float)
    D13 = params[["D13_x", "D13_y", "D13_z"]].to_numpy(dtype=float)
    D23 = params[["D23_x", "D23_y", "D23_z"]].to_numpy(dtype=float)

    F = F0
    F += np.dot(mu, s1)
    F += np.dot(mu, s2)
    F += np.dot(mu, s3)

    s1_times_s2 = s1 * s2
    s1_times_s3 = s1 * s3
    s2_times_s3 = s2 * s3

    F += np.dot(J12, s1_times_s2)
    F += np.dot(J13, s1_times_s3)
    F += np.dot(J23, s2_times_s3)

    s1_cross_s2 = np.cross(s1, s2)
    s1_cross_s3 = np.cross(s1, s3)
    s2_cross_s3 = np.cross(s2, s3)
    s2_cross_s1 = np.cross(s2, s1)
    s3_cross_s1 = np.cross(s3, s1)
    s3_cross_s2 = np.cross(s3, s2)

    F += np.dot(D12, s1_cross_s2)/2
    F += np.dot(D13, s1_cross_s3)/2
    F += np.dot(D23, s2_cross_s3)/2

    F += np.dot(D12, s2_cross_s1)/2
    F += np.dot(D13, s3_cross_s1)/2
    F += np.dot(D23, s3_cross_s2)/2

    return F[0]


def free_energy_theta(thetas, df, dist):
    theta1, theta2, theta3 = thetas
    s1 = spin(theta1)
    s2 = spin(theta2)
    s3 = spin(theta3)

    return free_energy_dist(df, dist, s1, s2, s3)


def spin_to_vec(s):
    spins = {
        +1: [1, 0, 0],
        +2: [0, 1, 0],
        +3: [0, 0, 1],
        -1: [-1, 0, 0],
        -2: [0, -1, 0],
        -3: [0, 0, -1],
    }
    return np.array(spins[s])


def find_error_dist(filename, dist):
    spins = [1, 2, 3, -1, -2, -3]
    err = 0
    for s1, s2, s3 in product(spins, repeat=3):
        names = ["dist", "spin1", "spin2", "spin3", "free_energy"]
        df_raw = pd.read_csv(f"./data/{filename}.csv",
                             skipinitialspace=True, names=names, header=0)
        df_raw = df_raw.sort_values(by=names)
        F_check = read_free_energy(df_raw[df_raw['dist'] == dist],
                                   s1, s2, s3)

        df = load_proc(f"{filename}_proc")
        F = free_energy_dist(df, dist,
                             spin_to_vec(s1),
                             spin_to_vec(s2),
                             spin_to_vec(s3))
        err += np.abs(F-F_check)/216
    return err
