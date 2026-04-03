import numpy as np
import ast
import pandas as pd
import glob


def read_jsondir_to_df(file_path):
    """
    Read a directory of JSON files and convert them to a DataFrame.
    """
    files = sorted(glob.glob(f"{file_path}/results_*.json"))
    dfs = [pd.read_json(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    return df


def to_complex(x):
    """
    Convert an entry to a complex number.
    """
    if pd.isna(x):
        return 0.0 + 0.0j

    if isinstance(x, dict):
        return complex(float(x.get("real", 0.0)), float(x.get("imag", 0.0)))

    if isinstance(x, complex):
        return x

    if isinstance(x, (int, float, np.integer, np.floating)):
        return complex(float(x), 0.0)

    if isinstance(x, str):
        s = x.strip()
        try:
            y = ast.literal_eval(s)
            if isinstance(y, dict):
                return complex(
                    float(y.get("real", 0.0)),
                    float(y.get("imag", 0.0))
                )
            if isinstance(y, complex):
                return y
            if isinstance(y, (int, float)):
                return complex(float(y), 0.0)
        except Exception:
            pass

    raise ValueError(f"Cannot parse value {x!r} as complex.")


def canon_phase(vec, tol=1e-6):
    """
    Canonicalize a complex vector modulo global phase.
    """
    v = np.array(vec, dtype=np.complex128).copy()

    mags = np.abs(v)
    if np.all(mags < tol):
        return np.zeros_like(v)

    # use largest-magnitude component as anchor
    k = np.argmax(mags)
    phase = np.exp(-1j * np.angle(v[k]))
    v *= phase

    # enforce anchor positive real
    if v[k].real < 0:
        v *= -1

    # remove tiny noise
    v.real[np.abs(v.real) < tol] = 0.0
    v.imag[np.abs(v.imag) < tol] = 0.0

    return v


def rows_phase_equivalent(v1, v2, tol=1e-6):
    """
    Compare two already-canonicalized vectors.
    """
    return np.allclose(v1, v2, atol=tol, rtol=0.0)


def filter_df(df, tol=1e-6):
    """
    For each (mu, T), remove rows equivalent up to a global complex phase,
    while keeping rows that differ in magnitude or column structure.
    """
    df2 = df.copy()
    F_cols = [c for c in df2.columns if c.startswith("F_")]

    # complex vectors
    vecs = df2[F_cols].map(to_complex)

    # sorting keys: real parts, descending, left-to-right
    sort_aux = vecs.map(lambda z: z.real)
    sort_aux.columns = [f"{c}__real" for c in F_cols]

    df2 = pd.concat([df2, sort_aux], axis=1)

    sort_cols = ["mu", "T"] + list(sort_aux.columns)
    ascending = [True, True] + [False] * len(F_cols)

    df2 = df2.sort_values(
            sort_cols, ascending=ascending, kind="mergesort"
        ).reset_index(drop=True)

    kept_rows = []

    for (_, _), g in df2.groupby(["mu", "T"], sort=False):
        reps = []          # canonical representative vectors
        keep_idx = []

        for idx, row in g.iterrows():
            v = np.array(
                [to_complex(row[c]) for c in F_cols],
                dtype=np.complex128)
            v_can = canon_phase(v, tol=tol)

            equivalent = any(
                rows_phase_equivalent(v_can, rep, tol=tol)
                for rep in reps
            )

            if not equivalent:
                reps.append(v_can)
                keep_idx.append(idx)

        kept_rows.extend(keep_idx)

    out = (
        df2.loc[kept_rows]
        .drop(columns=list(sort_aux.columns))
        .reset_index(drop=True)
    )

    return out
