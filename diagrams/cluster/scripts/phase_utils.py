import numpy as np
import pandas as pd

from constants import corr_strings


def stable_config(out, atol=1e-6, rtol=0.1):
    stable = {}
    F_max = out.corr[np.argmax(np.abs(out.corr))]
    if np.abs(F_max) < atol:
        return stable
    for i, c in enumerate(out[1:-3]):
        if np.abs(c) > atol and np.abs(c) > rtol * np.abs(F_max):
            stable[corr_strings[i]] = c
    return stable


def same_stable_dict(d1, d2, atol=1e-6, rtol=1e-3):
    if d1.keys() != d2.keys():
        return False

    for k in d1:
        if not np.isclose(d1[k], d2[k], atol=atol, rtol=rtol):
            return False

    return True


def flatten_df(df):
    df_flat = df.explode("configs").reset_index(drop=True)

    # Split out the free energy and stable dict
    df_flat["free"] = df_flat["configs"].apply(
        lambda x: x.get("free") if isinstance(x, dict) else 0
    )
    df_flat["stable"] = df_flat["configs"].apply(
        lambda x: x.get("stable") if isinstance(x, dict) else {}
    )

    # Expand the stable dictionaries into columns
    stable_df = pd.DataFrame(df_flat["stable"].tolist())

    # Ensure all desired columns exist
    stable_df = stable_df.reindex(columns=corr_strings)

    # Combine everything
    df_flat = pd.concat(
        [
            df_flat.drop(columns=["configs", "stable"]),
            stable_df
        ],
        axis=1
    )
    df_flat.fillna(0, inplace=True)

    return df_flat
