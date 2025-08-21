from __future__ import annotations
import numpy as np
import pandas as pd

def build_lineup_priors(labels: pd.DataFrame, n_samples: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = labels[["game_id"]].copy()
    df["lineup_top4_ops_prior"] = 0.75 + rng.normal(0, 0.02, size=len(df))
    df["lineup_samples"] = n_samples
    return df
