from __future__ import annotations
import numpy as np
import pandas as pd

def build_first_inning_prior(labels: pd.DataFrame,
                             team_feat: pd.DataFrame,
                             starter_feat: pd.DataFrame,
                             lineup_feat: pd.DataFrame,
                             parkweather_feat: pd.DataFrame) -> pd.Series:
    df = labels[["game_id","game_pk"]].copy()
    x = (
        df.merge(team_feat,        on=["game_id","game_pk"], how="left")
          .merge(starter_feat,     on=["game_id","game_pk"], how="left")
          .merge(lineup_feat,      on=["game_id","game_pk"], how="left")
          .merge(parkweather_feat, on=["game_id","game_pk"], how="left")
    )
    mu = (
        0.8 * x["team_fi_rate_sdt"].fillna(0.25) +
        0.6 * x["starter_fi_allow_rate_sdt"].fillna(0.25) +
        0.3 * (x["lineup_top4_ops_prior"].fillna(0.75) - 0.7)
    )
    mu *= x["park_factor_runs"].fillna(1.0) * (1.0 + 0.02 * (1.0 - x["air_density_proxy"].fillna(1.0)))
    mu = mu.clip(0.05, 0.60)
    prior = 1.0 - np.exp(-mu)  # P(run(s)>0) under Poisson(mu)
    # align to labels order
    return pd.Series(prior.values, index=labels.index, name="prior_yrfi")
