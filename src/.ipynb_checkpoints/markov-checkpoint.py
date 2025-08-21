from __future__ import annotations
import numpy as np
import pandas as pd

def build_first_inning_prior(labels, team_feat, starter_feat, lineup_feat, parkweather_feat):
    # start from labels and keep both keys
    df = labels[["game_id","game_pk"]].copy()

    # one-to-one merges on the keys; will raise if a table still has dup rows
    df = df.merge(team_feat,    on=["game_id","game_pk"], how="left", validate="one_to_one")
    df = df.merge(starter_feat, on=["game_id","game_pk"], how="left", validate="one_to_one")
    df = df.merge(lineup_feat,  on=["game_id","game_pk"], how="left", validate="one_to_one")
    df = df.merge(parkweather_feat, on=["game_id","game_pk"], how="left", validate="one_to_one")

    # ... your mu computation stays the same ...
    mu = (
        0.25
        + 0.15 * df.get("team_att_runrate", 0)
        + 0.15 * df.get("starter_kbb_adj", 0)
        + 0.10 * df.get("lineup_top4_ops_prior", 0)
        + 0.05 * df.get("park_factor_runs", 0)
        # add whatever else you're already using
    )

    p_run = 1.0 - np.exp(-mu)
    # return aligned to labels index
    return pd.Series(p_run.values, index=labels.index, name="prior_y1")

