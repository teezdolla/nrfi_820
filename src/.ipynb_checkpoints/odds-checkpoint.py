from __future__ import annotations
import numpy as np
import pandas as pd

def american_to_decimal(x):
    x = pd.Series(x, dtype="float64")
    pos = x > 0
    dec = pd.Series(index=x.index, dtype="float64")
    dec[pos]  = 1 + x[pos]/100.0
    dec[~pos] = 1 + 100.0/(-x[~pos])
    return dec

def devig_two_way(dec_y: pd.Series, dec_n: pd.Series):
    py = 1.0 / dec_y; pn = 1.0 / dec_n
    s = py + pn
    py_fair = py / s; pn_fair = pn / s
    return py_fair, pn_fair

def load_odds_if_exists(path: pd.Series):
    if Path(path).exists():
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame(columns=["date","game_pk","yrfi_price_american","nrfi_price_american","book"])

def choose_side_and_kelly(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if "yrfi_price_american" in df.columns and df["yrfi_price_american"].notna().any():
        dec_y = american_to_decimal(df["yrfi_price_american"])
        dec_n = american_to_decimal(df["nrfi_price_american"])
        py, pn = devig_two_way(dec_y, dec_n)
        df["book_prob_yrfi"] = py; df["book_prob_nrfi"] = pn
        df["edge_yrfi"] = df["prob_yrfi"] - df["book_prob_yrfi"]
        df["edge_nrfi"] = (1 - df["prob_yrfi"]) - df["book_prob_nrfi"]
        pick_yrfi = df["edge_yrfi"] >= df["edge_nrfi"]
        df["rec_side"] = np.where(pick_yrfi, "YRFI", "NRFI")
        dec_used = np.where(pick_yrfi, dec_y, dec_n)
        b = dec_used - 1.0
        p = np.where(pick_yrfi, df["prob_yrfi"], 1 - df["prob_yrfi"])
        q = 1 - p
        kelly = (b*p - q) / b
        df["kelly_fraction"] = np.clip(kelly, 0, cfg["odds"]["kelly_fraction_cap"])
        df["decimal_odds_used"] = dec_used
        thin = (df["edge_yrfi"].abs() < cfg["odds"]["min_edge_abs"])
        df["notes"] = np.where(thin, "edge<thresh", "")
    else:
        df["notes"] = "no_odds"
    return df
