from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from odds import devig_two_way, american_to_decimal, load_odds_if_exists
from utils import read_parquet

def load_and_devig_odds(path: Path) -> pd.DataFrame:
    if not path.exists():
        df = pd.DataFrame([dict(date="2024-04-12", game_pk="2024-04-12_NYY_BOS", yrfi_price_american=110, nrfi_price_american=-120, book="SampleBook")])
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    py, pn = devig_two_way(
        american_to_decimal(df["yrfi_price_american"]),
        american_to_decimal(df["nrfi_price_american"])
    )
    df["book_prob_yrfi"] = py; df["book_prob_nrfi"] = pn
    return df

def simulate_bankroll(cfg: dict) -> dict:
    proc = Path(cfg["paths"]["processed_dir"])
    preds = read_parquet(proc / "train_predictions.parquet")
    odds = load_and_devig_odds(Path(cfg["paths"]["odds_dir"]) / "odds.csv")
    df = preds.merge(odds, on="date", how="left")
    df["edge_yrfi"] = df["prob_yrfi"] - df["book_prob_yrfi"]
    dec = american_to_decimal(df["yrfi_price_american"]); b = dec - 1
    p = df["prob_yrfi"]; q = 1 - p
    kelly = (b*p - q) / b
    df["stake"] = np.clip(kelly, 0, cfg["odds"]["kelly_fraction_cap"]) * cfg["backtest"]["bankroll_start"]
    y = df["yrfi"]; pnl = df["stake"] * (np.where(y==1, b, -1))
    df["bankroll"] = cfg["backtest"]["bankroll_start"] + pnl.cumsum().fillna(0)
    (Path(cfg["paths"]["reports_dir"]) / "profit_curve.csv").parent.mkdir(parents=True, exist_ok=True)
    df[["date","bankroll"]].to_csv(Path(cfg["paths"]["reports_dir"]) / "profit_curve.csv", index=False)
    final = float(df["bankroll"].iloc[-1]) if len(df) else cfg["backtest"]["bankroll_start"]
    return {"summary": {"final_bankroll": final}}
