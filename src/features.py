from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
from utils import assert_columns

def _eb_shrink(rate: pd.Series, n: pd.Series, prior: float, strength: float) -> pd.Series:
    n = n.fillna(0).clip(lower=0)
    return ((rate * n) + (prior * strength)) / (n + strength)

def build_team_fi_features(labels: pd.DataFrame, eb_prior_strength: float = 50) -> pd.DataFrame:
    df = labels.copy()
    assert_columns(df, ["date","game_id","game_pk","away_team","home_team","game_datetime_utc","yrfi"], "team features input")

    long = []
    for side, teamcol, oppcol in [("away","away_team","home_team"), ("home","home_team","away_team")]:
        part = df[["date","game_datetime_utc","game_id","game_pk",teamcol,oppcol,"yrfi"]].copy()
        part = part.rename(columns={teamcol:"team", oppcol:"opponent"})
        part["is_batting_first"] = int(side == "away")
        long.append(part)
    hist = pd.concat(long, ignore_index=True).sort_values("game_datetime_utc")

    hist["one"] = 1
    hist["fi_runs_proxy"] = hist["yrfi"]
    hist["cum_games_team"]    = hist.groupby("team")["one"].cumsum().shift(1).fillna(0)
    hist["cum_yrfi_for_team"] = hist.groupby("team")["fi_runs_proxy"].cumsum().shift(1).fillna(0)
    hist["rate_team"]         = hist["cum_yrfi_for_team"] / hist["cum_games_team"].replace(0, np.nan)

    league_prior = hist["fi_runs_proxy"].mean()
    hist["rate_team_eb"] = _eb_shrink(hist["rate_team"], hist["cum_games_team"], league_prior, eb_prior_strength)

    out = hist[["game_id","game_pk","game_datetime_utc","rate_team_eb"]].copy()
    out["asof_time"] = pd.to_datetime(out["game_datetime_utc"]) - pd.Timedelta(seconds=1)
    out = out.rename(columns={"rate_team_eb":"team_fi_rate_sdt"})
    return out[["game_id","game_pk","asof_time","team_fi_rate_sdt"]]

def build_starter_fi_features(labels: pd.DataFrame, eb_prior_strength: float = 50) -> pd.DataFrame:
    df = labels.copy()
    long = []
    for side, teamcol, oppcol in [("away","away_team","home_team"), ("home","home_team","away_team")]:
        part = df[["date","game_datetime_utc","game_id","game_pk",teamcol,oppcol,"yrfi"]].copy()
        part = part.rename(columns={teamcol:"team", oppcol:"opponent"})
        part["opp_fi_run_proxy"] = part["yrfi"]
        long.append(part)
    hist = pd.concat(long, ignore_index=True).sort_values("game_datetime_utc")

    hist["one"] = 1
    hist["cum_g_opp"]    = hist.groupby("opponent")["one"].cumsum().shift(1).fillna(0)
    hist["cum_yrfi_opp"] = hist.groupby("opponent")["opp_fi_run_proxy"].cumsum().shift(1).fillna(0)
    hist["opp_rate"]     = hist["cum_yrfi_opp"] / hist["cum_g_opp"].replace(0, np.nan)

    league_prior = hist["opp_fi_run_proxy"].mean()
    hist["opp_rate_eb"] = _eb_shrink(hist["opp_rate"], hist["cum_g_opp"], league_prior, eb_prior_strength)

    out = hist[["game_id","game_pk","game_datetime_utc","opp_rate_eb"]].copy()
    out["asof_time"] = pd.to_datetime(out["game_datetime_utc"]) - pd.Timedelta(seconds=1)
    out = out.rename(columns={"opp_rate_eb":"starter_fi_allow_rate_sdt"})
    return out[["game_id","game_pk","asof_time","starter_fi_allow_rate_sdt"]]

def build_pitch_matchup_features(labels: pd.DataFrame) -> pd.DataFrame:
    df = labels.copy()
    out = df[["game_id","game_pk"]].copy()
    # placeholders; keep numeric
    out["pitch_fb_usage_x_whiff"] = 0.0
    out["pitch_breaking_usage_x_whiff"] = 0.0
    return out

def build_park_weather_features(labels: pd.DataFrame,
                                stadiums: pd.DataFrame,
                                park_factors: pd.DataFrame,
                                reference_dir,
                                default_local_time: str) -> pd.DataFrame:
    df = labels[["game_id"]].copy()
    if "game_pk" in labels.columns:
        df["game_pk"] = labels["game_pk"]

    # Park factors by home team if available, else neutral
    pf = park_factors.groupby("team_code", as_index=False)["park_factor_runs"].mean().rename(columns={"team_code":"home_team"})
    if "home_team" in labels.columns:
        tmp = labels[["game_id","home_team"]].copy()
        if "game_pk" in labels.columns: tmp["game_pk"] = labels["game_pk"]
        tmp = tmp.merge(pf, on="home_team", how="left")
        df = df.merge(tmp[["game_id"] + (["game_pk"] if "game_pk" in labels.columns else []) + ["park_factor_runs"]],
                      on="game_id", how="left")
    df["park_factor_runs"] = df["park_factor_runs"].fillna(1.00)

    # Weather optional; neutral if absent
    wx_path = Path(reference_dir) / "weather_sample.csv"
    if wx_path.exists():
        wx = pd.read_csv(wx_path, dtype={"game_id": str})
        df = df.merge(wx[["game_id","temp_c","rel_humidity","wind_kph","mslp_hpa"]], on="game_id", how="left")
    df["temp_c"] = df.get("temp_c", pd.Series(index=df.index)).fillna(20.0)
    df["rel_humidity"] = df.get("rel_humidity", pd.Series(index=df.index)).fillna(50.0)
    df["wind_kph"] = df.get("wind_kph", pd.Series(index=df.index)).fillna(8.0)
    df["mslp_hpa"] = df.get("mslp_hpa", pd.Series(index=df.index)).fillna(1015.0)

    df["air_density_proxy"] = (1013 / df["mslp_hpa"]) * (15 / df["temp_c"].clip(lower=1.0))
    cols = ["game_id","park_factor_runs","temp_c","rel_humidity","wind_kph","mslp_hpa","air_density_proxy"]
    if "game_pk" in df.columns:
        cols.insert(1, "game_pk")
    return df[cols]
