from __future__ import annotations
import os, warnings, requests
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from utils import (assert_non_empty, assert_columns, Timer)

@dataclass
class DataManager:
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    reference_dir: Path
    sample_dir: Path

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DataManager":
        root = Path(os.getenv("NRFI_PROJECT_ROOT", Path.cwd())).resolve()
        def _abs(p: str) -> Path:
            pth = Path(p)
            return pth if pth.is_absolute() else (root / pth)
        return cls(
            data_dir=_abs(cfg["paths"]["data_dir"]),
            raw_dir=_abs(cfg["paths"]["raw_dir"]),
            processed_dir=_abs(cfg["paths"]["processed_dir"]),
            reference_dir=_abs(cfg["paths"]["reference_dir"]),
            sample_dir=_abs(cfg["paths"]["sample_dir"]),
        )

    def ensure_dirs(self):
        for d in [self.data_dir, self.raw_dir, self.processed_dir, self.reference_dir, self.sample_dir]:
            d.mkdir(parents=True, exist_ok=True)

# -------------------
# Statcast / PBP
# -------------------

def _statcast_fetch(start_date: str, end_date: str) -> pd.DataFrame:
    from pybaseball import statcast
    with Timer(f"pybaseball.statcast {start_date}..{end_date}"):
        df = statcast(start_dt=start_date, end_dt=end_date)
    return df

def _sample_pbp() -> pd.DataFrame:
    rows = []
    games = [
        ("2024-04-12", "NYY", "BOS", "2024-04-12T23:05:00Z", "2024-04-12_NYY_BOS"),
        ("2024-04-14", "LAD", "SF",  "2024-04-15T02:10:00Z", "2024-04-14_LAD_SF"),
        ("2024-05-02", "CHC", "NYM", "2024-05-03T00:10:00Z", "2024-05-02_CHC_NYM"),
    ]
    for gd, away, home, gtime, gid in games:
        rows.append(dict(game_date=gd, game_pk=gid, game_time_utc=gtime, inning=1, inning_topbot="Top",
                         home_team=home, away_team=away, events="single", rbi=0))
        rows.append(dict(game_date=gd, game_pk=gid, game_time_utc=gtime, inning=1, inning_topbot="Top",
                         home_team=home, away_team=away, events="home_run", rbi=1))
        rows.append(dict(game_date=gd, game_pk=gid, game_time_utc=gtime, inning=1, inning_topbot="Bot",
                         home_team=home, away_team=away, events="strikeout", rbi=0))
        rows.append(dict(game_date=gd, game_pk=gid, game_time_utc=gtime, inning=2, inning_topbot="Top",
                         home_team=home, away_team=away, events="groundout", rbi=0))
    df = pd.DataFrame(rows)
    df["game_datetime_utc"] = pd.to_datetime(df["game_time_utc"], utc=True)
    return df

def fetch_statcast_range(start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
    try:
        df = _statcast_fetch(start_date, end_date)
        if "game_time_utc" not in df.columns:
            df["game_time_utc"] = pd.to_datetime(df["game_date"]) + pd.to_timedelta("00:00:00")
        df["game_datetime_utc"] = pd.to_datetime(df["game_time_utc"], utc=True, errors="coerce")
        assert_non_empty(df, "statcast_fetch")
        return df
    except Exception as e:
        warnings.warn(f"Statcast fetch failed ({e}). Using sample PBP.")
        return _sample_pbp()

DataManager.fetch_statcast_range = fetch_statcast_range

# -------------------
# Labels
# -------------------

def build_first_inning_labels(pbp: pd.DataFrame) -> pd.DataFrame:
    req = ["game_pk", "game_date", "inning", "inning_topbot", "home_team", "away_team"]
    assert_columns(pbp, req, "build_first_inning_labels")
    df = pbp.copy()
    df["game_datetime_utc"] = pd.to_datetime(df.get("game_datetime_utc", pd.NaT), utc=True, errors="coerce")
    df["game_time_utc"]     = pd.to_datetime(df.get("game_time_utc", df["game_datetime_utc"]), utc=True, errors="coerce")

    df1 = df[df["inning"] == 1].copy()
    if "rbi" in df1.columns:
        by_game = df1.groupby("game_pk", as_index=False).agg(
            first_inning_runs=("rbi", "sum"),
            game_date=("game_date", "first"),
            game_time_utc=("game_time_utc", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
    else:
        df1["runs_proxy"] = np.where(df1["events"].fillna("").str.contains("home_run"), 1, 0)
        by_game = df1.groupby("game_pk", as_index=False).agg(
            first_inning_runs=("runs_proxy", "sum"),
            game_date=("game_date", "first"),
            game_time_utc=("game_time_utc", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
    by_game["yrfi"] = (by_game["first_inning_runs"] > 0).astype(int)
    by_game["game_datetime_utc"] = pd.to_datetime(by_game["game_time_utc"], utc=True)
    by_game["date"] = pd.to_datetime(by_game["game_date"])
    by_game["game_id"] = by_game["date"].dt.strftime("%Y-%m-%d") + "_" + by_game["away_team"] + "_" + by_game["home_team"]
    return by_game[["date","game_id","game_pk","game_datetime_utc","away_team","home_team","yrfi"]]

    # --- Add to src/data.py ---
TEAM_ABBR_MAP = {
    # Normalize API / PBP discrepancies as needed
    "WSN": "WSH", "WAS": "WSH",
    "CWS": "CHW", "KCR": "KC", "TBR": "TB", "SFG": "SF", "LAA": "ANA",  # adjust to your PBP codes
    # add more iff your Statcast abbreviations differ from Stats API
}

def normalize_team_codes(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_ABBR_MAP)

def attach_scheduled_times(labels: pd.DataFrame,
                           stadiums: pd.DataFrame,
                           default_local_time: str = "19:00") -> pd.DataFrame:
    """
    Fills labels['game_datetime_utc'] with schedule times when available,
    else imputes using stadium timezone + default local first pitch.
    """
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"])
    # 1) Try MLB schedule by day (robust to offline fallback via fetch_schedule_probables)
    dates = sorted(labels["date"].dt.date.unique())
    sched_parts = []
    for d in dates:
        sd = fetch_schedule_probables(str(d))  # already returns UTC datetimes
        if not sd.empty:
            sd["away_team"] = normalize_team_codes(sd["away_team"])
            sd["home_team"] = normalize_team_codes(sd["home_team"])
            sd["game_id"] = pd.to_datetime(sd["date"]).dt.strftime("%Y-%m-%d") + "_" + sd["away_team"] + "_" + sd["home_team"]
            sched_parts.append(sd[["game_id","game_datetime_utc","game_pk"]])
    schedule = pd.concat(sched_parts, ignore_index=True) if sched_parts else pd.DataFrame(columns=["game_id","game_datetime_utc","game_pk"])

    # 2) Merge schedule first
    out = labels.merge(schedule[["game_id","game_datetime_utc"]], on="game_id", how="left", suffixes=("","_sched"))

    # 3) Impute where missing using stadium timezone + default_local_time
    need = out["game_datetime_utc"].isna()
    if need.any():
        st = stadiums[["team_code","timezone"]].rename(columns={"team_code":"home_team"})
        tmp = out.loc[need].merge(st, on="home_team", how="left")
        local_naive = pd.to_datetime(tmp["date"].dt.strftime("%Y-%m-%d") + " " + default_local_time)
        # localize by stadium tz then convert to UTC
        local = local_naive.dt.tz_localize(tmp["timezone"].fillna("UTC"), nonexistent="NaT", ambiguous="NaT")
        utc = local.dt.tz_convert("UTC")
        out.loc[need, "game_datetime_utc"] = utc.values

    return out

# -------------------
# Stadiums & park factors
# -------------------

def ensure_stadium_reference(reference_dir: Path) -> pd.DataFrame:
    reference_dir.mkdir(parents=True, exist_ok=True)
    f = reference_dir / "stadiums.csv"
    if f.exists():
        df = pd.read_csv(f)
    else:
        df = pd.read_csv(Path("data/reference/stadiums.csv"))
        df.to_csv(f, index=False)
    assert_non_empty(df, "stadiums.csv")
    assert_columns(df, ["team_code","lat","lon","timezone"], "stadiums.csv")
    return df

def compute_park_factors(pbp: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    try:
        if pbp is None or "home_team" not in pbp.columns or "rbi" not in pbp.columns:
            raise ValueError("insufficient pbp")
        df = pbp.copy()
        df["season"] = pd.to_datetime(df["game_date"]).dt.year
        df1 = df.groupby(["home_team","season"], as_index=False)["rbi"].sum().rename(columns={"home_team":"team_code","rbi":"home_runs"})
        g = df.groupby(["game_pk","home_team","season"], as_index=False).size().rename(columns={"home_team":"team_code","size":"pa"})
        games = g.groupby(["team_code","season"], as_index=False).size().rename(columns={"size":"games"})
        merged = df1.merge(games, on=["team_code","season"], how="left")
        merged["rpg"] = merged["home_runs"] / merged["games"].clip(lower=1)
        league = merged.groupby("season", as_index=False)["rpg"].mean().rename(columns={"rpg":"league_rpg"})
        out = merged.merge(league, on="season", how="left")
        out["park_factor_runs"] = (out["rpg"] / out["league_rpg"]).clip(0.8, 1.3)
        return out[["team_code","season","park_factor_runs"]]
    except Exception:
        return pd.read_csv(Path("data/reference/park_factors.csv"))

def ensure_park_factors(reference_dir: Path, pbp: Optional[pd.DataFrame], seasons: List[int]) -> pd.DataFrame:
    f = reference_dir / "park_factors.csv"
    if f.exists():
        return pd.read_csv(f)
    pf = compute_park_factors(pbp, seasons)
    pf.to_csv(f, index=False)
    return pf

# -------------------
# Weather
# -------------------

def fetch_weather_hourly(lat: float, lon: float, start_dt_utc: pd.Timestamp, end_dt_utc: pd.Timestamp) -> Optional[pd.DataFrame]:
    try:
        from meteostat import Hourly, Point
        p = Point(lat, lon)
        data = Hourly(p, start_dt_utc.tz_convert(None), end_dt_utc.tz_convert(None)).fetch()
        if data is None or data.empty:
            return None
        return data.reset_index().rename(columns={"time":"timestamp"})
    except Exception:
        return None

def fetch_schedule_probables(target_date: Optional[str] = None) -> pd.DataFrame:
    if target_date is None:
        target_date = pd.Timestamp.utcnow().date().isoformat()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}&hydrate=probablePitcher(note)"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        js = r.json()
        rows=[]
        for d in js.get("dates", []):
            for g in d.get("games", []):
                gid = g.get("gamePk")
                away = g["teams"]["away"]["team"]["abbreviation"]
                home = g["teams"]["home"]["team"]["abbreviation"]
                dt   = pd.to_datetime(g["gameDate"], utc=True)
                p_away = g["teams"]["away"].get("probablePitcher", {}).get("fullName", None)
                p_home = g["teams"]["home"].get("probablePitcher", {}).get("fullName", None)
                rows.append(dict(
                    date=pd.to_datetime(d["date"]),
                    game_pk=gid, game_datetime_utc=dt, away_team=away, home_team=home,
                    probable_away=p_away, probable_home=p_home
                ))
        df = pd.DataFrame(rows)
        if df.empty: raise ValueError("empty schedule")
        df["game_id"] = df["date"].dt.strftime("%Y-%m-%d") + "_" + df["away_team"] + "_" + df["home_team"]
        return df
    except Exception:
        sample = pd.DataFrame([dict(
            date=pd.to_datetime("2024-04-12"),
            game_pk="2024-04-12_NYY_BOS",
            game_datetime_utc=pd.to_datetime("2024-04-12T23:05:00Z"),
            away_team="NYY", home_team="BOS",
            probable_away="A Sample", probable_home="B Sample"
        )])
        sample["game_id"] = sample["date"].dt.strftime("%Y-%m-%d") + "_" + sample["away_team"] + "_" + sample["home_team"]
        return sample

def fetch_weather_for_games(labels: pd.DataFrame,
                            stadiums: pd.DataFrame,
                            default_local_time: str,
                            reference_dir: Path) -> pd.DataFrame:
    assert_columns(stadiums, ["team_code","lat","lon","timezone"], "stadiums ref")
    df = labels.copy()
    df["date_local"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.merge(stadiums[["team_code","lat","lon","timezone"]],
                  left_on="home_team", right_on="team_code", how="left")
    df["scheduled_local"] = pd.to_datetime(df["date_local"] + " " + default_local_time).astype("datetime64[ns]")
    df["scheduled_utc"] = df["scheduled_local"].dt.tz_localize(df["timezone"].fillna("UTC"),
                                                                nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
    out_rows=[]
    for _, r in df.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        start = r["scheduled_utc"] - pd.Timedelta(minutes=30)
        end   = r["scheduled_utc"] + pd.Timedelta(minutes=30)
        w = fetch_weather_hourly(lat, lon, start, end)
        if w is None or w.empty:
            fallback = pd.read_csv(reference_dir / "weather_sample.csv")
            m = fallback[fallback["game_id"] == r["game_id"]]
            if m.empty:
                out_rows.append(dict(game_id=r["game_id"], temp_c=20.0, rel_humidity=50.0, wind_kph=8.0, mslp_hpa=1015))
            else:
                out_rows.append(dict(game_id=r["game_id"],
                                     temp_c=float(m["temp_c"].iloc[0]),
                                     rel_humidity=float(m["rel_humidity"].iloc[0]),
                                     wind_kph=float(m["wind_kph"].iloc[0]),
                                     mslp_hpa=float(m["mslp_hpa"].iloc[0])))
        else:
            ww = w.iloc[0]
            out_rows.append(dict(game_id=r["game_id"],
                                 temp_c=ww.get("temp", np.nan),
                                 rel_humidity=ww.get("rhum", np.nan),
                                 wind_kph=ww.get("wspd", np.nan),
                                 mslp_hpa=ww.get("pres", np.nan)))
    wx = pd.DataFrame(out_rows)
    assert_non_empty(wx, "weather fetch")
    return wx

def write_sample_bundle(sample_dir: Path, labels: pd.DataFrame, pbp: pd.DataFrame,
                        stadiums: pd.DataFrame, weather: pd.DataFrame):
    sample_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(sample_dir / "labels.parquet", index=False)
    pbp.to_parquet(sample_dir / "pbp.parquet", index=False)
    stadiums.to_csv(sample_dir / "stadiums.csv", index=False)
    weather.to_csv(sample_dir / "weather.csv", index=False)


