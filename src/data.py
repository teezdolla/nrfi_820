from __future__ import annotations

import os, warnings, requests
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np

# repo utils
from utils import assert_non_empty, assert_columns, Timer


# ---------------------------------------------------------------------
# Team code normalization
# ---------------------------------------------------------------------
TEAM_ABBR_MAP = {
    # Canonical + common alternates
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CHW": "CHW", "CWS": "CHW",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET",
    "HOU": "HOU", "KC": "KC", "KCR": "KC",
    "LAA": "LAA", "ANA": "LAA",
    "LAD": "LAD", "MIA": "MIA", "FLA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY",
    "OAK": "OAK", "PHI": "PHI", "PIT": "PIT",
    "SD": "SD", "SDP": "SD",
    "SEA": "SEA", "SF": "SF", "SFG": "SF",
    "STL": "STL", "TB": "TB", "TBR": "TB",
    "TEX": "TEX", "TOR": "TOR",
    "WSH": "WSH", "WSN": "WSH", "WAS": "WSH",
}

def normalize_team_codes(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_ABBR_MAP)


# ---------------------------------------------------------------------
# Data manager
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Statcast / PBP
# ---------------------------------------------------------------------
def _statcast_fetch(start_date: str, end_date: str) -> pd.DataFrame:
    from pybaseball import statcast
    with Timer(f"pybaseball.statcast {start_date}..{end_date}"):
        df = statcast(start_dt=start_date, end_dt=end_date)
    return df

def _sample_pbp() -> pd.DataFrame:
    # tiny offline sample
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

# allow dm.fetch_statcast_range(...)
DataManager.fetch_statcast_range = fetch_statcast_range


# ---------------------------------------------------------------------
# Labels (score-delta when available; robust fallback)
# ---------------------------------------------------------------------
def build_first_inning_labels(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Build labels with YRFI using score deltas in the 1st inning.
    Fallback to RBI + text hints (scores/WP/PB/balk) if score columns missing.
    """
    req = ["game_pk", "game_date", "inning", "inning_topbot", "home_team", "away_team"]
    assert_columns(pbp, req, "build_first_inning_labels")

    df = pbp.copy()
    # normalize time columns
    df["game_datetime_utc"] = pd.to_datetime(df.get("game_datetime_utc", pd.NaT), utc=True, errors="coerce")
    df["game_time_utc"]     = pd.to_datetime(df.get("game_time_utc", df["game_datetime_utc"]), utc=True, errors="coerce")

    # first inning only
    df1 = df[df["inning"] == 1].copy()

    has_scores = {"home_score", "away_score"}.issubset(df1.columns)
    if has_scores:
        df1 = df1.sort_values(["game_pk", "game_time_utc", "game_datetime_utc"])

        def _runs_in_inning(g: pd.DataFrame) -> int:
            dh = (g["home_score"].max() - g["home_score"].min())
            da = (g["away_score"].max() - g["away_score"].min())
            return int(max(0, dh) + max(0, da))

        r = df1.groupby("game_pk").apply(_runs_in_inning).rename("first_inning_runs").reset_index()
        meta = df1.groupby("game_pk", as_index=False).agg(
            game_date=("game_date", "first"),
            game_time_utc=("game_time_utc", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
        by_game = meta.merge(r, on="game_pk", how="left")
    else:
        # Fallback (no score columns): RBI + textual hints to catch WP/PB/balk
        if "rbi" not in df1.columns:
            df1["rbi"] = 0
        desc = df1.get("description", pd.Series([""] * len(df1)))
        ev = df1.get("events", pd.Series([""] * len(df1)))
        scored_text = desc.fillna("").str.contains("scores", case=False)
        wp_pb_balk = ev.fillna("").str.contains("wild_pitch|passed_ball|balk", case=False)
        df1["run_flag"] = (df1["rbi"] > 0) | scored_text | wp_pb_balk

        by_game = df1.groupby("game_pk", as_index=False).agg(
            first_inning_runs=("run_flag", "sum"),
            game_date=("game_date", "first"),
            game_time_utc=("game_time_utc", "first"),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )

    by_game["yrfi"] = (by_game["first_inning_runs"] > 0).astype(int)
    by_game["game_datetime_utc"] = pd.to_datetime(by_game["game_time_utc"], utc=True)
    by_game["date"] = pd.to_datetime(by_game["game_date"])
    by_game["game_id"] = (
        by_game["date"].dt.strftime("%Y-%m-%d")
        + "_"
        + by_game["away_team"]
        + "_"
        + by_game["home_team"]
    )
    return by_game[["date","game_id","game_pk","game_datetime_utc","away_team","home_team","yrfi"]]


# ---------------------------------------------------------------------
# Schedule → game_datetime_utc (cached) + fast tz fallback
# ---------------------------------------------------------------------
def fetch_schedule_probables(target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Minimal per-day schedule fetch with probable pitchers and UTC gameDate.
    Response is normalized later by normalize_team_codes in attach_scheduled_times.
    """
    if target_date is None:
        target_date = pd.Timestamp.utcnow().date().isoformat()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}&hydrate=probablePitcher(note)"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        js = r.json()
        rows = []
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
        # tiny offline stub
        sample = pd.DataFrame([dict(
            date=pd.to_datetime("2024-04-12"),
            game_pk="2024-04-12_NYY_BOS",
            game_datetime_utc=pd.to_datetime("2024-04-12T23:05:00Z"),
            away_team="NYY", home_team="BOS",
            probable_away="A Sample", probable_home="B Sample"
        )])
        sample["game_id"] = sample["date"].dt.strftime("%Y-%m-%d") + "_" + sample["away_team"] + "_" + sample["home_team"]
        return sample


def attach_scheduled_times(
    labels: pd.DataFrame,
    stadiums: pd.DataFrame,
    default_local_time: str = "19:00",
    reference_dir: Path | str | None = None,
    use_schedule: bool = True,
    cache_seconds: int = 86400,  # 1 day
) -> pd.DataFrame:
    """
    Attaches labels['game_datetime_utc'] using:
      1) Cached MLB schedule by day (fast)
      2) Live schedule (optional), then cache
      3) Fallback: stadium timezone + default first-pitch local time
    """
    labels = labels.copy()
    labels["date"] = pd.to_datetime(labels["date"])

    cache_root = Path(reference_dir) if reference_dir else Path("data/reference")
    cache_dir = cache_root / "schedule_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # collect schedule rows
    sched_cols = ["game_id", "game_datetime_utc", "game_pk"]
    schedule = pd.DataFrame(columns=sched_cols)

    if use_schedule:
        dates = sorted(labels["date"].dt.date.unique())
        parts: List[pd.DataFrame] = []
        now = pd.Timestamp.utcnow()

        for d in dates:
            cf = cache_dir / f"{d.isoformat()}.parquet"
            # try cache
            if cf.exists():
                try:
                    age = now - pd.to_datetime(pd.Timestamp(cf.stat().st_mtime, unit="s"), utc=True)
                    if age.total_seconds() <= cache_seconds:
                        parts.append(pd.read_parquet(cf))
                        continue
                except Exception:
                    pass
            # live fetch
            try:
                sd = fetch_schedule_probables(d.isoformat())
                if not sd.empty:
                    sd["away_team"] = normalize_team_codes(sd["away_team"])
                    sd["home_team"] = normalize_team_codes(sd["home_team"])
                    sd["game_id"] = sd["date"].dt.strftime("%Y-%m-%d") + "_" + sd["away_team"] + "_" + sd["home_team"]
                    keep = sd[["game_id", "game_datetime_utc", "game_pk"]].copy()
                    parts.append(keep)
                    try:
                        keep.to_parquet(cf, index=False)
                    except Exception:
                        pass
            except Exception:
                continue

        if parts:
            schedule = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["game_id"])

    # 1) merge schedule times
    out = labels.merge(schedule[["game_id", "game_datetime_utc"]], on="game_id", how="left")

    # 2) tz fallback
    need = out["game_datetime_utc"].isna()
    if need.any():
        assert_columns(stadiums, ["team_code", "timezone"], "stadiums ref")
        tz_map = stadiums[["team_code", "timezone"]].rename(columns={"team_code": "home_team"})
        tmp = out.loc[need].merge(tz_map, on="home_team", how="left")

        local_naive = pd.to_datetime(tmp["date"].dt.strftime("%Y-%m-%d") + " " + default_local_time)
        local = local_naive.dt.tz_localize(tmp["timezone"].fillna("UTC"), nonexistent="NaT", ambiguous="NaT")
        out.loc[need, "game_datetime_utc"] = local.dt.tz_convert("UTC").values

    return out


# ---------------------------------------------------------------------
# Stadiums & Park Factors
# ---------------------------------------------------------------------
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
    """
    Very rough park factor example; falls back to reference file if pbp missing.
    """
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


# ---------------------------------------------------------------------
# Weather (kept minimal with fallback in notebooks)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Sample bundle writer (for offline debugging)
# ---------------------------------------------------------------------
def write_sample_bundle(out_dir: Path, labels: pd.DataFrame, pbp: pd.DataFrame, stadiums: pd.DataFrame, weather: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.head(200).to_csv(Path(out_dir) / "labels.csv", index=False)
    pbp.head(1000).to_csv(Path(out_dir) / "pbp.csv", index=False)
    stadiums.to_csv(Path(out_dir) / "stadiums.csv", index=False)
    weather.head(200).to_csv(Path(out_dir) / "weather.csv", index=False)
