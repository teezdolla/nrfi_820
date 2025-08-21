from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Iterable, List, Optional, Dict, Any

def set_seed(seed: int = 42) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        import xgboost as xgb
        xgb.random.set_global_seed(seed)
    except Exception:
        pass

def _ver(mod):
    try:
        return mod.__version__
    except Exception:
        return "n/a"

def print_run_header(name: str) -> None:
    import pandas as pd, sklearn, numpy, platform
    try:
        import xgboost as xgb; xgbv = xgb.__version__
    except Exception:
        xgbv = "n/a"
    print(f"=== {name} ===")
    print(f"Python {sys.version.split()[0]} | pandas {_ver(pd)} | numpy {_ver(np)} | sklearn {_ver(sklearn)} | xgboost {xgbv}")
    print(f"Platform: {platform.system()} {platform.release()} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

class Timer:
    def __init__(self, label: str): self.label = label
    def __enter__(self): self.t0 = time.time(); print(f"[{self.label}] start"); return self
    def __exit__(self, exc_type, exc, tb): print(f"[{self.label}] done in {time.time()-self.t0:.2f}s")

def assert_non_empty(df: pd.DataFrame, stage: str):
    assert isinstance(df, pd.DataFrame), f"{stage}: not a DataFrame"
    assert len(df) > 0, f"{stage}: empty dataframe"

def assert_columns(df: pd.DataFrame, cols: Iterable[str], stage: str):
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"{stage}: missing columns {missing}"

def assert_no_all_nan(df: pd.DataFrame, cols: Iterable[str], stage: str):
    bad = [c for c in cols if c in df.columns and df[c].isna().all()]
    assert not bad, f"{stage}: all-NaN columns {bad}"

def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def try_import_duckdb():
    try:
        import duckdb
        return duckdb
    except Exception:
        return None

def asof_join(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str,
              by: Optional[List[str]] = None, direction: str = "backward") -> pd.DataFrame:
    duckdb = try_import_duckdb()
    if duckdb is not None and not by:
        ldf = left.copy(); rdf = right.copy()
        ldf[left_on] = pd.to_datetime(ldf[left_on]); rdf[right_on] = pd.to_datetime(rdf[right_on])
        return pd.merge_asof(ldf.sort_values(left_on), rdf.sort_values(right_on),
                             left_on=left_on, right_on=right_on, direction=direction)
    return pd.merge_asof(left.sort_values(left_on), right.sort_values(right_on),
                         left_on=left_on, right_on=right_on, by=by, direction=direction)

def strict_asof_check(features: pd.DataFrame, time_col: str, game_time_col: str, stage: str):
    bad = features[features[time_col] >= features[game_time_col]]
    assert bad.empty, f"{stage}: leakage detected (feature time >= game time) in {len(bad)} rows"

def to_datetime(x: Any) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")

def save_joblib(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path: Path) -> Any:
    return joblib.load(path)
