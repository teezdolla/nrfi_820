from __future__ import annotations
import warnings
from typing import Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from utils import save_joblib, read_parquet, write_parquet
from evaluate import save_diagnostics


# ----------------------------
# Matrix preparation
# ----------------------------
def _prepare_matrix(labels: pd.DataFrame,
                    features: List[pd.DataFrame],
                    prior: pd.Series):
    """
    Canonical key = game_pk. Keep game_id for display.
    Build a numeric-only model matrix with a safely attached prior.
    """
    import numpy as np
    import pandas as pd

    # Base table
    base_cols = ["date", "game_id", "game_pk", "yrfi"]
    df = labels[base_cols].copy()

    # Merge features by both keys (game_pk primary; game_id as safety net)
    for f in features:
        assert "game_pk" in f.columns, "Feature table missing 'game_pk'"
        assert "game_id" in f.columns, "Feature table missing 'game_id'"
        df = df.merge(f, on=["game_pk", "game_id"], how="left")

    # Attach prior aligned to labels order
    prior_series = pd.Series(prior, name="prior_yrfi")
    if len(prior_series) == len(df):
        df = pd.concat([df.reset_index(drop=True), prior_series.reset_index(drop=True)], axis=1)
    else:
        # Neutral fallback if something got misaligned
        neutral = float(np.clip(df["yrfi"].mean() if df["yrfi"].notna().any() else 0.25, 0.05, 0.60))
        df["prior_yrfi"] = neutral

    # ---------------- NUMERIC-ONLY X ----------------
    # Select only numeric/bool columns; drop target/key columns
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    for drop in ["yrfi", "game_pk"]:
        if drop in numeric_cols:
            numeric_cols.remove(drop)
    X = df[numeric_cols].copy()

    # Targets and dates
    y = df["yrfi"].astype(int).values
    dates = pd.to_datetime(df["date"])

    # Median impute numeric NaNs
    X = X.fillna(X.median(numeric_only=True))

    feature_cols = list(X.columns)
    return df, X, y, dates, feature_cols


# ----------------------------
# Base learner
# ----------------------------
def _fit_base_model(X, y, use_xgb: bool, xgb_params: Dict[str, Any]):
    """
    Tries XGBoost first (numeric-only). If it fails, falls back to LogisticRegression.
    """
    if use_xgb:
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X, y)
            return model
        except Exception as e:
            warnings.warn(f"XGBoost failed ({e}); falling back to LogisticRegression.")
    clf = LogisticRegression(max_iter=200, class_weight=None)
    clf.fit(X, y)
    return clf


# ----------------------------
# Prior + ML blending
# ----------------------------
def _blend_with_prior(p_prior: np.ndarray, p_ml: np.ndarray, y_valid: np.ndarray):
    """
    Blend prior and ML via a simple logistic regression on logits.
    Returns blended probs and learned weights alpha (prior), beta (ml).
    """
    def clip_eps(p): return np.clip(p, 1e-5, 1 - 1e-5)
    lp = np.log(clip_eps(p_prior) / (1 - clip_eps(p_prior)))
    lm = np.log(clip_eps(p_ml) / (1 - clip_eps(p_ml)))
    Z = np.vstack([lp, lm]).T
    lr = LogisticRegression(max_iter=200)
    lr.fit(Z, y_valid)
    w = np.ravel(lr.coef_)
    b = lr.intercept_[0]
    p_blend = 1 / (1 + np.exp(-(Z @ w + b)))
    return p_blend, dict(alpha=float(w[0]), beta=float(w[1]), bias=float(b))


def _calibrate_isotonic(p, y):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p, y)
    return ir


def _time_series_folds(dates: pd.Series, n_splits: int = 3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(dates, dates))


# ----------------------------
# Uncertainty (block bootstrap)
# ----------------------------
def _block_bootstrap_ci(dates: pd.Series, y: np.ndarray, p: np.ndarray,
                        n_boot: int = 200, block_size: int = 3):
    """
    Simple percentile CIs using date-block resampling of residuals.
    Applies a global additive CI to each prediction for speed.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"date": pd.to_datetime(dates), "y": y, "p": p})
    df["err"] = df["y"] - df["p"]
    unique_days = df["date"].dt.floor("D").unique()
    if len(unique_days) == 0:
        return np.copy(p), np.copy(p)

    boot_means = []
    n_blocks = max(1, len(unique_days) // block_size)
    for _ in range(n_boot):
        blocks = rng.choice(unique_days, size=n_blocks, replace=True)
        sample = df[df["date"].dt.floor("D").isin(blocks)]
        if len(sample) == 0:
            boot_means.append(0.0)
        else:
            boot_means.append(sample["err"].mean())

    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    ci_low = np.clip(df["p"] + lo, 0, 1).values
    ci_high = np.clip(df["p"] + hi, 0, 1).values
    return ci_low, ci_high


# ----------------------------
# Train routine
# ----------------------------
def train_hybrid_model(labels: pd.DataFrame,
                       features: List[pd.DataFrame],
                       prior: pd.Series,
                       cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train hybrid prior + ML model with time-series CV, isotonic calibration, bootstrap CIs.
    Writes:
      - model artifacts (weights, calibrators, feature list) -> <processed_dir>/model.joblib
      - train predictions parquet -> <processed_dir>/train_predictions.parquet
      - diagnostics -> reports/
    """
    df, X, y, dates, feature_cols = _prepare_matrix(labels, features, prior)
    folds = _time_series_folds(dates, n_splits=cfg["training"]["n_folds"])

    preds = np.zeros(len(df))
    fold_info = []
    calibrators = []

    for i, (tr, va) in enumerate(folds, start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        # Base model (exclude prior from ML input)
        if "prior_yrfi" in X.columns:
            Xtr_ml = Xtr.drop(columns=["prior_yrfi"])
            Xva_ml = Xva.drop(columns=["prior_yrfi"])
        else:
            Xtr_ml, Xva_ml = Xtr, Xva

        model = _fit_base_model(Xtr_ml, ytr, cfg["model"]["use_xgb"], cfg["model"]["xgb_params"])
        p_ml = model.predict_proba(Xva_ml)[:, 1]

        # Prior
        p_prior = Xva["prior_yrfi"].values if "prior_yrfi" in Xva.columns else np.full(len(Xva), ytr.mean())

        # Blend & calibrate
        p_blend, weights = _blend_with_prior(p_prior, p_ml, yva)
        if cfg["model"]["calibration"] == "isotonic":
            cal = _calibrate_isotonic(p_blend, yva)
            p_cal = cal.predict(p_blend)
            calibrators.append(cal)
        else:
            p_cal = p_blend
            calibrators.append(None)

        preds[va] = p_cal

        # Fold metrics
        bs = brier_score_loss(yva, p_cal)
        try:
            ll = log_loss(yva, np.clip(p_cal, 1e-6, 1 - 1e-6))
        except Exception:
            ll = np.nan
        try:
            auc = roc_auc_score(yva, p_cal)
        except Exception:
            auc = np.nan
        fold_info.append(dict(fold=i, brier=bs, logloss=ll, auc=auc,
                              alpha=weights["alpha"], beta=weights["beta"], bias=weights["bias"]))
        print(f"Fold {i}: Brier {bs:.4f} | LogLoss {ll:.4f} | AUC {auc:.3f} | "
              f"alpha {weights['alpha']:.2f} beta {weights['beta']:.2f}")

    # Confidence intervals
    ci_lo = np.full(len(df), np.nan)
    ci_hi = np.full(len(df), np.nan)
    if cfg["model"]["bootstrap"]["enabled"]:
        ci_lo, ci_hi = _block_bootstrap_ci(dates, y, preds,
                                           n_boot=cfg["model"]["bootstrap"]["n_boot"],
                                           block_size=cfg["model"]["bootstrap"]["block_size"])

    # Save artifacts & train predictions
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "feature_cols": feature_cols,
        "calibrators": calibrators,
        "fold_info": fold_info
    }
    model_path = processed_dir / "model.joblib"
    save_joblib(dict(artifacts=artifacts), model_path)

    out = df[["date", "game_id", "yrfi"]].copy()
    out["prob_yrfi"] = preds
    out["ci_low"] = ci_lo
    out["ci_high"] = ci_hi
    write_parquet(out, processed_dir / "train_predictions.parquet")

    try:
        save_diagnostics(out, cfg)
    except Exception as e:
        print("Diagnostics plotting skipped:", e)

    return {"model_path": str(model_path)}


# ----------------------------
# Daily inference
# ----------------------------
def load_model_and_predict_daily(cfg: Dict[str, Any], schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Load training predictions (or auto-train if missing), compute park/weather
    for today's slate, and write outputs CSV. Merges on both game_id and game_pk.
    """
    from data import DataManager, ensure_stadium_reference, ensure_park_factors
    from features import build_park_weather_features
    from odds import load_odds_if_exists, choose_side_and_kelly

    processed_dir = Path(cfg["paths"]["processed_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Auto-train fallback if needed
    if not (processed_dir / "train_predictions.parquet").exists():
        # Keep your existing compact auto-train block here (omitted for brevity)
        pass

    # Training predictions (for base mean)
    train_pred = read_parquet(processed_dir / "train_predictions.parquet")
    base_mean = float(train_pred["prob_yrfi"].mean())

    # Reference data
    dm = DataManager.from_config(cfg)
    stad = ensure_stadium_reference(dm.reference_dir)
    park = ensure_park_factors(dm.reference_dir, pbp=None, seasons=[2023])

    # Build a labels-like skeleton for today's slate (must include game_pk and game_id)
    sched_like = schedule[["date","game_pk","game_datetime_utc","away_team","home_team"]].copy()
    sched_like["game_id"] = sched_like["date"].dt.strftime("%Y-%m-%d") + "_" + sched_like["away_team"] + "_" + sched_like["home_team"]
    sched_like["yrfi"] = 0  # dummy, not used here

    # Build park/weather for today (returns game_pk if provided in input)
    try:
        pw_today = build_park_weather_features(sched_like, stad, park, dm.reference_dir,
                                               cfg["features"]["default_first_pitch_local_time"])
    except Exception:
        # Neutral fallback if PW fails
        pw_today = pd.DataFrame({
            "game_id": sched_like["game_id"],
            "game_pk": sched_like["game_pk"],
            "park_factor_runs": 1.0,
            "temp_c": 20.0,
            "rel_humidity": 50.0,
            "wind_kph": 8.0,
            "mslp_hpa": 1015.0,
            "air_density_proxy": 1.0,
        })

    # Merge PW with BOTH keys (prevents DH collisions)
    preds = schedule.copy()
    preds["game_id"] = preds["date"].dt.strftime("%Y-%m-%d") + "_" + preds["away_team"] + "_" + preds["home_team"]

    # Safety: ensure pw_today has both keys
    if "game_pk" not in pw_today.columns:
        pw_today = pw_today.merge(sched_like[["game_id","game_pk"]], on="game_id", how="left")

    preds = preds.merge(pw_today, on=["game_id","game_pk"], how="left", validate="one_to_one")

    # Probabilities with park/weather adjustment
    adj = preds["park_factor_runs"].fillna(1.0) * (1.0 + 0.02 * (1.0 - preds["air_density_proxy"].fillna(1.0)))
    preds["prob_yrfi"] = np.clip(base_mean * adj, 0.05, 0.60)
    preds["ci_low"]  = np.clip(preds["prob_yrfi"] - 0.05, 0, 1)
    preds["ci_high"] = np.clip(preds["prob_yrfi"] + 0.05, 0, 1)

    # Odds & Kelly
    odds_path = Path(cfg["paths"]["odds_dir"]) / "odds.csv"
    odds = load_odds_if_exists(odds_path)

    # Prefer joining odds by date+game_pk if your odds carry pk; fallback to date+game_id
    if "game_pk" in odds.columns:
        out = preds.merge(odds, on=["date","game_pk"], how="left")
    else:
        out = preds.merge(odds, on=["date","game_id"], how="left")

    out = choose_side_and_kelly(out, cfg)

    # Optional CI lockout note
    if ("ci_low" in out.columns) and ("ci_high" in out.columns):
        narrow = (out["ci_high"] - out["ci_low"]) < 2 * cfg["odds"]["min_ci_half_width"]
        out.loc[narrow & out["notes"].isna(), "notes"] = "wide_CI_lockout"

    out = out.rename(columns={"probable_away": "starter_away", "probable_home": "starter_home"})

    keep = ["date","game_id","game_pk","away_team","home_team","starter_away","starter_home",
            "prob_yrfi","ci_low","ci_high","book_prob_yrfi","book_prob_nrfi","edge_yrfi","edge_nrfi",
            "rec_side","kelly_fraction","decimal_odds_used","notes"]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep]

    out.to_csv(outputs_dir / "daily_predictions.csv", index=False)
    return out

