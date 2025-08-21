from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_calibration_curve(y_true, y_prob, path: Path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title("Calibration")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight"); plt.close()

def reliability_table(y_true, y_prob, n_bins=10) -> pd.DataFrame:
    q = pd.qcut(y_prob, q=n_bins, duplicates="drop")
    return pd.DataFrame({"y": y_true, "p": y_prob, "bin": q}).groupby("bin", as_index=False).agg(
        n=("y","size"), mean_pred=("p","mean"), mean_obs=("y","mean"))

def save_diagnostics(train_pred: pd.DataFrame, cfg: dict):
    y = train_pred["yrfi"].values; p = train_pred["prob_yrfi"].values
    plot_calibration_curve(y, p, Path(cfg["paths"]["reports_dir"]) / "calibration.png")
    reliability_table(y, p).to_csv(Path(cfg["paths"]["reports_dir"]) / "reliability.csv", index=False)
