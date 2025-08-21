#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from utils import print_run_header
from data import DataManager, fetch_schedule_probables
from model import load_model_and_predict_daily

def main():
    print_run_header("run_daily")
    ap=argparse.ArgumentParser()
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (defaults to today)")
    args=ap.parse_args()

    cfg=yaml.safe_load(Path("configs/config.yaml").read_text(encoding="utf-8"))
    dm=DataManager.from_config(cfg); dm.ensure_dirs()
    sched=fetch_schedule_probables(args.date)
    preds=load_model_and_predict_daily(cfg, schedule=sched)
    out=Path(cfg["paths"]["outputs_dir"])/"daily_predictions.csv"
    print("Wrote:", out)

if __name__=="__main__":
    main()
