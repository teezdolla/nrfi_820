# NRFI / YRFI Modeling (leak-safe, reproducible)

Predict YRFI vs NRFI with calibrated probabilities, confidence bands, and bet sizing.
Open data only: `pybaseball` (Statcast/PBP), MLB Stats API (schedule + probables), `meteostat` (weather).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke run with sample data (no network needed)
python scripts/run_daily.py --date 2024-04-12
# -> outputs/daily_predictions.csv
```

## Repro

1. Open notebooks in order:

- `01_data_build.ipynb` fetch/cache raw, label games, create reference files
- `02_feature_store.ipynb` leak-safe features and persistence
- `03_model_train.ipynb` hybrid prior + ML, calibration, CIs
- `04_backtest_odds.ipynb` profitability vs market (needs `data/odds/odds.csv`)
- `05_daily_pregame.ipynb` today’s predictions

2. Run tests

```bash
pytest -q
```

## Guardrails

- **Universal project root detection** in notebooks (works from repo root or `notebooks/`)
- Strict **as-of** joins (DuckDB when available, fallback to `merge_asof`)
- Required column assertions with friendly errors
- Fallbacks: if an API fails, pipeline uses cached or sample data; never crashes
- Confidence bands via **block bootstrap** on time-ordered residuals
- Kelly staking with **CI lockout** to avoid thin edges

## Data sources

- Statcast/PBP: `pybaseball`
- Schedule & Probables: MLB Stats API `/api/v1/schedule?hydrate=probablePitcher`
- Weather: `meteostat` hourly at stadium lat/lon
- Stadium coords/timezones: compiled from public sources
- Park factors: league-normalized runs/game (see `src/data.py::compute_park_factors`)

## Output schema

`outputs/daily_predictions.csv`

```
date,game_id,away_team,home_team,starter_away,starter_home,prob_yrfi,ci_low,ci_high,book_prob_yrfi,book_prob_nrfi,edge_yrfi,edge_nrfi,rec_side,kelly_fraction,decimal_odds_used,notes
```
"refresh index"