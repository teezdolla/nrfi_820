from pathlib import Path
import yaml, numpy as np
from src.data import DataManager, fetch_statcast_range, build_first_inning_labels, ensure_stadium_reference, ensure_park_factors, fetch_weather_for_games
from src.features import build_team_fi_features, build_starter_fi_features, build_pitch_matchup_features, build_park_weather_features
from src.lineups import build_lineup_priors
from src.markov import build_first_inning_prior
from src.model import train_hybrid_model
from src.odds import american_to_decimal, devig_two_way

def test_train_and_ci_and_odds_math():
    cfg=yaml.safe_load(Path("configs/config.yaml").read_text())
    dm=DataManager.from_config(cfg); dm.ensure_dirs()
    pbp=fetch_statcast_range(cfg["training"]["start_date"], cfg["training"]["end_date"])
    labels=build_first_inning_labels(pbp)
    stad=ensure_stadium_reference(dm.reference_dir)
    park=ensure_park_factors(dm.reference_dir, pbp=None, seasons=[2023])
    wx=fetch_weather_for_games(labels, stad, cfg["features"]["default_first_pitch_local_time"], dm.reference_dir)
    team=build_team_fi_features(labels); starter=build_starter_fi_features(labels)
    lineup=build_lineup_priors(labels); pitch=build_pitch_matchup_features(labels)
    pw=build_park_weather_features(labels, stad, park, dm.reference_dir, cfg["features"]["default_first_pitch_local_time"])
    prior=build_first_inning_prior(labels, team, starter, lineup, pw)
    train_hybrid_model(labels,[team,starter,lineup,pitch,pw], prior, cfg)
    dec=american_to_decimal([110,-120])
    py,pn=devig_two_way(dec.iloc[0:1], dec.iloc[1:2])
    assert np.isclose((py.iloc[0]+pn.iloc[0]),1.0)
