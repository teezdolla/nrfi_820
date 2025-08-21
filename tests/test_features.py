from pathlib import Path
import yaml
from src.features import build_team_fi_features, build_starter_fi_features
from src.data import DataManager, fetch_statcast_range, build_first_inning_labels

def test_no_leakage_feature_times():
    cfg=yaml.safe_load(Path("configs/config.yaml").read_text())
    dm=DataManager.from_config(cfg); dm.ensure_dirs()
    pbp=fetch_statcast_range(cfg["training"]["start_date"], cfg["training"]["end_date"])
    labels=build_first_inning_labels(pbp)
    team=build_team_fi_features(labels); starter=build_starter_fi_features(labels)
    assert len(team)==len(labels) and len(starter)==len(labels)
