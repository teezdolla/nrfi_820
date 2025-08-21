from pathlib import Path
import yaml
from src.data import DataManager, fetch_statcast_range, build_first_inning_labels, ensure_stadium_reference, ensure_park_factors, fetch_weather_for_games

def test_label_creation_and_weather():
    cfg=yaml.safe_load(Path("configs/config.yaml").read_text())
    dm=DataManager.from_config(cfg); dm.ensure_dirs()
    pbp=fetch_statcast_range(cfg["training"]["start_date"], cfg["training"]["end_date"])
    labels=build_first_inning_labels(pbp); assert len(labels)>0
    stad=ensure_stadium_reference(dm.reference_dir)
    pf=ensure_park_factors(dm.reference_dir, pbp=None, seasons=[2023])
    wx=fetch_weather_for_games(labels, stad, cfg["features"]["default_first_pitch_local_time"], dm.reference_dir)
    assert len(wx)>0 and {"temp_c","rel_humidity","wind_kph","mslp_hpa"}.issubset(wx.columns)
