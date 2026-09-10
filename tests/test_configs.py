"""WP1 skeleton: configs load and are sane."""
from pathlib import Path

import yaml

from wqa import config

ROOT = Path(__file__).resolve().parents[1]


def test_core_configs_load() -> None:
    for name in ("collection", "features", "splits", "budgets"):
        assert config.load(name)


def test_email_not_blank() -> None:
    assert config.user_agent(config.load("collection")).startswith("wqa-msc-project/0.1 (")


def test_splits_sum_to_one() -> None:
    s = config.load("splits")
    assert abs(s["train"] + s["val"] + s["test"] - 1.0) < 1e-9 and s["freeze"] is True


def test_six_grades() -> None:
    assert config.load("collection")["grades"] == ["Stub", "Start", "C", "B", "GA", "FA"]


def test_model_configs_load() -> None:
    files = list((ROOT / "configs/models").glob("*.yaml"))
    assert len(files) >= 13
    for f in files:
        cfg = yaml.safe_load(f.read_text())
        assert cfg.get("model") and cfg.get("features")
    assert (ROOT / "configs/models/metadata_noH.yaml").exists()
    assert (ROOT / "configs/models/text_plus_H.yaml").exists()
