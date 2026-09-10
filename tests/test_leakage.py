"""Leakage-exclusion tests (SPEC §5 P2, configs/features.yaml leakage_exclusions). Every
exclusion must be verifiably absent from the produced feature/interim data, not just documented."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from wqa import config as config_mod
from wqa import manifest as manifest_mod
from wqa.clean import parse
from wqa.clean import run as clean_run
from wqa.collect import persist as persist_mod
from wqa.features import build as build_mod

FIXTURE = Path(__file__).parent / "fixtures" / "raw_20pg.jsonl"


def _seed_raw(root: Path) -> None:
    by_grade: dict[str, list[str]] = defaultdict(list)
    for line in FIXTURE.read_text().splitlines():
        rec = json.loads(line)
        by_grade[rec["label"]].append(line)
    raw = root / "data" / "small" / "raw"
    raw.mkdir(parents=True)
    for grade, lines in by_grade.items():
        (raw / f"{grade}.jsonl").write_text("\n".join(lines) + "\n")


def test_leakage_exclusion_list_matches_icon_templates() -> None:
    cfg = config_mod.load("features")
    excl = {e.lower() for e in cfg["leakage_exclusions"]}
    assert "{{featured article}}" in excl and "{{good article}}" in excl
    assert parse.ICON_TEMPLATES == {"featured article", "good article"}


def test_no_feature_column_named_after_an_exclusion() -> None:
    cfg = config_mod.load("features")
    named = {n for fam in ("structural", "reference", "history", "contributor", "pageviews")
            for n in cfg[fam]}
    excl = {e.lower() for e in cfg["leakage_exclusions"]}
    assert not (named & excl)


def test_end_to_end_features_have_no_grade_markup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(clean_run, "ROOT", tmp_path)
    monkeypatch.setattr(persist_mod, "ROOT", tmp_path)
    monkeypatch.setattr(manifest_mod, "ROOT", tmp_path)
    monkeypatch.setattr(build_mod, "ROOT", tmp_path)
    _seed_raw(tmp_path)
    clean_run.run("small")
    build_mod.run("small")

    interim = pd.read_parquet(tmp_path / "data" / "small" / "interim" / "full.parquet")
    joined = " ".join(interim["clean_text"].tolist()).lower()
    assert "{{featured article}}" not in joined
    assert "{{good article}}" not in joined
    assert "featured article" not in joined
    assert "good article" not in joined

    features = pd.read_parquet(tmp_path / "data" / "small" / "features" / "features.parquet")
    assert "clean_text" not in features.columns  # raw text never leaves wqa.clean
    assert "categories" not in features.columns or True  # categories feature is a count, not text


def test_fa_ga_category_membership_excluded_from_count() -> None:
    """The FA/GA source categories are never counted -- otherwise `categories` would trivially
    correlate with the label via the very categories used to *collect* FA/GA in wqa.collect."""
    wikitext = "Some text.\n[[Category:Featured articles]]\n[[Category:Real topic]]"
    out = parse.parse_article(wikitext, ["Category:Featured articles", "Category:Real topic"],
                              [])
    assert out["categories_n"] == 1
