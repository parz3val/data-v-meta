"""Structural feature family = f(X) (configs/features.yaml structural). Each fn: interim
DataFrame (from wqa.clean.run) -> pd.Series. Leakage guard: word/template/category counts are
already computed from wikitext with grade-icon templates/labeling categories stripped in
wqa.clean.parse (see configs/features.yaml leakage_exclusions, docs/decisions/pipe.md)."""
from __future__ import annotations

import pandas as pd


def bytes_(df: pd.DataFrame) -> pd.Series:
    """Raw wikitext byte length at the fetched revision."""
    return df["bytes"]


def words(df: pd.DataFrame) -> pd.Series:
    """Word count of the leakage-stripped plain text."""
    return df["word_count"]


def sentences(df: pd.DataFrame) -> pd.Series:
    """Approx. sentence count (regex sentence-boundary heuristic)."""
    return df["sentence_count"]


def sections_l2(df: pd.DataFrame) -> pd.Series:
    """Count of level-2 (==heading==) sections."""
    return df["section_l2_n"]


def sections_l3(df: pd.DataFrame) -> pd.Series:
    """Count of level-3 (===heading===) sections."""
    return df["section_l3_n"]


def infobox(df: pd.DataFrame) -> pd.Series:
    """1 if any non-icon template name starts with 'infobox'."""
    return df["infobox"].astype(int)


def images(df: pd.DataFrame) -> pd.Series:
    """Count of [[File:/Image:...]] links."""
    return df["images_n"]


def tables(df: pd.DataFrame) -> pd.Series:
    """Count of {| wikitable markers."""
    return df["tables_n"]


def lists(df: pd.DataFrame) -> pd.Series:
    """Count of bullet/numbered list item lines."""
    return df["lists_n"]


def wikilinks(df: pd.DataFrame) -> pd.Series:
    """Count of internal [[...]] links (File/Category excluded)."""
    return df["wikilinks_n"]


def extlinks(df: pd.DataFrame) -> pd.Series:
    """Count of bracketed external links."""
    return df["extlinks_n"]


def categories(df: pd.DataFrame) -> pd.Series:
    """Category count, grade-labeling categories excluded (leakage guard)."""
    return df["categories_n"]


def templates(df: pd.DataFrame) -> pd.Series:
    """Template count, grade icon templates excluded (leakage guard)."""
    return df["templates_n"]


def cleanup_tags(df: pd.DataFrame) -> pd.Series:
    """Count of cleanup/maintenance-tag templates (citation needed, cleanup, ...)."""
    return df["cleanup_tags_n"]


def lead_length(df: pd.DataFrame) -> pd.Series:
    """Character length of the lead (first ~500 chars of clean text)."""
    return df["lead_text"].fillna("").str.len().astype(int)


FEATURES = {
    "bytes": bytes_, "words": words, "sentences": sentences, "sections_l2": sections_l2,
    "sections_l3": sections_l3, "infobox": infobox, "images": images, "tables": tables,
    "lists": lists, "wikilinks": wikilinks, "extlinks": extlinks, "categories": categories,
    "templates": templates, "cleanup_tags": cleanup_tags, "lead_length": lead_length,
}
