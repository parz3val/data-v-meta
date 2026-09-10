"""Reference feature family = part of f(X) (configs/features.yaml reference). Each fn: interim
DataFrame -> pd.Series."""
from __future__ import annotations

import pandas as pd


def ref_n(df: pd.DataFrame) -> pd.Series:
    """Count of <ref> tags."""
    return df["ref_n"]


def refs_per_1k_words(df: pd.DataFrame) -> pd.Series:
    """Ref density: ref_n per 1,000 words of clean text."""
    return df["ref_n"] / df["word_count"].clip(lower=1) * 1000


def refs_distinct(df: pd.DataFrame) -> pd.Series:
    """Count of distinct (deduplicated by content) ref tags."""
    return df["refs_distinct"]


def cite_template_mix(df: pd.DataFrame) -> pd.Series:
    """Share of refs using a Citation Style {{cite ...}}/{{citation}} template, capped at 1."""
    return (df["cite_templates_n"] / df["ref_n"].clip(lower=1)).clip(upper=1.0)


def doi_n(df: pd.DataFrame) -> pd.Series:
    """Count of doi= parameter occurrences in the wikitext."""
    return df["doi_n"]


def isbn_n(df: pd.DataFrame) -> pd.Series:
    """Count of isbn= parameter occurrences in the wikitext."""
    return df["isbn_n"]


FEATURES = {
    "ref_n": ref_n, "refs_per_1k_words": refs_per_1k_words, "refs_distinct": refs_distinct,
    "cite_template_mix": cite_template_mix, "doi_n": doi_n, "isbn_n": isbn_n,
}
