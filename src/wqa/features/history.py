"""History + contributor + pageviews feature family = g(H) (configs/features.yaml comment).
top_editor_share is NOT COLLECTED: top-editor byte-share attribution needs a per-article
WikiWho/Xtools call, outside the WP3 collection budget -- emits NaN; wqa.features.build
records this in the features manifest row so it is visible, not silently dropped."""
from __future__ import annotations

import numpy as np
import pandas as pd


def edits(df: pd.DataFrame) -> pd.Series:
    """Total revisions counted (capped at HISTORY_CAP, see wqa.collect.enrich)."""
    return df["edits"]


def editors(df: pd.DataFrame) -> pd.Series:
    """Distinct editor usernames/IPs over the counted revisions."""
    return df["editors"]


def edits_per_editor(df: pd.DataFrame) -> pd.Series:
    """edits / editors."""
    return df["edits"] / df["editors"].clip(lower=1)


def anon_share(df: pd.DataFrame) -> pd.Series:
    """Fraction of counted edits made by unregistered (IP) editors."""
    return df["anon_n"] / df["edits"].clip(lower=1)


def bot_share(df: pd.DataFrame) -> pd.Series:
    """Fraction of counted edits made by bot-named accounts (username heuristic)."""
    return df["bot_n"] / df["edits"].clip(lower=1)


def reverted_n(df: pd.DataFrame) -> pd.Series:
    """Count of edits tagged rollback/undo/manual revert."""
    return df["reverted_n"]


def age_days(df: pd.DataFrame) -> pd.Series:
    """Days between the earliest counted revision and fetch time."""
    return _days_between(df["created"], df["fetch_ts"])


def days_since_edit(df: pd.DataFrame) -> pd.Series:
    """Days between the latest revision and fetch time."""
    return _days_between(df["last_edit"], df["fetch_ts"])


def edits_90d(df: pd.DataFrame) -> pd.Series:
    """Edits within the trailing 90 days of fetch time."""
    return df["edits_90d"]


def top_editor_share(df: pd.DataFrame) -> pd.Series:
    """NOT COLLECTED in SMALL (needs WikiWho/Xtools; out of WP3 budget). Always NaN."""
    return pd.Series(np.nan, index=df.index)


def views_90d_mean(df: pd.DataFrame) -> pd.Series:
    """Mean daily pageviews over the trailing 90 days."""
    return df["pageviews_90d_mean"]


def views_90d_log(df: pd.DataFrame) -> pd.Series:
    """log1p of views_90d_mean (skew correction)."""
    return np.log1p(df["pageviews_90d_mean"])


def _days_between(start_col: pd.Series, end_col: pd.Series) -> pd.Series:
    start = pd.to_datetime(start_col, utc=True, errors="coerce")
    end = pd.to_datetime(end_col, utc=True, errors="coerce")
    return (end - start).dt.total_seconds() / 86400.0


FEATURES = {
    "edits": edits, "editors": editors, "edits_per_editor": edits_per_editor,
    "anon_share": anon_share, "bot_share": bot_share, "reverted_n": reverted_n,
    "age_days": age_days, "days_since_edit": days_since_edit, "edits_90d": edits_90d,
    "top_editor_share": top_editor_share, "views_90d_mean": views_90d_mean,
    "views_90d_log": views_90d_log,
}
