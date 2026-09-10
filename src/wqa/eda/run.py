"""P3 EDA session (SPEC §5 P3, OPS §3.1/§3.1b). Reads data/<mode>/features/features.parquet
(+ interim word_count for truncation), writes results/<mode>/eda/<run_id>/, repoints
eda/current, generates a read-only notebooks/eda_<mode>_<run_id>.ipynb.
NOTE: report/generated/<mode>/eda_*.tex is out of this package's write scope (see
docs/decisions/pipe.md) -- the .tex/.json twins are written to <run_id>/report_tex/ instead.
Usage: from wqa.eda.run import run; run("small")."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat as nbf
import pandas as pd
import structlog
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

from wqa import charts
from wqa.config import ROOT
from wqa.monitor.budget import budget

log = structlog.get_logger()
TOP_N_DIST = 12
LEAKAGE_ACC_THRESHOLD = 0.80
TRUNCATION_BINS = [(0, 512), (512, 4096), (4096, float("inf"))]
TRUNCATION_LABELS = ["<=512w", "513-4096w", ">4096w"]


def run(mode: str) -> dict[str, Any]:
    with budget("P3_eda"):
        features = pd.read_parquet(ROOT / "data" / mode / "features" / "features.parquet")
        interim = pd.read_parquet(ROOT / "data" / mode / "interim" / "full.parquet",
                                  columns=["pageid", "word_count"])
        df = features.merge(interim, on="pageid", how="left")
        feat_cols = [c for c in features.columns if c not in ("pageid", "label", "grade_rank")]

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / "results" / mode / "eda" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        report_dir = out_dir / "report_tex"

        stats = _feature_stats(df, feat_cols)
        summary, flagged = _feature_summary(df, feat_cols)
        stats.to_parquet(out_dir / "feature_stats.parquet", index=False)
        summary.to_parquet(out_dir / "feature_summary.parquet", index=False)

        chart_ids = _write_charts(df, summary, out_dir, report_dir)
        (out_dir / "notes.md").write_text(_notes(df, summary, flagged))
        _write_index_and_session(out_dir, mode, run_id, df, feat_cols, chart_ids, flagged)
        _repoint_current(mode, run_id)
        _write_notebook(mode, run_id, chart_ids)

    log.info("eda_done", run_id=run_id, rows=len(df), flagged=len(flagged))
    return {"run_id": run_id, "rows": len(df), "flagged": flagged}


def _feature_stats(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """One row per (feature, grade): count/mean/std/p10/p50/p90/missing (UI_SPEC §4.3)."""
    rows = []
    for feat in feat_cols:
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        for grade, g in df.groupby("label"):
            s = g[feat]
            rows.append({"feature": feat, "grade": grade, "count": int(s.count()),
                        "mean": s.mean(), "std": s.std(), "p10": s.quantile(0.10),
                        "p50": s.quantile(0.50), "p90": s.quantile(0.90),
                        "missing": float(s.isna().mean())})
    return pd.DataFrame(rows)


def _feature_summary(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """One row per feature: MI with label, Spearman with ordinal grade, missingness, and a
    single-feature train-accuracy smell test (>=0.80 -> leakage suspect, SPEC §5 P3)."""
    numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    # median-impute, then 0-fill any column that is entirely NaN (e.g. top_editor_share,
    # NOT COLLECTED -- its MI/accuracy come out as ~0/uninformative rather than crashing).
    x_filled = df[numeric_cols].apply(lambda c: c.fillna(c.median())).fillna(0.0)
    mi = mutual_info_classif(x_filled, df["grade_rank"], random_state=42)
    rows, flagged = [], []
    for feat, mi_val in zip(numeric_cols, mi, strict=True):
        spearman = df[feat].corr(df["grade_rank"], method="spearman")
        missing = float(df[feat].isna().mean())
        acc = _single_feature_accuracy(x_filled[[feat]], df["label"])
        rows.append({"feature": feat, "mi": float(mi_val), "spearman": spearman,
                    "missing_frac": missing, "single_feature_acc": acc})
        if acc >= LEAKAGE_ACC_THRESHOLD:
            flagged.append({"feature": feat, "single_feature_acc": round(acc, 3)})
    summary = pd.DataFrame(rows).sort_values("mi", ascending=False).reset_index(drop=True)
    return summary, flagged


def _single_feature_accuracy(x_one: pd.DataFrame, y: pd.Series) -> float:
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(x_one, y)
    return float(clf.score(x_one, y))


def _write_charts(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path,
                  report_dir: Path) -> list[str]:
    grades = [g for g in charts.GRADE_COLOR if g in set(df["label"])]
    counts = df["label"].value_counts().reindex(grades).fillna(0).astype(int)
    ids = [_emit(charts.spec("eda_balance", "Grade balance", "bar", x=grades,
                             series=[{"name": "n", "y": counts.tolist()}],
                             xlabel="grade", ylabel="count"), out_dir, report_dir)]

    top = summary.head(TOP_N_DIST)["feature"].tolist()
    corr = df[top].corr(method="spearman")
    corr_series = [{"name": f, "y": corr[f].round(3).tolist()} for f in top]
    ids.append(_emit(charts.spec("eda_corr", "Feature correlation (Spearman, top-MI)", "table",
                                 x=top, series=corr_series, notes="rows=cols=top-MI features"),
                     out_dir, report_dir))

    for feat in top:
        med = df.groupby("label")[feat].median().reindex(grades)
        ids.append(_emit(charts.spec(f"eda_dist_{feat}", f"{feat} by grade (median)", "bar",
                                     x=grades, series=[{"name": feat, "y": med.tolist()}],
                                     xlabel="grade", ylabel=feat,
                                     notes="median per grade; p10/p90 in feature_stats.parquet"),
                         out_dir, report_dir))

    counts_bin = [int(((df["word_count"] >= lo) & (df["word_count"] < hi)).sum())
                 for lo, hi in TRUNCATION_BINS]
    ids.append(_emit(charts.spec("eda_truncation", "Token/word truncation buckets", "bar",
                                 x=TRUNCATION_LABELS, series=[{"name": "n_articles", "y": counts_bin}],
                                 xlabel="word_count bucket", ylabel="n_articles",
                                 notes="word_count is a token-count proxy, SPEC text_input"),
                     out_dir, report_dir))
    return ids


def _emit(spec_: dict[str, Any], out_dir: Path, report_dir: Path) -> str:
    charts.save_spec(spec_, out_dir)
    charts.to_png(spec_, out_dir / f"{spec_['id']}.png")
    charts.save_spec(spec_, report_dir)
    (report_dir / f"{spec_['id']}.tex").write_text(charts.to_tex(spec_))
    return spec_["id"]


def _notes(df: pd.DataFrame, summary: pd.DataFrame, flagged: list[dict[str, Any]]) -> str:
    top3 = ", ".join(summary.head(3)["feature"]) if len(summary) else "n/a"
    bal = df["label"].value_counts().to_dict()
    worst_missing = (summary.loc[summary["missing_frac"].idxmax()]
                     if len(summary) else None)
    lines = ["# EDA notes", f"- {len(df)} articles, {df['label'].nunique()} grades; counts: {bal}",
             f"- Top-MI features: {top3}"]
    if worst_missing is not None:
        lines.append(f"- Highest missingness: {worst_missing['feature']} "
                     f"({worst_missing['missing_frac']:.0%})")
    if flagged:
        names = ", ".join(f"{f['feature']}({f['single_feature_acc']})" for f in flagged)
        lines.append(f"- LEAKAGE SUSPECT (train-acc>=0.80 alone): {names} -- see docs/decisions/pipe.md")
    else:
        lines.append("- No single feature reaches >=0.80 train-accuracy alone")
    return "\n".join(lines) + "\n"


def _write_index_and_session(out_dir: Path, mode: str, run_id: str, df: pd.DataFrame,
                             feat_cols: list[str], chart_ids: list[str],
                             flagged: list[dict[str, Any]]) -> None:
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    index = {"run_id": run_id, "mode": mode, "created": created, "n_rows": len(df),
             "n_features": len(feat_cols), "charts": chart_ids, "leakage_suspects": flagged}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    summary_line = f"{len(df)} rows, {len(feat_cols)} features, {len(flagged)} leakage suspects"
    (out_dir / "session.json").write_text(json.dumps({"created": created, "summary": summary_line},
                                                      indent=1))


def _repoint_current(mode: str, run_id: str) -> None:
    base = ROOT / "results" / mode / "eda"
    current = base / "current"
    if current.is_symlink() or current.exists():
        current.unlink()
    current.symlink_to(run_id)


def _write_notebook(mode: str, run_id: str, chart_ids: list[str]) -> None:
    """One cell per chart, reads parquet/PNG from results/ directly -- no pipeline imports."""
    rel = f"../results/{mode}/eda/{run_id}"
    cells = [nbf.v4.new_markdown_cell(f"# EDA {mode} {run_id}\nGenerated by `make eda`; read-only."),
            nbf.v4.new_code_cell(
                "import json\nimport pandas as pd\n"
                f"stats = pd.read_parquet('{rel}/feature_stats.parquet')\n"
                f"summary = pd.read_parquet('{rel}/feature_summary.parquet')\n"
                "summary.sort_values('mi', ascending=False).head(12)")]
    for cid in chart_ids:
        cells.append(nbf.v4.new_code_cell(
            "import matplotlib.pyplot as plt\nimport matplotlib.image as mpimg\n"
            f"spec = json.load(open('{rel}/{cid}.json'))\n"
            f"plt.figure(figsize=(6, 4)); plt.imshow(mpimg.imread('{rel}/{cid}.png')); "
            "plt.axis('off'); plt.title(spec['title'])"))
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb_dir = ROOT / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, nb_dir / f"eda_{mode}_{run_id}.ipynb")
