"""Chart spec -> {PNG, Chart.js dict, LaTeX booktabs+.json twin} (OPS §3.1). One spec-dict format
for every chart id (eda_*, select_candidates, confusion_*, ...); this module only renders --
callers build the spec dict. Grade colours from UI_SPEC §1; PALETTE is seaborn "deep",
colourblind-safe. Kept <=200 lines by design (OPS §3.1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GRADE_COLOR = {"Stub": "#F0A3A3", "Start": "#F5B97D", "C": "#EFE07A", "B": "#B9E486",
              "GA": "#7FDDA1", "FA": "#8FB6F5"}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
Kind = Literal["bar", "line", "scatter", "hist", "table"]


def spec(id: str, title: str, kind: Kind, x: list[Any], series: list[dict[str, Any]],
        xlabel: str = "", ylabel: str = "", xscale: str | None = None,
        notes: str = "") -> dict[str, Any]:
    """Build one chart spec dict (OPS §3.1 schema)."""
    return {"id": id, "title": title, "kind": kind, "x": x, "series": series,
           "xlabel": xlabel, "ylabel": ylabel, "xscale": xscale, "notes": notes}


def save_spec(spec_: dict[str, Any], out_dir: Path) -> Path:
    """Write results/<mode>/charts/<id>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{spec_['id']}.json"
    p.write_text(json.dumps(spec_, indent=1, default=str))
    return p


def _colors(spec_: dict[str, Any]) -> list[str]:
    if spec_["x"] and all(v in GRADE_COLOR for v in spec_["x"]):
        return [GRADE_COLOR[v] for v in spec_["x"]]
    return [PALETTE[i % len(PALETTE)] for i in range(len(spec_["series"]))]


def to_png(spec_: dict[str, Any], path: Path) -> Path:
    """Render one spec to a 300dpi PNG (matplotlib, colourblind-safe palette)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    kind, colors = spec_["kind"], _colors(spec_)
    if kind == "bar":
        s = spec_["series"][0]
        bars = colors if len(colors) == len(spec_["x"]) else [PALETTE[0]] * len(spec_["x"])
        ax.bar([str(v) for v in spec_["x"]], s["y"], color=bars)
    elif kind in ("line", "scatter"):
        plot = ax.plot if kind == "line" else ax.scatter
        for i, s in enumerate(spec_["series"]):
            plot(spec_["x"], s["y"], label=s["name"], color=PALETTE[i % len(PALETTE)])
        if len(spec_["series"]) > 1:
            ax.legend()
    elif kind == "hist":
        ax.hist(spec_["series"][0]["y"], bins=30, color=PALETTE[0])
    elif kind == "table":
        ax.axis("off")
        rows = [[str(v)] + [str(s["y"][i]) for s in spec_["series"]]
               for i, v in enumerate(spec_["x"])]
        ax.table(cellText=rows, colLabels=["x"] + [s["name"] for s in spec_["series"]],
                 loc="center")
    if spec_.get("xscale") == "log":
        ax.set_xscale("log")
    ax.set_xlabel(spec_.get("xlabel", "")); ax.set_ylabel(spec_.get("ylabel", ""))
    ax.set_title(spec_["title"])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def to_chartjs(spec_: dict[str, Any]) -> dict[str, Any]:
    """Spec -> a Chart.js config dict (panel renders this directly)."""
    ctype = {"hist": "bar", "table": "bar"}.get(spec_["kind"], spec_["kind"])
    colors = _colors(spec_)
    datasets = [{"label": s["name"], "data": s["y"],
                "backgroundColor": colors if len(colors) == len(spec_["x"])
                else PALETTE[i % len(PALETTE)]} for i, s in enumerate(spec_["series"])]
    return {"type": ctype, "data": {"labels": [str(v) for v in spec_["x"]], "datasets": datasets},
           "options": {"plugins": {"title": {"display": True, "text": spec_["title"]}},
                       "scales": {"x": {"title": {"display": bool(spec_.get("xlabel")),
                                                  "text": spec_.get("xlabel", "")}},
                                 "y": {"title": {"display": bool(spec_.get("ylabel")),
                                                "text": spec_.get("ylabel", "")}}}}}


def to_tex(spec_: dict[str, Any], path: Path | None = None) -> str:
    """Spec -> booktabs LaTeX table string; if `path` given, also write it there."""
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tex = to_tex(spec_)
        path.write_text(tex)
        return tex
    def q(v: Any) -> str:  # escape LaTeX-active chars in non-numeric cells
        s = str(v)
        for ch in "&%#_": s = s.replace(ch, "\\" + ch)
        return s
    header = " & ".join(["x", *[q(s["name"]) for s in spec_["series"]]]) + r" \\"
    rows = []
    for i, v in enumerate(spec_["x"]):
        cells = [q(v)] + [f"{s['y'][i]:.4g}" if isinstance(s["y"][i], (int, float))
                          else q(s["y"][i]) for s in spec_["series"]]
        rows.append(" & ".join(cells) + r" \\")
    body = "\n".join(rows)
    ncols = "r" * len(spec_["series"])
    return (f"% {spec_['title']} ({spec_['id']})\n"
           f"\\begin{{tabular}}{{l{ncols}}}\n\\toprule\n{header}\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
