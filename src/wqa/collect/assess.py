"""Resolve (project,class) pairs -> label (configs/collection.yaml label_rule=highest_grade_wins).
In: ApiClient, <=50 mainspace titles, fallback_label for pages with no WikiProject banner.
Out: title -> {pageid, pairs, label, n_projects, n_disagree}."""
from __future__ import annotations

from typing import Any

from wqa.collect.client import ApiClient

RANK = {"Stub": 0, "Start": 1, "C": 2, "B": 3, "GA": 4, "FA": 5}


def batch_assessments(client: ApiClient, titles: list[str],
                      fallback_label: str | None = None) -> dict[str, dict[str, Any]]:
    """<=50 titles -> resolved label per SPEC label_rule; n_disagree = ratings != label."""
    data = client.query(prop="pageassessments", titles="|".join(titles))
    out: dict[str, dict[str, Any]] = {}
    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title")
        if not title or "missing" in page:
            continue
        pairs = [(proj, (info.get("class") or "").strip())
                for proj, info in (page.get("pageassessments") or {}).items()
                if (info.get("class") or "").strip() in RANK]
        if not pairs:
            if fallback_label is None:
                continue
            out[title] = {"pageid": page["pageid"], "pairs": [], "label": fallback_label,
                          "n_projects": 0, "n_disagree": 0}
            continue
        label = max((c for _, c in pairs), key=lambda c: RANK[c])
        n_disagree = sum(1 for _, c in pairs if c != label)
        out[title] = {"pageid": page["pageid"], "pairs": pairs, "label": label,
                      "n_projects": len(pairs), "n_disagree": n_disagree}
    return out
