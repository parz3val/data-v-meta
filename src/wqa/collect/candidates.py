"""Grade -> candidate mainspace titles (SPEC §5 P1).
FA/GA: direct mainspace categories.
Stub-B: NOT a single flat "<Grade>-Class articles" bucket -- live-checked and that generic
category is near-empty for C/B (WMF's WPBannerMeta rollout categorizes per-WikiProject only,
e.g. "Category:B-Class Aesthetics articles", not into one shared bucket, for C/B; Stub/Start
still have populated flat buckets but for consistency all four grades use the same mechanism).
So: enumerate every "Category:<Grade>-Class <Project> articles" talk category (allpages over
ns=14), seed-shuffle them, round-robin categorymembers across a capped random subset -- an
unbiased sample over the union of WikiProjects within an API-call budget (see
docs/decisions/pipe.md for the max_projects cap rationale).
In: ApiClient, grade/category. Out: generator of mainspace titles."""
from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from wqa.collect.client import ApiClient

MAX_PROJECTS = 80
PER_PROJECT_PAGE = 25


def fa_ga_titles(client: ApiClient, category: str) -> Iterator[str]:
    """Mainspace titles directly in an FA/GA category (ns=0)."""
    cont: dict[str, Any] = {}
    while True:
        data = client.query(list="categorymembers", cmtitle=category, cmnamespace=0,
                            cmlimit=500, **cont)
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"]
        if "continue" not in data:
            return
        cont = data["continue"]


def project_categories(client: ApiClient, grade: str) -> list[str]:
    """All "Category:<Grade>-Class <Project> articles" talk categories (ns=14), excluding
    the class x importance cross-tab subcategories (e.g. "... of Low-importance")."""
    prefix = f"{grade}-Class "
    out: list[str] = []
    cont: dict[str, Any] = {}
    while True:
        data = client.query(list="allpages", apnamespace=14, apprefix=prefix, aplimit=500, **cont)
        for p in data.get("query", {}).get("allpages", []):
            title = p["title"]
            if "-importance" in title:
                continue
            out.append(title)
        if "continue" not in data:
            return out
        cont = {"apcontinue": data["continue"]["apcontinue"]}


def lower_grade_titles(client: ApiClient, grade: str, seed: int = 42) -> Iterator[str]:
    """Mainspace titles for Stub/Start/C/B: round-robin talk-page members across a
    seed-shuffled, capped sample of per-WikiProject class categories, 'Talk:' stripped."""
    projects = project_categories(client, grade)
    rng = random.Random(seed)
    rng.shuffle(projects)
    projects = projects[:MAX_PROJECTS]
    states: dict[str, dict[str, Any]] = {p: {} for p in projects}
    active = list(projects)
    seen: set[str] = set()
    while active:
        nxt = []
        for proj in active:
            data = client.query(list="categorymembers", cmtitle=proj, cmnamespace=1,
                                cmlimit=PER_PROJECT_PAGE, **states[proj])
            for m in data.get("query", {}).get("categorymembers", []):
                title = m["title"]
                if title.startswith("Talk:"):
                    mt = title[len("Talk:"):]
                    if mt not in seen:
                        seen.add(mt)
                        yield mt
            if "continue" in data:
                states[proj] = data["continue"]
                nxt.append(proj)
        active = nxt
