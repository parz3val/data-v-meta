"""Per-article field enrichment beyond (project,class) assessment (SPEC §5 P1 record fields).
batch_core: one API call per <=50 titles (revisions+info+pageprops only -- categories/templates
are deliberately NOT batched: MediaWiki truncates list-type props like cl/tl to only the first
few pages of a multi-title request and silently returns empty lists for the rest, so those are
fetched per-article via lists_for(), which is call-continuation-safe for a single title.
history_counts: one call per article, capped at HISTORY_CAP most-recent revisions (documented
budget cap in docs/decisions/pipe.md, not full history for heavily-edited pages).
pageviews_90d: one REST call per article."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from wqa.collect.client import ApiClient

BOT_RE = re.compile(r"bot\b", re.IGNORECASE)
HISTORY_CAP = 500


def batch_core(client: ApiClient, titles: list[str]) -> dict[str, dict[str, Any]]:
    """<=50 titles -> latest revision content/id/ts, length, protection, disambiguation/redirect
    flags. Categories/templates are fetched separately per-article, see module docstring."""
    data = client.query(prop="revisions|info|pageprops", titles="|".join(titles),
                        rvprop="ids|timestamp|content", rvslots="main", inprop="protection",
                        ppprop="disambiguation")
    out: dict[str, dict[str, Any]] = {}
    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title")
        if not title or "missing" in page:
            continue
        rev = (page.get("revisions") or [{}])[0]
        slot = rev.get("slots", {}).get("main", {})
        out[title] = {
            "pageid": page["pageid"],
            "revid": rev.get("revid"),
            "wikitext": slot.get("*", slot.get("content", "")),
            "bytes": page.get("length", 0),
            "last_edit": rev.get("timestamp"),
            "protection": [f"{p['type']}:{p['level']}" for p in page.get("protection", [])],
            "is_redirect": "redirect" in page,
            "is_disambig": "disambiguation" in page.get("pageprops", {}),
        }
    return out


def lists_for(client: ApiClient, title: str) -> dict[str, list[str]]:
    """Full categories + templates for one article (cllimit/tllimit=max; single-title requests
    do not suffer the multi-title truncation described in the module docstring)."""
    data = client.query(prop="categories|templates", titles=title, cllimit="max", tllimit="max")
    page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
    return {"categories": [c["title"] for c in page.get("categories", [])],
           "templates": [t["title"] for t in page.get("templates", [])]}


def history_counts(client: ApiClient, title: str, days_recent: int = 90) -> dict[str, Any]:
    """Edits/editors/anon/bot/reverted over the most recent HISTORY_CAP revisions, plus
    edits in the last `days_recent` days and the earliest timestamp seen (approx. created
    for pages under the cap)."""
    users: set[str] = set()
    anon = bot = reverted = edits = edits_recent = 0
    created: str | None = None
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_recent)
    cont: dict[str, Any] = {}
    while edits < HISTORY_CAP:
        data = client.query(prop="revisions", titles=title, rvprop="timestamp|user|anon|tags",
                            rvlimit=min(500, HISTORY_CAP - edits), rvdir="older", **cont)
        page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
        revs = page.get("revisions", [])
        if not revs:
            break
        for r in revs:
            edits += 1
            user = r.get("user", "")
            users.add(user)
            if "anon" in r:
                anon += 1
            elif BOT_RE.search(user):
                bot += 1
            if set(r.get("tags", [])) & {"mw-rollback", "mw-undo", "mw-manual-revert"}:
                reverted += 1
            ts = r.get("timestamp")
            if ts:
                created = ts
                if _parse_ts(ts) >= cutoff:
                    edits_recent += 1
        if "continue" not in data:
            break
        cont = data["continue"]
    return {"edits": edits, "editors": len(users), "anon_n": anon, "bot_n": bot,
            "reverted_n": reverted, "edits_90d": edits_recent, "created": created}


def pageviews_90d(client: ApiClient, title: str, days: int = 90) -> float:
    """Mean daily pageviews over the trailing `days` days, ending yesterday (avoids partial-day)."""
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    safe = title.replace(" ", "_")
    data = client.pageviews(safe, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    items = data.get("items", [])
    if not items:
        return 0.0
    return sum(i["views"] for i in items) / len(items)


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
