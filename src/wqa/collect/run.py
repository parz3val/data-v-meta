"""Orchestrate WP3 collection end to end (SPEC §5 P1, §4 manifest, §4.3 API rules).
Usage: from wqa.collect.run import run; run("small", smoke=False)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from wqa import config, manifest
from wqa.collect import assess, candidates, enrich, persist
from wqa.collect.client import ApiClient
from wqa.monitor.budget import budget

log = structlog.get_logger()
GRADES = ["Stub", "Start", "C", "B", "GA", "FA"]
RECORD_FIELDS = 22


def run(mode: str, smoke: bool = False) -> dict[str, Any]:
    cfg = config.load("collection")
    client = ApiClient(cfg)
    target = _target(cfg, mode, smoke)
    excluded: dict[str, int] = {}
    per_grade: dict[str, int] = {}
    t0 = time.time()
    with budget("P1_collect_full"):
        stop_p = config.ROOT / "data" / "STOP"
        if stop_p.exists():
            stop_p.unlink()
        for g in GRADES:  # counts so far (resume) so progress.json is complete from the first batch
            per_grade[g] = len(persist.existing_pageids(mode, g))
        prog = {"mode": mode, "target": target, "per_grade": per_grade, "excluded": excluded,
                "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_start": sum(per_grade.values()), "log": f"status/jobs/data_{mode}.log", "state": "running"}
        try:
            for grade in GRADES:
                prog["current_grade"] = grade
                per_grade[grade] = collect_grade(client, cfg, mode, grade, target, excluded, prog)
                log.info("grade_done", grade=grade, n=per_grade[grade], target=target)
        except _StopRequested:
            prog["state"] = "stopped"
            _write_progress(mode, prog)
            log.info("collect_stopped", per_grade=per_grade)
            return {"per_grade": per_grade, "excluded": excluded, "stopped": True}
        prog["state"] = "done"
        _write_progress(mode, prog)
    _finalize_manifest(mode, target, per_grade, excluded)
    counts = _validate(mode)
    log.info("collect_done", secs=round(time.time() - t0, 1), per_grade=per_grade,
             excluded=excluded, **counts)
    return {"per_grade": per_grade, "excluded": excluded, **counts}


class _StopRequested(Exception):
    """data/STOP seen between batches (make stop): index saved, resume later without refetch."""


def _write_progress(mode: str, prog: dict[str, Any]) -> None:
    """data/<mode>/raw/progress.json: live per-grade counts, rate, ETA (panel Data tab)."""
    now = datetime.now(timezone.utc)
    done = sum(prog["per_grade"].values())
    mins = max((now - datetime.fromisoformat(prog["started"])).total_seconds() / 60, 1e-6)
    rate = (done - prog["n_start"]) / mins
    remaining = max(prog["target"] * len(GRADES) - done, 0)
    prog.update(updated=now.isoformat(timespec="seconds"), rate_per_min=round(rate, 1),
                eta_min=round(remaining / rate) if rate > 0 else None, done=done, total=prog["target"] * len(GRADES))
    (persist.raw_dir(mode) / "progress.json").write_text(json.dumps(prog, indent=1))


def collect_grade(client: ApiClient, cfg: dict[str, Any], mode: str, grade: str, target: int,
                  excluded: dict[str, int], prog: dict[str, Any] | None = None) -> int:
    """Fill data/<mode>/raw/<grade>.jsonl up to `target` rows; resumable, no refetch."""
    n = len(persist.existing_pageids(mode, grade))
    index = persist.load_index(mode)
    # resume guard: index.json can lag the jsonl by <1 batch after a crash; never re-append a pageid
    for g in GRADES:
        for pid in persist.existing_pageids(mode, g):
            index.setdefault(str(pid), {"file": f"raw/{g}.jsonl", "offset": -1, "length": 0})
    if n >= target:
        log.info("grade_already_full", grade=grade, n=n)
        return n
    sources = cfg["sources"]
    cand_iter = (candidates.fa_ga_titles(client, sources["fa_category"]) if grade == "FA" else
                candidates.fa_ga_titles(client, sources["ga_category"]) if grade == "GA" else
                candidates.lower_grade_titles(client, grade))
    batch: list[str] = []
    batch_last = ""
    for title in cand_iter:
        batch.append(title)
        batch_last = title
        if len(batch) < cfg["batch_titles"]:
            continue
        n = _process_batch(client, cfg, mode, grade, batch, target, n, index, excluded)
        batch = []
        if prog is not None:
            prog["per_grade"][grade] = n
            prog["last_title"] = batch_last
            _write_progress(mode, prog)
            log.info("batch", grade=grade, n=n, target=target, rate_per_min=prog.get("rate_per_min"))
            if (config.ROOT / "data" / "STOP").exists():
                persist.save_index(mode, index)
                raise _StopRequested()
        if n >= target:
            break
    if batch and n < target:
        n = _process_batch(client, cfg, mode, grade, batch, target, n, index, excluded)
    persist.save_index(mode, index)
    return n


def _process_batch(client: ApiClient, cfg: dict[str, Any], mode: str, grade: str,
                   titles: list[str], target: int, n: int, index: dict[str, Any],
                   excluded: dict[str, int]) -> int:
    assessed = assess.batch_assessments(client, titles, fallback_label=grade)
    wanted = [t for t in titles if assessed.get(t, {}).get("label") == grade]
    if not wanted:
        return n
    core = enrich.batch_core(client, wanted)
    fetch_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for title in wanted:
        if n >= target:
            break
        c = core.get(title)
        if not c or str(c["pageid"]) in index:
            continue
        reason = _excluded_cheap(title, cfg, c, grade)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        lists = enrich.lists_for(client, title)
        list_cat = cfg["exclude"].get("list_category")
        if list_cat and list_cat in lists["categories"]:
            excluded["list_category"] = excluded.get("list_category", 0) + 1
            continue
        hist = enrich.history_counts(client, title, cfg["pageviews_days"])
        views = enrich.pageviews_90d(client, title, cfg["pageviews_days"])
        rec = _record(title, grade, assessed[title], c, lists, hist, views, fetch_ts)
        persist.append(mode, grade, rec, index)
        n += 1
    persist.save_index(mode, index)
    return n


def _excluded_cheap(title: str, cfg: dict[str, Any], core: dict[str, Any], grade: str) -> str | None:
    """Exclusion checks needing only the batched core fields (no extra API call). The
    list_category check needs the per-article categories fetch and is done by the caller."""
    ex = cfg["exclude"]
    if core.get("is_redirect"):
        return "redirects"
    if core.get("is_disambig"):
        return "disambigs"
    if ex.get("list_prefix") and title.startswith(ex["list_prefix"]):
        return "list_prefix"
    if grade != "Stub" and core.get("bytes", 0) < ex.get("min_bytes_non_stub", 0):
        return "min_bytes_non_stub"
    return None


def _record(title: str, grade: str, a: dict[str, Any], core: dict[str, Any],
           lists: dict[str, list[str]], hist: dict[str, Any], views: float,
           fetch_ts: str) -> dict[str, Any]:
    return {
        "pageid": a["pageid"], "title": title, "project_classes": a["pairs"], "label": grade,
        "n_projects": a["n_projects"], "n_disagree": a["n_disagree"], "revid": core["revid"],
        "wikitext": core["wikitext"], "bytes": core["bytes"], "protection": core["protection"],
        "categories": lists["categories"], "templates": lists["templates"],
        "pageviews_90d_mean": views, "edits": hist["edits"], "editors": hist["editors"],
        "anon_n": hist["anon_n"], "bot_n": hist["bot_n"], "reverted_n": hist["reverted_n"],
        "edits_90d": hist["edits_90d"], "created": hist["created"] or core["last_edit"],
        "last_edit": core["last_edit"], "fetch_ts": fetch_ts,
    }


def _target(cfg: dict[str, Any], mode: str, smoke: bool) -> int:
    if smoke:
        return int(cfg["smoke"]["per_grade"])
    return int(cfg[mode]["per_grade"])


def _finalize_manifest(mode: str, target: int, per_grade: dict[str, int],
                       excluded: dict[str, int]) -> None:
    m = manifest.load(mode)
    m["target_per_grade"] = target
    m["per_grade"] = per_grade
    m["excluded"] = excluded
    manifest.save(mode, m)
    idx_path = persist.raw_dir(mode) / "index.json"
    if idx_path.exists():
        manifest.update_dataset(mode, name="raw", stage="raw", file=idx_path,
                                rows=sum(per_grade.values()), cols=RECORD_FIELDS,
                                producer_target="data")


def _validate(mode: str) -> dict[str, Any]:
    """Counts, dup pageids, missing<1% across required fields (SPEC §5 P1)."""
    seen: set[int] = set()
    dups = missing = total = 0
    required = ("pageid", "title", "label", "revid", "wikitext", "bytes")
    for grade in GRADES:
        p = persist.raw_dir(mode) / f"{grade}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            total += 1
            if rec["pageid"] in seen:
                dups += 1
            seen.add(rec["pageid"])
            if any(rec.get(k) in (None, "") for k in required):
                missing += 1
    rate = missing / total if total else 0.0
    if dups:
        log.warning("dup_pageids", n=dups)
    if rate >= 0.01:
        log.warning("missing_rate_high", rate=round(rate, 4))
    return {"total": total, "dups": dups, "missing": missing, "missing_rate": round(rate, 4)}
