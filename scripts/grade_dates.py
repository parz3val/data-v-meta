"""Recover grade-assessment dates from talk-page history (claim 2 validation upgrade).
For each val+test article: walk Talk: page revision timestamps (ids+timestamp only, cheap),
bisect content fetches to find when the class= parameter last became the current grade.
Resumable via results/<mode>/staleness/grade_dates.json. Sequential, polite (collector client).
Usage: python scripts/grade_dates.py --mode small [--limit N]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wqa import config  # noqa: E402
from wqa.collect.client import ApiClient  # noqa: E402
from wqa.models import common  # noqa: E402

CLASS_RE = re.compile(r"\bclass\s*=\s*([A-Za-z-]+)", re.I)
CANON = {"stub": "Stub", "start": "Start", "c": "C", "b": "B", "ga": "GA", "fa": "FA"}


def classes_in(text: str) -> set[str]:
    return {CANON[m.lower()] for m in CLASS_RE.findall(text or "") if m.lower() in CANON}


def rev_content(client, revid: int) -> str:
    r = client.query(prop="revisions", revids=revid, rvprop="content", rvslots="main")
    pages = r.get("query", {}).get("pages", {})
    for p in pages.values():
        for rev in p.get("revisions", []):
            return rev.get("slots", {}).get("main", {}).get("*", "") or ""
    return ""


def find_grade_date(client, title: str, grade: str) -> tuple[str | None, int]:
    """Earliest revision of the current contiguous run where talk page carries `grade`."""
    talk = "Talk:" + title
    revs = []
    cont = {}
    while True:
        r = client.query(prop="revisions", titles=talk, rvprop="ids|timestamp",
                         rvlimit=500, rvdir="older", **cont)
        pages = r.get("query", {}).get("pages", {})
        for p in pages.values():
            revs.extend(p.get("revisions", []))
        cont = r.get("continue", {})
        if not cont or len(revs) >= 2000:
            break
    if not revs:
        return None, 0
    n_fetch = 0
    # revs[0] = newest. Confirm newest has the grade; then bisect for the boundary of
    # the newest contiguous run (grade present in [0..k], absent at k+1).
    newest = rev_content(client, revs[0]["revid"])
    n_fetch += 1
    if grade not in classes_in(newest):
        return None, n_fetch
    lo, hi = 0, len(revs) - 1  # invariant: grade at lo; unknown at hi
    if grade in classes_in(rev_content(client, revs[hi]["revid"])):
        return revs[hi]["timestamp"], n_fetch + 1  # graded since first snapshot we see
    n_fetch += 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        has = grade in classes_in(rev_content(client, revs[mid]["revid"]))
        n_fetch += 1
        if has:
            lo = mid
        else:
            hi = mid
    return revs[lo]["timestamp"], n_fetch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="small")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_p = ROOT / "results" / args.mode / "staleness" / "grade_dates.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_p.exists():
        done = json.loads(out_p.read_text())

    df = common.load_table(args.mode)
    part = df[df["split"].isin(["val", "test"])]
    # titles come from raw jsonl (features table may not carry title)
    titles = {}
    for g in ("Stub", "Start", "C", "B", "GA", "FA"):
        p = ROOT / "data" / args.mode / "raw" / f"{g}.jsonl"
        if p.exists():
            with p.open() as f:
                for line in f:
                    d = json.loads(line)
                    titles[int(d["pageid"])] = (d["title"], d["label"])

    client = ApiClient(config.load("collection"))
    todo = [int(p) for p in part["pageid"] if str(int(p)) not in done and int(p) in titles]
    if args.limit:
        todo = todo[:args.limit]
    print(f"grade_dates: {len(done)} done, {len(todo)} to go")
    total_fetch = 0
    for i, pid in enumerate(todo):
        title, grade = titles[pid]
        try:
            ts, nf = find_grade_date(client, title, grade)
        except Exception as e:
            ts, nf = None, 0
            print(f"ERR {pid} {title[:40]}: {e}")
        total_fetch += nf
        done[str(pid)] = {"title": title, "grade": grade, "grade_ts": ts}
        if i % 20 == 0 or i == len(todo) - 1:
            out_p.write_text(json.dumps(done, indent=0))
            print(f"[{i+1}/{len(todo)}] fetches={total_fetch} last={title[:36]} ts={ts}")
        if (ROOT / "data" / "STOP").exists():
            print("STOP honoured")
            break
    out_p.write_text(json.dumps(done, indent=0))
    ok = sum(1 for v in done.values() if v["grade_ts"])
    print(f"grade_dates: {ok}/{len(done)} recovered")


if __name__ == "__main__":
    main()
