"""Offline, fixture-driven tests for wqa.collect (WP3). No network: FakeClient returns
canned MediaWiki-shaped JSON so parsing/resolution logic is verified without hitting the API."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from wqa.collect import assess, candidates, enrich, persist
from wqa.collect.run import _excluded_cheap, _record


class FakeClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def query(self, **params: Any) -> dict[str, Any]:
        self.calls.append(params)
        return self.responses.pop(0)

    def pageviews(self, title_safe: str, start: str, end: str) -> dict[str, Any]:
        self.calls.append({"title": title_safe, "start": start, "end": end})
        return self.responses.pop(0)


def test_fa_ga_titles_paginates() -> None:
    c = FakeClient([
        {"query": {"categorymembers": [{"title": "A"}, {"title": "B"}]},
         "continue": {"cmcontinue": "x"}},
        {"query": {"categorymembers": [{"title": "C"}]}},
    ])
    assert list(candidates.fa_ga_titles(c, "Category:Featured articles")) == ["A", "B", "C"]


def test_project_categories_excludes_importance_crosstabs() -> None:
    c = FakeClient([{"query": {"allpages": [
        {"title": "Category:B-Class Dogs articles"},
        {"title": "Category:B-Class Dogs articles of Low-importance"}]}}])
    assert candidates.project_categories(c, "B") == ["Category:B-Class Dogs articles"]


def test_lower_grade_titles_round_robins_projects(monkeypatch) -> None:
    monkeypatch.setattr(candidates, "MAX_PROJECTS", 2)
    c = FakeClient([
        {"query": {"allpages": [{"title": "Category:B-Class Dogs articles"},
                                {"title": "Category:B-Class Cats articles"}]}},
        {"query": {"categorymembers": [{"title": "Talk:Rex"}]}},   # Dogs (order depends on shuffle)
        {"query": {"categorymembers": [{"title": "Talk:Tom"}]}},   # Cats
    ])
    out = sorted(candidates.lower_grade_titles(c, "B", seed=1))
    assert out == ["Rex", "Tom"]


def test_batch_assessments_resolves_highest_and_disagree() -> None:
    c = FakeClient([{"query": {"pages": {"1": {"pageid": 1, "title": "X", "pageassessments": {
        "P1": {"class": "B"}, "P2": {"class": "GA"}, "P3": {"class": "B"}}}}}}])
    out = assess.batch_assessments(c, ["X"])
    assert out["X"]["label"] == "GA"
    assert out["X"]["n_projects"] == 3
    assert out["X"]["n_disagree"] == 2


def test_batch_assessments_fallback_when_no_banner() -> None:
    c = FakeClient([{"query": {"pages": {"1": {"pageid": 1, "title": "Y"}}}}])
    out = assess.batch_assessments(c, ["Y"], fallback_label="FA")
    assert out["Y"] == {"pageid": 1, "pairs": [], "label": "FA", "n_projects": 0, "n_disagree": 0}


def test_batch_assessments_missing_page_skipped() -> None:
    c = FakeClient([{"query": {"pages": {"1": {"title": "Z", "missing": ""}}}}])
    assert assess.batch_assessments(c, ["Z"], fallback_label="FA") == {}


def test_batch_core_extracts_fields() -> None:
    page = {"pageid": 5, "title": "Dog", "length": 12345,
            "revisions": [{"revid": 999, "timestamp": "2024-01-01T00:00:00Z",
                           "slots": {"main": {"*": "wikitext here"}}}],
            "protection": [{"type": "edit", "level": "autoconfirmed"}],
            "pageprops": {"disambiguation": ""}}
    c = FakeClient([{"query": {"pages": {"5": page}}}])
    out = enrich.batch_core(c, ["Dog"])
    r = out["Dog"]
    assert r["revid"] == 999 and r["bytes"] == 12345 and r["wikitext"] == "wikitext here"
    assert r["protection"] == ["edit:autoconfirmed"]
    assert r["is_disambig"] is True and r["is_redirect"] is False


def test_batch_core_skips_missing_pages() -> None:
    c = FakeClient([{"query": {"pages": {"1": {"title": "Gone", "missing": ""}}}}])
    assert enrich.batch_core(c, ["Gone"]) == {}


def test_lists_for_single_title_not_truncated() -> None:
    page = {"pageid": 5, "title": "Dog", "categories": [{"title": "Category:Mammals"}],
            "templates": [{"title": "Template:Infobox"}]}
    c = FakeClient([{"query": {"pages": {"5": page}}}])
    out = enrich.lists_for(c, "Dog")
    assert out == {"categories": ["Category:Mammals"], "templates": ["Template:Infobox"]}


def test_history_counts_tallies_anon_bot_reverted_recent() -> None:
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    old = (now - timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%SZ")
    revs = [
        {"timestamp": recent, "user": "Alice", "tags": []},
        {"timestamp": recent, "user": "1.2.3.4", "anon": "", "tags": []},
        {"timestamp": old, "user": "ExampleBot", "tags": []},
        {"timestamp": old, "user": "Bob", "tags": ["mw-rollback"]},
    ]
    c = FakeClient([{"query": {"pages": {"1": {"revisions": revs}}}}])
    out = enrich.history_counts(c, "Dog")
    assert out["edits"] == 4 and out["editors"] == 4
    assert out["anon_n"] == 1 and out["bot_n"] == 1 and out["reverted_n"] == 1
    assert out["edits_90d"] == 2
    assert out["created"] == old


def test_history_counts_paginates_up_to_cap() -> None:
    page1 = [{"timestamp": "2024-01-01T00:00:00Z", "user": f"u{i}", "tags": []} for i in range(500)]
    page2 = [{"timestamp": "2023-01-01T00:00:00Z", "user": "u500", "tags": []}]
    c = FakeClient([
        {"query": {"pages": {"1": {"revisions": page1}}}, "continue": {"rvcontinue": "x"}},
        {"query": {"pages": {"1": {"revisions": page2}}}},
    ])
    out = enrich.history_counts(c, "Dog")
    assert out["edits"] == 500  # capped, second page not consumed because cap already hit
    assert len(c.calls) == 1


def test_pageviews_90d_averages() -> None:
    c = FakeClient([{"items": [{"views": 10}, {"views": 20}, {"views": 30}]}])
    assert enrich.pageviews_90d(c, "Dog") == 20.0


def test_pageviews_90d_empty_returns_zero() -> None:
    c = FakeClient([{}])
    assert enrich.pageviews_90d(c, "Dog") == 0.0


def test_excluded_cheap_reasons() -> None:
    cfg = {"exclude": {"list_prefix": "List of", "list_category": "Category:Lists",
                       "min_bytes_non_stub": 100}}
    core_redirect = {"is_redirect": True}
    core_disambig = {"is_redirect": False, "is_disambig": True}
    core_short = {"is_redirect": False, "is_disambig": False, "bytes": 10}
    core_ok = {"is_redirect": False, "is_disambig": False, "bytes": 500}
    assert _excluded_cheap("Dog", cfg, core_redirect, "C") == "redirects"
    assert _excluded_cheap("Dog", cfg, core_disambig, "C") == "disambigs"
    assert _excluded_cheap("List of dogs", cfg, core_ok, "C") == "list_prefix"
    assert _excluded_cheap("Dog", cfg, core_short, "C") == "min_bytes_non_stub"
    assert _excluded_cheap("Dog", cfg, core_short, "Stub") is None  # stub exempt from min_bytes
    assert _excluded_cheap("Dog", cfg, core_ok, "C") is None


def test_record_shape_has_all_spec_fields() -> None:
    a = {"pageid": 1, "pairs": [("P1", "B")], "label": "B", "n_projects": 1, "n_disagree": 0}
    core = {"revid": 2, "wikitext": "wt", "bytes": 100, "protection": [],
            "last_edit": "2024-01-01T00:00:00Z"}
    lists = {"categories": [], "templates": []}
    hist = {"edits": 1, "editors": 1, "anon_n": 0, "bot_n": 0, "reverted_n": 0, "edits_90d": 1,
            "created": "2020-01-01T00:00:00Z"}
    rec = _record("Dog", "B", a, core, lists, hist, 5.0, "2024-06-01T00:00:00Z")
    expected = {"pageid", "title", "project_classes", "label", "n_projects", "n_disagree",
               "revid", "wikitext", "bytes", "protection", "categories", "templates",
               "pageviews_90d_mean", "edits", "editors", "anon_n", "bot_n", "reverted_n",
               "edits_90d", "created", "last_edit", "fetch_ts"}
    assert set(rec) == expected


def test_persist_roundtrip_and_resume(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(persist, "ROOT", tmp_path)
    index: dict[str, Any] = {}
    rec = {"pageid": 42, "title": "Dog", "label": "B"}
    persist.append("small", "B", rec, index)
    assert index["42"]["file"] == "raw/B.jsonl"
    assert persist.existing_pageids("small", "B") == {42}
    persist.save_index("small", index)
    reloaded = persist.load_index("small")
    assert reloaded == index
    p = tmp_path / "data" / "small" / "raw" / "B.jsonl"
    line = p.read_text().splitlines()[0]
    assert json.loads(line) == rec
