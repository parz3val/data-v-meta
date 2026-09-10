"""Wikitext -> leakage-safe structural fields (SPEC §5 P2). Icon templates that state the
grade ({{featured article}}, {{good article}}) are stripped before any text/template/category
count is derived -- see configs/features.yaml leakage_exclusions and docs/decisions/pipe.md.
In: raw wikitext + collected categories/templates lists. Out: dict of parsed fields."""
from __future__ import annotations

import re
from typing import Any

import mwparserfromhell as mwp

ICON_TEMPLATES = {"featured article", "good article"}
LABEL_CATEGORIES = {"category:featured articles", "category:good articles"}
CLEANUP_KEYWORDS = ("cleanup", "citation needed", "clarify", "dubious", "disputed",
                    "original research", "unreferenced", "weasel", "vague", "update",
                    "expand section", "copyedit", "fact", "pov")

_LINK_RE = re.compile(r"\[\[([^\]|]+)")
_FILE_RE = re.compile(r"^(file|image):", re.IGNORECASE)
_EXTLINK_RE = re.compile(r"\[https?://[^\s\]]+", re.IGNORECASE)
_LIST_LINE_RE = re.compile(r"^[*#:;]+\s*\S", re.MULTILINE)
_SENTENCE_RE = re.compile(r"[.!?]+(?=\s+[A-Z]|\s*$)")
_DOI_RE = re.compile(r"\bdoi\s*=", re.IGNORECASE)
_ISBN_RE = re.compile(r"\bisbn(?:1[03])?\s*=", re.IGNORECASE)


def _norm(name: str) -> str:
    return name.strip().lower().removeprefix("template:")


def parse_article(wikitext: str, categories: list[str], templates: list[str]) -> dict[str, Any]:
    """One article's wikitext + collected lists -> leakage-stripped structural/reference fields."""
    code = mwp.parse(wikitext or "")
    all_templates = [_norm(str(t.name)) for t in code.filter_templates()]
    kept_templates = [t for t in all_templates if t not in ICON_TEMPLATES]
    kept_categories = [c for c in categories if c.strip().lower() not in LABEL_CATEGORIES]

    for tmpl in code.filter_templates():
        if _norm(str(tmpl.name)) in ICON_TEMPLATES:
            try:
                code.remove(tmpl)
            except ValueError:
                pass

    refs_raw = [str(tag.contents) if tag.contents else "" for tag in code.filter_tags()
               if str(tag.tag).lower() == "ref"]
    headings = code.filter_headings()
    l2 = sum(1 for h in headings if h.level == 2)
    l3 = sum(1 for h in headings if h.level == 3)

    clean_text = _to_plain_text(code)
    lead_text = _lead(clean_text)

    links = [m.group(1) for m in _LINK_RE.finditer(wikitext)]
    images_n = sum(1 for link in links if _FILE_RE.match(link.strip()))
    wikilinks_n = sum(1 for link in links if not _FILE_RE.match(link.strip())
                      and not link.strip().lower().startswith("category:"))

    return {
        "clean_text": clean_text, "lead_text": lead_text,
        "word_count": len(clean_text.split()),
        "sentence_count": max(1, len(_SENTENCE_RE.findall(clean_text))) if clean_text else 0,
        "section_l2_n": l2, "section_l3_n": l3,
        "infobox": any(t.startswith("infobox") for t in kept_templates),
        "images_n": images_n, "tables_n": wikitext.count("{|"),
        "wikilinks_n": wikilinks_n, "extlinks_n": len(_EXTLINK_RE.findall(wikitext)),
        "lists_n": len(_LIST_LINE_RE.findall(wikitext)),
        "cleanup_tags_n": sum(1 for t in kept_templates
                              if any(k in t for k in CLEANUP_KEYWORDS)),
        "templates_n": len(kept_templates), "categories_n": len(kept_categories),
        "cite_templates_n": sum(1 for t in kept_templates
                                if t.startswith("cite ") or t == "citation"),
        "refs_raw": refs_raw, "ref_n": len(refs_raw),
        "refs_distinct": len(set(r.strip() for r in refs_raw if r.strip())),
        "doi_n": len(_DOI_RE.findall(wikitext)), "isbn_n": len(_ISBN_RE.findall(wikitext)),
    }


def _to_plain_text(code: mwp.wikicode.Wikicode) -> str:
    for link in code.filter_wikilinks():
        title = str(link.title).strip().lower()
        if title.startswith(("file:", "image:", "category:")):
            try:
                code.remove(link)
            except ValueError:
                pass
    return str(code.strip_code(normalize=True, collapse=True)).strip()


def _lead(clean_text: str, max_chars: int = 500) -> str:
    return clean_text[:max_chars]
