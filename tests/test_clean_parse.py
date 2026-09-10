"""Unit tests for wqa.clean.parse -- synthetic wikitext, no network (WP4)."""
from __future__ import annotations

from wqa.clean import parse

WIKITEXT = """{{Featured article}}
{{Infobox person
| name = Test
}}
'''Test''' is a person. This is the lead. It has two sentences.

==History==
Some history text with a [[Wikipedia:Link|link]] and a [[File:Pic.jpg|thumb]] and
an external link [http://example.com Example] and a list:
* item one
* item two

{{Citation needed}}

{| class="wikitable"
| a || b
|}

Text with a ref<ref>{{cite web |url=http://x |doi=10.1/x}}</ref> and another<ref>plain ref</ref>.
ISBN reference isbn=978-0-00-000000-0.

===Subsection===
More text here.

[[Category:Test category]]
[[Category:Featured articles]]
"""

CATEGORIES = ["Category:Test category", "Category:Featured articles"]
TEMPLATES = ["Template:Infobox person", "Template:Citation needed", "Template:Featured article",
            "Template:Cite web"]


def test_icon_template_stripped_from_clean_text() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert "featured article" not in out["clean_text"].lower()
    assert "Featured article" not in out["clean_text"]


def test_label_category_excluded_from_count() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["categories_n"] == 1  # "Category:Featured articles" excluded


def test_icon_template_excluded_from_template_count() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    # 4 templates collected minus the icon template = 3
    assert out["templates_n"] == 3


def test_headings_counted_by_level() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["section_l2_n"] == 1
    assert out["section_l3_n"] == 1


def test_infobox_detected() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["infobox"] is True


def test_refs_extracted() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["ref_n"] == 2
    assert out["refs_distinct"] == 2


def test_images_wikilinks_extlinks_counted() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["images_n"] == 1
    assert out["wikilinks_n"] == 1
    assert out["extlinks_n"] == 1


def test_lists_and_tables_counted() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["lists_n"] == 2
    assert out["tables_n"] == 1


def test_cleanup_tag_counted() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["cleanup_tags_n"] == 1


def test_doi_and_isbn_counted() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["doi_n"] == 1
    assert out["isbn_n"] == 1


def test_cite_templates_counted() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["cite_templates_n"] == 1


def test_empty_wikitext_does_not_crash() -> None:
    out = parse.parse_article("", [], [])
    assert out["word_count"] == 0
    assert out["ref_n"] == 0


def test_lead_is_prefix_of_clean_text() -> None:
    out = parse.parse_article(WIKITEXT, CATEGORIES, TEMPLATES)
    assert out["clean_text"].startswith(out["lead_text"][:50])
