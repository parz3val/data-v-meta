"""WP1 skeleton: manifest idempotency (SPEC §10 required test)."""
from pathlib import Path

from wqa import manifest


def test_update_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(manifest, "ROOT", tmp_path)
    f = tmp_path / "data" / "small" / "raw" / "x.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("{}\n")
    for _ in range(2):
        manifest.update_dataset("small", name="raw", stage="raw", file=f, rows=1, cols=3,
                                producer_target="data")
    m = manifest.load("small")
    assert len(m["datasets"]) == 1
    assert m["datasets"][0]["rows"] == 1 and m["datasets"][0]["sha256"]
