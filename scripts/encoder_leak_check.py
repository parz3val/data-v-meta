"""Inference-only check: does the saved encoder depend on the grade icon templates left in raw wikitext?
Scores the test partition with the checkpoint on (a) raw wikitext as trained, (b) raw wikitext with the
grade templates removed, (c) leakage-stripped plain text from wqa.clean.parse. Writes results/<mode>/encoder/leak_check.json."""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wqa.config import ROOT
from wqa.models import common
from wqa.models.encoder import load_texts, encode
from wqa.clean.parse import parse_article
from wqa.eval.metrics import macro_f1
from transformers import AutoTokenizer, AutoModelForSequenceClassification

mode = sys.argv[1] if len(sys.argv) > 1 else "full"
tag = "rung3_distilroberta_base_headtail"
GRADE_TPL = re.compile(r"\{\{\s*(featured (article|list)|good article|ga icon|fa icon)\s*(\|[^}]*)?\}\}", re.I)
df = common.load_table(mode); te = df[df["split"] == "test"]
texts = load_texts(mode)
tok = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=6)
model.load_state_dict(torch.load(ROOT / "results" / mode / "models" / f"{tag}_s42.pt", map_location="cpu"))
dev = "mps" if torch.backends.mps.is_available() else "cpu"; model.to(dev).eval()
y = te["label"].to_numpy(int); pids = te["pageid"].to_numpy(int)
raw = [texts.get(int(p), "") for p in pids]
pos = [m.start() for t in raw for m in [GRADE_TPL.search(t)] if m]
variants = {"raw_as_trained": raw, "raw_minus_grade_templates": [GRADE_TPL.sub("", t) for t in raw],
            "clean_plain_text": [parse_article(t, [], [])["clean_text"] for t in raw]}
out = {"n_test": int(len(y)), "n_with_grade_template": len(pos), "template_char_pos_median": float(np.median(pos)) if pos else None}
with torch.no_grad():
    for name, tx in variants.items():
        enc = encode(tok, tx, 512, True); preds = []
        t0 = time.time()
        for i in range(0, len(tx), 16):
            b = {k: v[i:i + 16].to(dev) for k, v in enc.items()}
            preds.append(model(**b).logits.argmax(1).cpu().numpy())
        p = np.concatenate(preds); f = macro_f1(y, p)
        per = [float(np.mean(p[y == k] == k)) for k in range(6)]
        out[name] = {"macro_f1": round(float(f), 4), "acc": round(float((p == y).mean()), 4), "recall_per_grade": [round(v, 3) for v in per], "secs": round(time.time() - t0, 1)}
        print(name, out[name], flush=True)
(ROOT / "results" / mode / "encoder" / "leak_check.json").write_text(json.dumps(out, indent=1))
