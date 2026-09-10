"""The text encoder runs in its own process: LightGBM/CatBoost and torch each bring an OpenMP runtime and deadlock
when loaded into one interpreter on macOS. The worker receives plain text over a pipe and returns six probabilities."""
from __future__ import annotations
import sys, time
from pathlib import Path


def serve(conn, ckpt: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import numpy as np, torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from wqa.models.encoder import encode
    tok = AutoTokenizer.from_pretrained("distilroberta-base")
    m = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=6)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; m.to(dev).eval()
    conn.send(("ready", dev))
    while True:
        text = conn.recv()
        if text is None:
            break
        t0 = time.perf_counter(); enc = encode(tok, [text], 512, True)
        with torch.no_grad():
            logits = m(**{k: v.to(dev) for k, v in enc.items()}).logits.float().cpu().numpy()[0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        conn.send((p.tolist(), (time.perf_counter() - t0) * 1000, dev))
