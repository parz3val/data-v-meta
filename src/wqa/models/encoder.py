"""Rung 3 text arm: fine-tuned encoder on wikitext (MPS/CUDA/CPU auto).
Clears the WP5 "NOT RUN (no GPU)" disclosure using Apple-silicon MPS (owner big-task 2026-08-25).
Usage: python -m wqa.models.encoder --mode small --model distilroberta-base --epochs 3
       [--seed 42] [--headtail] [--max-len 512] [--tag rung3_distilroberta]
Outputs: live/<tag>_s<seed>.jsonl per epoch; encoder/eval_rows.json row (merged by wqa.eval);
         encoder/logits_<tag>_s<seed>.parquet (train/val/test logits -> text_plus_H arm);
         models/<tag>_s<seed>.pt best checkpoint. Honours data/STOP per epoch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from wqa.config import ROOT
from wqa.eval.metrics import all_metrics, bootstrap_ci, macro_f1, qwk
from wqa.models import common

log = structlog.get_logger()
GRADES = ["Stub", "Start", "C", "B", "GA", "FA"]


def load_texts(mode: str, clean: bool = False) -> dict[int, str]:
    """Raw wikitext per pageid; with clean=True the leakage-stripped plain text of wqa.clean.parse
    (grade icon/assessment templates removed, markup reduced to prose), as the metadata arm's cleaning does."""
    from wqa.clean.parse import parse_article
    texts: dict[int, str] = {}
    for g in GRADES:
        p = ROOT / "data" / mode / "raw" / f"{g}.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                d = json.loads(line)
                raw = d.get("wikitext") or ""
                texts[int(d["pageid"])] = parse_article(raw, d.get("categories", []), d.get("templates", []))["clean_text"] if clean else raw
    return texts


def encode(tok, texts: list[str], max_len: int, headtail: bool):
    if not headtail:
        return tok(texts, truncation=True, max_length=max_len, padding="max_length",
                   return_tensors="pt")
    # head+tail truncation (Sun et al. 2019 finding): first 128 + last max_len-130 tokens
    import torch
    ids_list, mask_list = [], []
    head = 128
    for t in texts:
        ids = tok(t, truncation=False, add_special_tokens=False)["input_ids"]
        budget = max_len - 2
        if len(ids) > budget:
            ids = ids[:head] + ids[-(budget - head):]
        ids = [tok.cls_token_id] + ids + [tok.sep_token_id]
        pad = max_len - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [tok.pad_token_id] * pad
        ids_list.append(ids)
        mask_list.append(mask)
    return {"input_ids": torch.tensor(ids_list), "attention_mask": torch.tensor(mask_list)}


def ece_probs(y: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.sum():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="small")
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--headtail", action="store_true")
    ap.add_argument("--clean-text", action="store_true", help="fine-tune on leakage-stripped plain text instead of raw wikitext")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or ("rung3_" + args.model.split("/")[-1].replace("-", "_")
                       + ("_headtail" if args.headtail else ""))

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")

    df = common.load_table(args.mode)
    texts = load_texts(args.mode, clean=args.clean_text)
    df = df[df["pageid"].astype(int).isin(texts)].copy()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=6)
    model.to(dev)

    parts = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        enc = encode(tok, [texts[int(p)] for p in sub["pageid"]], args.max_len, args.headtail)
        y = torch.tensor(sub["label"].to_numpy(int))
        parts[split] = (enc, y, sub["pageid"].to_numpy(int))
        log.info("encoded", split=split, n=len(y))

    tr_ds = TensorDataset(parts["train"][0]["input_ids"], parts["train"][0]["attention_mask"],
                          parts["train"][1])
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    live = ROOT / "results" / args.mode / "live"
    live.mkdir(parents=True, exist_ok=True)
    live_f = live / f"{tag}_s{args.seed}.jsonl"
    live_f.write_text("")  # one live log per run; epochs_ran is counted from it
    mdl_dir = ROOT / "results" / args.mode / "models"
    mdl_dir.mkdir(parents=True, exist_ok=True)
    ckpt = mdl_dir / f"{tag}_s{args.seed}.pt"

    @torch.no_grad()
    def infer(split: str) -> np.ndarray:
        model.eval()
        enc, _, _ = parts[split]
        outs = []
        for i in range(0, len(enc["input_ids"]), 16):
            b = {k: v[i:i + 16].to(dev) for k, v in enc.items()}
            outs.append(model(**b).logits.float().cpu().numpy())
        return np.vstack(outs)

    best_f1, best_state, wall0 = -1.0, None, time.monotonic()
    for epoch in range(1, args.epochs + 1):
        if common.stop_requested():
            log.info("stop_honoured", epoch=epoch)
            break
        model.train()
        tot = 0.0
        for ids, mask, yb in tr_dl:
            opt.zero_grad()
            out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                        labels=yb.to(dev))
            out.loss.backward()
            opt.step()
            tot += float(out.loss)
        val_logits = infer("val")
        vp = val_logits.argmax(axis=1)
        yv = parts["val"][1].numpy()
        f1 = macro_f1(yv, vp)
        rec = {"epoch": epoch, "ts": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
               "train_loss": round(tot / max(len(tr_dl), 1), 4),
               "val_macro_f1": round(f1, 4), "val_acc": round(float((vp == yv).mean()), 4),
               "val_qwk_argmax": round(qwk(yv, vp), 4),
               "wall_s": round(time.monotonic() - wall0, 1), "device": dev}
        with live_f.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("epoch", **rec)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, ckpt)
    model.to(dev)

    # test-time metrics row (merged into eval results by wqa.eval)
    t0 = time.monotonic()
    te_logits = infer("test")
    infer_s = time.monotonic() - t0
    yt = parts["test"][1].numpy()
    probs = np.exp(te_logits - te_logits.max(1, keepdims=True))
    probs = probs / probs.sum(1, keepdims=True)
    pred = probs.argmax(1)
    rr = common.expected_grade_round(probs)
    m = all_metrics(yt, pred)
    lo, hi = bootstrap_ci(yt, pred, "macro_f1")
    th = df[(df["split"] == "test") & (df.get("temporal_holdout", False))]
    th_f1 = None
    if len(th) > 10:
        idx = {int(p): i for i, p in enumerate(parts["test"][2])}
        pairs = [(int(lbl), idx[int(p)]) for p, lbl in zip(th["pageid"], th["label"])
                 if int(p) in idx]
        if pairs:
            th_f1 = macro_f1([a for a, _ in pairs], pred[[b for _, b in pairs]])
    row = {"config": tag, "family": "text", "macro_f1": m["macro_f1"], "ci_lo": lo, "ci_hi": hi,
           "qwk_argmax": m["qwk"], "qwk_regress_round": qwk(yt, rr),
           "within_one": m["within_one"], "acc": m["acc"], "ordinal_mae": m["ordinal_mae"],
           "ece": ece_probs(yt, probs), "temporal_holdout_f1": th_f1,
           "infer_s_per_1k": round(infer_s / max(len(yt), 1) * 1000, 4),
           "device": dev, "epochs_ran": len(live_f.read_text().splitlines()),
           "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    enc_dir = ROOT / "results" / args.mode / "encoder"
    enc_dir.mkdir(parents=True, exist_ok=True)
    rows_p = enc_dir / "eval_rows.json"
    rows = json.loads(rows_p.read_text()) if rows_p.exists() else []
    rows = [r for r in rows if r["config"] != tag] + [row]
    rows_p.write_text(json.dumps(rows, indent=1))

    # per-article logits for the text_plus_H fusion arm
    frames = []
    for split in ("train", "val", "test"):
        lg = infer(split)
        frames.append(pd.DataFrame({"pageid": parts[split][2], "split": split,
                                    **{f"logit_{i}": lg[:, i] for i in range(6)}}))
    pd.concat(frames).to_parquet(enc_dir / f"logits_{tag}_s{args.seed}.parquet", index=False)
    print(f"encoder {tag}: test macro_f1={m['macro_f1']:.4f} device={dev}")


if __name__ == "__main__":
    main()
