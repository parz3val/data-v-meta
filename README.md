# Predicting Wikipedia Article Quality from Metadata

Code, data and trained models for the MSc dissertation *Predicting Wikipedia Article Quality from Metadata: Accuracy, Trade-offs and Detection of Outdated Grades* (Harish Kunwar, London Metropolitan University, 2026).

The study compares a **metadata arm** (CatBoost and other tabular learners on 33 counts taken from an article's markup and edit history) with a **text arm** (a DistilRoBERTa encoder fine-tuned on the first 128 and last 382 tokens of the stripped article) on the same 24,000-article balanced split of English Wikipedia, six grades from Stub to FA. It also fits a fusion model, a distilled browser-sized student (WikiLite), a staleness score built from model-human disagreement, and SHAP explanations.

## What is here

| Path | Contents |
|---|---|
| `src/wqa/` | the pipeline: `collect` (MediaWiki and pageview APIs), `clean` (leakage-stripped parsing), `features`, `eda`, `models` (selection, training, encoder, fusion, custom plugins), `eval`, `xai`, `staleness` |
| `configs/` | collection, feature, split and model configuration files |
| `scripts/` | learning curve, calibration, grade-date recovery, tuned refits, the encoder leakage check, ORES comparison, artefact download |
| `app/` | the Flask demonstration app that grades a live article with every saved model |
| `data/full/` | manifest, feature table and frozen splits for the full level; raw wikitext and the cleaned table are release assets |
| `results/full/` | the model selection record and the test results; trained models are release assets |
| `tests/` | unit tests for parsing, leakage rules, features and configuration |

## Setup

```bash
git clone https://github.com/parz3val/data-v-meta.git
cd data-v-meta
python -m venv .venv && source .venv/bin/activate
pip install -e ".[enc]"          # drop [enc] if you do not need the text encoder
bash scripts/download_artefacts.sh   # trained models, raw wikitext and cleaned table (~3 GB, needs the gh CLI)
```

Python 3.11 or newer. The encoder trains and runs on Apple Silicon (MPS) or CUDA; on CPU it still runs, slowly.

## Run the demo app

```bash
make app
```

Open http://127.0.0.1:5055 and enter an article title (`Alan Turing`), a full link (`https://en.wikipedia.org/wiki/Alan_Turing`) or a bare path. The app fetches the live article through the same collector code as training, builds the 33 features, and shows every saved model's grade probabilities, the two decision rules, the staleness score against the grade on record and the top SHAP contributions. The encoder runs in a worker process because the tree libraries and PyTorch each bring an OpenMP runtime. Nothing typed is stored.

## Reproduce the pipeline

Each stage reads the previous one's output under `data/<mode>/` and `results/<mode>/`. `MODE` is `small` (150 articles per grade), `medium` (400) or `full` (4,000).

```bash
make collect MODE=full        # crawl by grade; resumable; polite rate; set your contact email in configs/collection.yaml
make clean-data MODE=full     # strip grade templates and markup, parse structure and references
make features MODE=full       # 33 features in five families
make splits MODE=full         # stratified 70/15/15 split, frozen (refuses to overwrite)
make eda MODE=full
make select MODE=full         # Optuna search per learner on validation macro-F1; writes status/models.json
make train-all MODE=full      # every configuration, three seeds, tuned settings
make encoder MODE=full        # DistilRoBERTa on leakage-stripped text (needs [enc]; hours on a laptop GPU)
make fusion MODE=full         # encoder logits + history features, fitted on the validation partition
make eval MODE=full           # single reading of the test partition with paired bootstrap intervals
make xai MODE=full            # SHAP, faithfulness, stability
make staleness MODE=full      # disagreement score against grade-age proxies
```

`scripts/run_pipeline.sh <mode>` runs the whole chain. `scripts/learning_curve.py` retrains the metadata arm at rising sizes, `scripts/refit_tuned.py` refits every learner with its tuned settings, `scripts/encoder_leak_check.py` scores a saved encoder on raw, template-stripped and clean inputs, and `scripts/ores_test_split.py` grades the test partition with Wikimedia's deployed ORES model for comparison.

## Trained models

After `scripts/download_artefacts.sh`, `results/full/models/` holds one pickle per configuration and seed (`<config>_s<seed>.pkl`, a dict with `est`, `family`, `kind`, `params`, `seed`) and the encoder state dict `rung3_distilroberta_base_headtail_s42.pt` for `distilroberta-base` with a six-way head. The WikiLite student is also exported as JSON for the dependency-free JavaScript scorer in `scripts/wikilite_js/`.

## Data

`data/full/raw/<grade>.jsonl` holds one record per article: page and revision identifiers, wikitext, categories, templates, edit and editor counts, and ninety days of pageviews, as collected in August 2026. `data/full/interim/full.parquet` is the cleaned table with the stripped text and parsed counts; `data/full/features/features.parquet` the 33 features with labels; `data/full/splits/` the frozen partitions. Wikipedia content is CC BY-SA 4.0.

## Licence

Code is MIT. Data and trained models carry the Creative Commons Attribution-ShareAlike terms of the text they derive from. See `LICENSE`.
