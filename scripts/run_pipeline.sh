#!/usr/bin/env bash
# Staged pipeline for one data level: bash scripts/run_pipeline.sh <mode>  (LEVELS=... for the learning curve).
# Stages: clean→features→splits→eda→select→train-all→eval→xai→curve→calibration→encoder(MPS)→fusion→re-eval→staleness→viz→status.
cd "$(dirname "$0")/.."
MODE=${1:-medium}
LEVELS=${LEVELS:-100,250,500,1000,2000,2800}
set -o pipefail
for t in clean-data features splits eda select train-all eval xai; do
  echo "=== $t $(date -u +%H:%M:%SZ)"
  make $t MODE=$MODE 2>&1 | tail -2 || { echo "=== ABORT $t $(date -u +%H:%M:%SZ)"; exit 1; }
done
echo "=== curve $(date -u +%H:%M:%SZ)"
python scripts/learning_curve.py --mode $MODE --levels $LEVELS 2>&1 | tail -8
echo "=== calibration $(date -u +%H:%M:%SZ)"
python scripts/calibration.py --mode $MODE 2>&1 | tail -1
echo "=== encoder $(date -u +%H:%M:%SZ)"
python -m wqa.models.encoder --mode $MODE --model distilroberta-base --epochs 3 --seed 42 --headtail 2>&1 | tail -3
echo "=== text_plus_h $(date -u +%H:%M:%SZ)"
python -m wqa.models.text_plus_h --mode $MODE --tag rung3_distilroberta_base_headtail --seed 42 2>&1 | tail -2
echo "=== re-eval with encoder rows $(date -u +%H:%M:%SZ)"
make eval MODE=$MODE 2>&1 | tail -2
make staleness MODE=$MODE 2>&1 | tail -2
echo "=== viz $(date -u +%H:%M:%SZ)"
python scripts/embedding_space.py --mode $MODE 2>&1 | tail -1
echo "=== DONE $MODE $(date -u +%H:%M:%SZ)"
