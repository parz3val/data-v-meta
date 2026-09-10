#!/usr/bin/env bash
# Fetch the trained models and the large data files from the GitHub release into place.
# Needs the GitHub CLI (gh) logged in, or edit the URLs to use curl.
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-v1.0}
mkdir -p results/full/models data/full/raw data/full/interim
gh release download "$TAG" -R parz3val/data-v-meta -p 'model_*' -D results/full/models --clobber
gh release download "$TAG" -R parz3val/data-v-meta -p 'raw_*.jsonl' -D data/full/raw --clobber
gh release download "$TAG" -R parz3val/data-v-meta -p 'interim_full.parquet' -D data/full/interim --clobber
# the release assets carry a prefix so they list cleanly; strip it
for f in results/full/models/model_*; do mv "$f" "results/full/models/${f##*/model_}"; done
for f in data/full/raw/raw_*; do mv "$f" "data/full/raw/${f##*/raw_}"; done
mv data/full/interim/interim_full.parquet data/full/interim/full.parquet
echo "artefacts in place"
