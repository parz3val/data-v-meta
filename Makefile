# Pipeline for one data level: make <target> MODE=full   (small = 150 articles per grade, medium = 400, full = 4,000)
PY ?= python
MODE ?= full
.PHONY: collect clean-data features splits eda select train-all eval xai staleness encoder fusion app models

collect:      ; $(PY) -m wqa.collect --mode $(MODE)
clean-data:   ; $(PY) -m wqa.clean --mode $(MODE)
features:     ; $(PY) -m wqa.features --mode $(MODE)
splits:       ; $(PY) -m wqa.features --mode $(MODE) --splits
eda:          ; $(PY) -m wqa.eda --mode $(MODE)
select:       ; $(PY) -m wqa.models --mode $(MODE) --select
train-all:    ; $(PY) -m wqa.models --mode $(MODE) --train-all
eval:         ; $(PY) -m wqa.eval --mode $(MODE)
xai:          ; $(PY) -m wqa.xai --mode $(MODE)
staleness:    ; $(PY) -m wqa.staleness --mode $(MODE)
encoder:      ; $(PY) -m wqa.models.encoder --mode $(MODE) --model distilroberta-base --epochs 3 --seed 42 --headtail --max-len 512 --clean-text
fusion:       ; $(PY) -m wqa.models.text_plus_h --mode $(MODE) --tag rung3_distilroberta_base_headtail --seed 42 --fit-split val
models:       ; bash scripts/download_artefacts.sh
app:          ; HF_HUB_OFFLINE=0 KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 WQA_MODE=$(MODE) $(PY) -m flask --app app.server run --port 5055
