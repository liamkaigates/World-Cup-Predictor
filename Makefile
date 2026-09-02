.PHONY: data data-worldcup download-international import-international download-kaggle import-kaggle train backtest predict simulate compare test serve serve-sample train-sample backtest-sample

PYTHONPATH := src
MODEL := artifacts/model.pkl
MATCHES ?= data/international_matches.csv
WORLD_CUP_MATCHES := data/world_cup_matches.csv
SAMPLE_MATCHES := data/sample_matches.csv
INTERNATIONAL_DATASET := martj42/international-football-results-from-1872-to-2017
HOLDOUT_YEAR ?= 2018

# Default: all international matches 1872-present (recommended; keeps ratings current).
data: download-international import-international

download-international:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.download --dataset $(INTERNATIONAL_DATASET) --data-dir data/international

import-international:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.kaggle --data-dir data/international --output $(MATCHES)

# Alternative: the World Cup-only dataset (1930-2014); pair with MATCHES=$(WORLD_CUP_MATCHES) HOLDOUT_YEAR=2010.
data-worldcup: download-kaggle import-kaggle

download-kaggle:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.download --data-dir data/kaggle

import-kaggle:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.kaggle --data-dir data/kaggle --output $(WORLD_CUP_MATCHES)

train:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli train --matches $(MATCHES) --model $(MODEL)

train-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli train --matches $(SAMPLE_MATCHES) --model $(MODEL)

backtest:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli backtest --matches $(MATCHES) --holdout-year $(HOLDOUT_YEAR)

backtest-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli backtest --matches $(SAMPLE_MATCHES) --holdout-year 2022

predict:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli predict --model $(MODEL) --home Argentina --away France --neutral true

simulate:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli simulate --model $(MODEL) --fixtures data/sample_fixtures.csv --runs 1000

compare:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli compare --matches $(MATCHES) --runs 10000 --output artifacts/wc_forecast.csv

test:
	PYTHONPATH=$(PYTHONPATH) python3 -m unittest discover -s tests

serve:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.api --model $(MODEL) --matches $(MATCHES) --port 8000

serve-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.api --model $(MODEL) --matches $(SAMPLE_MATCHES) --port 8000
