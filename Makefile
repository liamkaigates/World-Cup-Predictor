.PHONY: data download-kaggle import-kaggle train backtest predict simulate test serve

PYTHONPATH := src
MODEL := artifacts/model.pkl
MATCHES := data/world_cup_matches.csv
SAMPLE_MATCHES := data/sample_matches.csv

data: download-kaggle import-kaggle

download-kaggle:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.download --data-dir data/kaggle

import-kaggle:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.kaggle --data-dir data/kaggle --output $(MATCHES)

train:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli train --matches $(MATCHES) --model $(MODEL)

train-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli train --matches $(SAMPLE_MATCHES) --model $(MODEL)

backtest:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli backtest --matches $(MATCHES) --holdout-year 2022

backtest-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli backtest --matches $(SAMPLE_MATCHES) --holdout-year 2022

predict:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli predict --model $(MODEL) --home Argentina --away France --neutral true

simulate:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.cli simulate --model $(MODEL) --fixtures data/sample_fixtures.csv --runs 1000

test:
	PYTHONPATH=$(PYTHONPATH) python3 -m unittest discover -s tests

serve:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.api --model $(MODEL) --matches $(MATCHES) --port 8000

serve-sample:
	PYTHONPATH=$(PYTHONPATH) python3 -m wc_forecast.api --model $(MODEL) --matches $(SAMPLE_MATCHES) --port 8000
