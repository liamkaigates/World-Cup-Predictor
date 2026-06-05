# World Cup Forecasting Engine

Probabilistic international soccer match forecasting project built for ML engineering portfolios.

The project trains leakage-safe match outcome models from historical fixtures, evaluates them with backtests, serves predictions over HTTP, and runs Monte Carlo simulations for future fixtures.

- Builds rolling pre-match features instead of using future information.
- Predicts calibrated win/draw/loss probabilities, not only a winner.
- Includes Elo-style team strength, recent form, goal differential, tournament context, and neutral-site features.
- Supports reproducible model training, backtesting, CLI prediction, HTTP serving, and simulation.
- Uses a minimal dependency footprint: Python, NumPy, and scikit-learn.

## Project Structure

```text
world-cup-forecast/
  data/
    sample_matches.csv
  src/wc_forecast/
    api.py
    cli.py
    data.py
    features.py
    models.py
    simulate.py
  tests/
    test_pipeline.py
  requirements.txt
```

## Kaggle Data

This project is wired for the Kaggle dataset [FIFA World Cup by Abecklas](https://www.kaggle.com/datasets/abecklas/fifa-world-cup).

Kaggle usually requires an authenticated account/API token to download datasets. Download the dataset from Kaggle and place the files here:

```text
world-cup-forecast/data/kaggle/WorldCupMatches.csv
world-cup-forecast/data/kaggle/WorldCups.csv
```

With the Kaggle CLI configured, the download command is:

```bash
mkdir -p data/kaggle
kaggle datasets download -d abecklas/fifa-world-cup -p data/kaggle --unzip
```

Then convert the Kaggle schema into the predictor schema:

```bash
make import-kaggle
```

That writes:

```text
data/world_cup_matches.csv
```

The importer uses `WorldCupMatches.csv` for match results and `WorldCups.csv`, when present, to infer whether the match was neutral based on the host country.

## Predictor Data Schema

Use a CSV with these columns:

```text
date,home_team,away_team,home_goals,away_goals,neutral,tournament
```

`neutral` should be `true` or `false`. The included `data/sample_matches.csv` is a toy dataset for local testing. The Kaggle importer produces `data/world_cup_matches.csv` in this schema.

## Quickstart

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Train a model:

```bash
make import-kaggle
make train
```

Backtest by holding out matches from a year onward:

```bash
make backtest
```

Predict a matchup:

```bash
PYTHONPATH=src python -m wc_forecast.cli predict --model artifacts/model.pkl --home Argentina --away France --neutral true
```

Simulate fixtures:

```bash
PYTHONPATH=src python -m wc_forecast.cli simulate --model artifacts/model.pkl --fixtures data/sample_fixtures.csv --runs 1000
```

Run the local prediction server:

```bash
make serve
```

Then call:

```text
http://localhost:8000/predict?home=Argentina&away=France&neutral=true
```

Or open the interactive dashboard:

```text
http://localhost:8000
```

## Make Targets

```bash
make train
make backtest
make predict
make simulate
make test
make serve
make serve-sample
```

Use the toy dataset when the Kaggle files are not available:

```bash
make train-sample
make backtest-sample
```

## Docker

```bash
docker build -t world-cup-forecast .
docker run -p 8000:8000 world-cup-forecast
```

## Suggested Resume Bullet

Built a World Cup forecasting engine with leakage-safe feature engineering, Elo-style team ratings, rolling team-form features, probabilistic scikit-learn models, backtesting, Monte Carlo simulation, and HTTP prediction serving.
