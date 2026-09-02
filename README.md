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
    sample_fixtures.csv
  src/wc_forecast/
    api.py
    cli.py
    data.py
    features.py
    kaggle.py
    models.py
    simulate.py
  static/            # dashboard frontend
  tests/
    test_pipeline.py
  pyproject.toml
  requirements.txt
```

## Kaggle Data

This project is wired for the Kaggle dataset [FIFA World Cup by Abecklas](https://www.kaggle.com/datasets/abecklas/fifa-world-cup).

Downloading uses the Kaggle API directly (no extra dependencies), which needs an API key:

1. On kaggle.com go to **Settings → API → Create New Token**. This downloads a `kaggle.json` file.
2. Save it to `~/.kaggle/kaggle.json` and restrict its permissions:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Alternatively, export `KAGGLE_USERNAME` and `KAGGLE_KEY` in the environment (these take precedence over the file). Never commit the key: `kaggle.json` and `data/kaggle/` are gitignored.

Then download and convert in one step:

```bash
make data
```

That runs `make download-kaggle` (fetches and unzips the CSVs into `data/kaggle/`) followed by `make import-kaggle` (converts the Kaggle schema into the predictor schema). Each step also works on its own, or via the installed console scripts `wc-forecast-download` and `wc-forecast-import-kaggle`.

If you prefer not to create an API key, manually download the dataset from Kaggle and place the files here before running `make import-kaggle`:

```text
world-cup-forecast/data/kaggle/WorldCupMatches.csv
world-cup-forecast/data/kaggle/WorldCups.csv
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
python -m pip install -e .
```

Installing the package provides three console scripts: `wc-forecast` (CLI), `wc-forecast-api` (HTTP server), and `wc-forecast-import-kaggle` (data importer). The Makefile targets below also work without installing by setting `PYTHONPATH=src`.

Train a model:

```bash
make data
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

The image trains on the bundled sample data at build time, runs as a non-root user, and exposes a `HEALTHCHECK` against `/health`. To serve a model trained on the full Kaggle data, mount it and override the command:

```bash
docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" -v "$PWD/data:/app/data" world-cup-forecast \
  wc-forecast-api --model artifacts/model.pkl --matches data/world_cup_matches.csv --host 0.0.0.0 --port 8000
```

## Production Notes

- The HTTP server is threaded (`ThreadingHTTPServer`) with HTTP/1.1 keep-alive, per-connection timeouts, and JSON 500 responses instead of dropped connections.
- Dataset summary, team list, and match ordering are computed once at startup; static assets are cached in memory with mtime invalidation.
- Monte Carlo simulation predicts each fixture once and draws outcome counts from a multinomial, so cost is independent of the number of runs.
- Backtesting scores all holdout matches in a single batched model call.

## Suggested Resume Bullet

Built a World Cup forecasting engine with leakage-safe feature engineering, Elo-style team ratings, rolling team-form features, probabilistic scikit-learn models, backtesting, Monte Carlo simulation, and HTTP prediction serving.
