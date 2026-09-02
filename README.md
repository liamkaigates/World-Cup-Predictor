# World Cup Forecasting Engine

![CI](https://github.com/liamkaigates/World-Cup-Predictor/actions/workflows/ci.yml/badge.svg)

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

Two Kaggle datasets are supported:

- **[International football results 1872–present](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)** (martj42) — the default. All international matches through the present, so Elo ratings and form reflect current squads and post-2018 backtests are possible.
- **[FIFA World Cup by Abecklas](https://www.kaggle.com/datasets/abecklas/fifa-world-cup)** — World Cup finals matches only, 1930–2014.

Downloading uses the Kaggle API directly (no extra dependencies), which needs an API credential — either form works:

**Access token** (`KGAT_...`): save it to `~/.kaggle/access_token` (with `chmod 600`) or export it as `KAGGLE_API_TOKEN`. An access token takes precedence when both credential types are present.

**Username + key** (`kaggle.json`): on kaggle.com go to **Settings → API → Create New Token**, then:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

or export `KAGGLE_USERNAME` and `KAGGLE_KEY`. Never commit credentials: `kaggle.json` and `data/kaggle/` are gitignored.

Then download and convert in one step:

```bash
make data
```

That downloads the international dataset into `data/international/` and converts it to `data/international_matches.csv`. For the World Cup–only dataset use `make data-worldcup`, then point the other targets at it with `MATCHES=data/world_cup_matches.csv HOLDOUT_YEAR=2010`. Each step also works on its own, or via the installed console scripts `wc-forecast-download` and `wc-forecast-import-kaggle`.

If you prefer not to create an API key, manually download a dataset from Kaggle and place `results.csv` in `data/international/` (or `WorldCupMatches.csv`/`WorldCups.csv` in `data/kaggle/`) before running the import target.

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

## Tournament Simulation and the 2026 World Cup

`compare` forecasts an entire World Cup and scores the forecast against what actually happened. It reconstructs the real group draw and hosts from the match data, trains a model **only on matches played before the tournament started** (leakage-safe), Monte Carlos the whole tournament — group round-robin, best-thirds advancement, knockout rounds — and reports championship probabilities next to the actual results:

```bash
make compare
```

or directly:

```bash
PYTHONPATH=src python3 -m wc_forecast.cli compare --matches data/international_matches.csv --year 2026 --runs 10000
```

Against the actual 2026 World Cup (trained on the 49,416 matches played before June 11, 2026; 10,000 simulations):

| Predicted rank | Team | Champion prob. | Actual result |
| --- | --- | --- | --- |
| 1 | Argentina | 18.4% | Runner-up |
| 2 | **Spain** | 12.6% | **Champion** |
| 3 | England | 11.9% | Third place |
| 4 | Portugal | 9.3% | — |
| 5 | France | 7.4% | Fourth place |

The model's top two picks contested the actual final (Spain 1–0 Argentina), and all four actual semifinalists were in its top five. Match-level, the pre-tournament model scored 58.7% accuracy / 0.91 log-loss over all 104 matches.

Simulation approximations: knockout pairings re-seed each round by group-stage record rather than following FIFA's fixed bracket template, drawn knockout matches are decided by a 50/50 shootout, and group tiebreaks use Elo as a stand-in for goal difference.

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

- The HTTP server is threaded (`ThreadingHTTPServer`) with HTTP/1.1 keep-alive, per-connection timeouts, and JSON 500 responses instead of dropped connections. Pass `--log-requests` to enable access logging.
- Saved models carry a schema version; loading a model trained by an older feature layout fails at startup with a clear retrain message rather than mispredicting.
- This is a demo-grade server: for a real public deployment put a reverse proxy in front for TLS termination and rate limiting.
- Dataset summary, team list, and match ordering are computed once at startup; static assets are cached in memory with mtime invalidation.
- Monte Carlo simulation predicts each fixture once and draws outcome counts from a multinomial, so cost is independent of the number of runs.
- Backtesting scores all holdout matches in a single batched model call.

## Suggested Resume Bullet

Built a World Cup forecasting engine with leakage-safe feature engineering, Elo-style team ratings, rolling team-form features, probabilistic scikit-learn models, backtesting, Monte Carlo simulation, and HTTP prediction serving.
