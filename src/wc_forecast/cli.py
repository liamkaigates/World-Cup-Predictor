from __future__ import annotations

import argparse
import json

from wc_forecast.data import load_matches, parse_bool, write_predictions
from wc_forecast.models import backtest, load_model, save_model, train_model
from wc_forecast.simulate import load_fixtures, simulate_fixtures


def train_command(args: argparse.Namespace) -> None:
    matches = load_matches(args.matches)
    model = train_model(matches)
    save_model(model, args.model)
    print(json.dumps({"status": "trained", "matches": len(matches), "model": args.model}, indent=2))


def backtest_command(args: argparse.Namespace) -> None:
    matches = load_matches(args.matches)
    metrics = backtest(matches, args.holdout_year)
    print(json.dumps(metrics, indent=2))


def predict_command(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    probs = model.predict_proba(args.home, args.away, parse_bool(str(args.neutral)), args.tournament)
    response = {
        "home_team": args.home,
        "away_team": args.away,
        "prediction": max(probs.items(), key=lambda item: item[1])[0],
        "probabilities": probs,
    }
    print(json.dumps(response, indent=2))


def simulate_command(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    fixtures = load_fixtures(args.fixtures)
    rows = simulate_fixtures(model, fixtures, runs=args.runs, seed=args.seed)
    if args.output:
        write_predictions(args.output, rows)
    print(json.dumps(rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="World Cup match forecasting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a match forecasting model")
    train.add_argument("--matches", required=True)
    train.add_argument("--model", default="artifacts/model.pkl")
    train.set_defaults(func=train_command)

    bt = subparsers.add_parser("backtest", help="Backtest using a holdout year")
    bt.add_argument("--matches", required=True)
    bt.add_argument("--holdout-year", type=int, required=True)
    bt.set_defaults(func=backtest_command)

    predict = subparsers.add_parser("predict", help="Predict one match")
    predict.add_argument("--model", required=True)
    predict.add_argument("--home", required=True)
    predict.add_argument("--away", required=True)
    predict.add_argument("--neutral", default="true")
    predict.add_argument("--tournament", default="World Cup")
    predict.set_defaults(func=predict_command)

    simulate = subparsers.add_parser("simulate", help="Run Monte Carlo simulations over fixtures")
    simulate.add_argument("--model", required=True)
    simulate.add_argument("--fixtures", required=True)
    simulate.add_argument("--runs", type=int, default=1000)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--output")
    simulate.set_defaults(func=simulate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

