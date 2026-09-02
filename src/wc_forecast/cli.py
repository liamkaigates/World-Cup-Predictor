from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from wc_forecast.data import load_matches, parse_bool, write_predictions
from wc_forecast.features import WORLD_CUP_TOURNAMENTS
from wc_forecast.models import backtest, load_model, save_model, train_model
from wc_forecast.simulate import load_fixtures, simulate_fixtures
from wc_forecast.tournament import infer_spec, simulate_tournament, world_cup_matches


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


def shootout_winner(shootouts_csv: str, final) -> str | None:
    path = Path(shootouts_csv)
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("date") == final.date.isoformat()
                and {row.get("home_team"), row.get("away_team")} == {final.home_team, final.away_team}
            ):
                return row.get("winner") or None
    return None


def compare_command(args: argparse.Namespace) -> None:
    matches = load_matches(args.matches)
    year = args.year or max(
        (match.date.year for match in matches if match.tournament.strip().lower() in WORLD_CUP_TOURNAMENTS),
        default=None,
    )
    wc = world_cup_matches(matches, year) if year else []
    if not wc:
        raise SystemExit(f"No World Cup finals matches found for year {args.year} in {args.matches}")

    tournament_start = wc[0].date
    training = [match for match in matches if match.date < tournament_start]
    model = train_model(training)
    try:
        spec = infer_spec(wc)
    except ValueError as error:
        raise SystemExit(str(error))
    forecast = simulate_tournament(model, spec, runs=args.runs, seed=args.seed)

    # Score the pre-tournament model on every match that was actually played.
    y_true = np.asarray([match.outcome for match in wc], dtype=int)
    y_prob = model.predict_proba_batch(
        (match.home_team, match.away_team, match.neutral, match.tournament) for match in wc
    )
    match_metrics = {
        "matches": len(wc),
        "accuracy": float(accuracy_score(y_true, np.argmax(y_prob, axis=1))),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1, 2])),
    }

    final = wc[-1]
    if final.home_goals != final.away_goals:
        actual_champion = final.home_team if final.home_goals > final.away_goals else final.away_team
    else:
        actual_champion = shootout_winner(args.shootouts, final)
    actual_runner_up = (
        ({final.home_team, final.away_team} - {actual_champion}).pop() if actual_champion else None
    )
    forecast_rank = {row["team"]: position + 1 for position, row in enumerate(forecast)}

    report = {
        "year": year,
        "teams": len(spec.teams),
        "groups": len(spec.groups),
        "hosts": sorted(spec.hosts),
        "trained_on_matches_before": tournament_start.isoformat(),
        "training_matches": len(training),
        "simulation_runs": args.runs,
        "match_level": match_metrics,
        "predicted_top10": [
            {key: round(value, 4) if isinstance(value, float) else value for key, value in row.items()}
            for row in forecast[:10]
        ],
        "actual": {
            "champion": actual_champion,
            "runner_up": actual_runner_up,
            "final": f"{final.home_team} {final.home_goals}-{final.away_goals} {final.away_team}",
            "final_four": sorted(
                {team for match in wc[-2:] for team in (match.home_team, match.away_team)}
            ),
        },
        "actual_champion_forecast": {
            "predicted_rank": forecast_rank.get(actual_champion),
            "champion_probability": next(
                (round(row["champion"], 4) for row in forecast if row["team"] == actual_champion), None
            ),
        }
        if actual_champion
        else None,
    }
    if args.output:
        write_predictions(args.output, forecast)
    print(json.dumps(report, indent=2))


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

    compare = subparsers.add_parser(
        "compare",
        help="Simulate a whole World Cup with a pre-tournament model and compare against the actual results",
    )
    compare.add_argument("--matches", required=True, help="Full match history CSV (must include the tournament)")
    compare.add_argument("--year", type=int, help="World Cup year; defaults to the most recent in the data")
    compare.add_argument("--runs", type=int, default=5000)
    compare.add_argument("--seed", type=int, default=7)
    compare.add_argument(
        "--shootouts",
        default="data/international/shootouts.csv",
        help="Shootouts CSV used to resolve a final drawn in regulation",
    )
    compare.add_argument("--output", help="Optional CSV path for the full per-team forecast")
    compare.set_defaults(func=compare_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

