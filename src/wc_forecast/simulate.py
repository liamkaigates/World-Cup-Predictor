from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List

from wc_forecast.data import parse_bool
from wc_forecast.models import ForecastModel


def load_fixtures(path: str | Path) -> List[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"home_team", "away_team", "neutral"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing fixture columns: {sorted(missing)}")
        return [
            {
                "home_team": row["home_team"].strip(),
                "away_team": row["away_team"].strip(),
                "neutral": parse_bool(row["neutral"]),
            }
            for row in reader
        ]


def sample_outcome(probs: Dict[str, float]) -> str:
    labels = ["away_win", "draw", "home_win"]
    weights = [probs[label] for label in labels]
    return random.choices(labels, weights=weights, k=1)[0]


def simulate_fixtures(model: ForecastModel, fixtures: List[dict], runs: int = 1000, seed: int = 7) -> List[dict]:
    random.seed(seed)
    counts: Dict[str, Dict[str, int]] = {}

    for fixture in fixtures:
        matchup = f"{fixture['home_team']} vs {fixture['away_team']}"
        counts[matchup] = {"home_win": 0, "draw": 0, "away_win": 0}

    for _ in range(runs):
        for fixture in fixtures:
            probs = model.predict_proba(
                fixture["home_team"],
                fixture["away_team"],
                neutral=fixture["neutral"],
            )
            outcome = sample_outcome(probs)
            matchup = f"{fixture['home_team']} vs {fixture['away_team']}"
            counts[matchup][outcome] += 1

    rows = []
    for fixture in fixtures:
        matchup = f"{fixture['home_team']} vs {fixture['away_team']}"
        row = {"matchup": matchup}
        row.update({key: value / runs for key, value in counts[matchup].items()})
        rows.append(row)
    return rows

