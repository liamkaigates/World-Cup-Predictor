from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

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


def sample_outcome(probs: Dict[str, float], rng: np.random.Generator | None = None) -> str:
    labels = ["away_win", "draw", "home_win"]
    weights = np.asarray([probs[label] for label in labels], dtype=float)
    rng = rng or np.random.default_rng()
    return labels[rng.choice(len(labels), p=weights / weights.sum())]


def simulate_fixtures(model: ForecastModel, fixtures: List[dict], runs: int = 1000, seed: int = 7) -> List[dict]:
    if runs <= 0:
        raise ValueError("runs must be a positive integer")
    if not fixtures:
        return []

    rng = np.random.default_rng(seed)
    # Outcome probabilities are fixed per fixture, so predict each fixture once
    # and draw all runs from a single multinomial instead of resampling the model.
    probs = model.predict_proba_batch(
        (fixture["home_team"], fixture["away_team"], fixture["neutral"], "World Cup")
        for fixture in fixtures
    )

    rows = []
    for fixture, fixture_probs in zip(fixtures, probs):
        counts = rng.multinomial(runs, fixture_probs / fixture_probs.sum())
        rows.append(
            {
                "matchup": f"{fixture['home_team']} vs {fixture['away_team']}",
                "home_win": counts[2] / runs,
                "draw": counts[1] / runs,
                "away_win": counts[0] / runs,
            }
        )
    return rows
