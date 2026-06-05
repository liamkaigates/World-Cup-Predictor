from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Match:
    date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    neutral: bool
    tournament: str

    @property
    def outcome(self) -> int:
        """Return 2 for home win, 1 for draw, 0 for away win."""
        if self.home_goals > self.away_goals:
            return 2
        if self.home_goals == self.away_goals:
            return 1
        return 0


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_matches(path: str | Path) -> List[Match]:
    matches: List[Match] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "neutral",
            "tournament",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            matches.append(
                Match(
                    date=date.fromisoformat(row["date"]),
                    home_team=row["home_team"].strip(),
                    away_team=row["away_team"].strip(),
                    home_goals=int(row["home_goals"]),
                    away_goals=int(row["away_goals"]),
                    neutral=parse_bool(row["neutral"]),
                    tournament=row["tournament"].strip(),
                )
            )

    return sorted(matches, key=lambda match: match.date)


def write_predictions(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

