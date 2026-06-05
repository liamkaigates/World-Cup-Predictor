from __future__ import annotations

import argparse
import csv
import math
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


MATCH_COLUMNS = {
    "Year",
    "Datetime",
    "Stage",
    "Home Team Name",
    "Home Team Goals",
    "Away Team Goals",
    "Away Team Name",
}


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "na", "none"}:
        return ""
    return text


def clean_team(value: str) -> str:
    replacements = {
        "rn\">Republic of Ireland": "Republic of Ireland",
        "rn\">United Arab Emirates": "United Arab Emirates",
        "rn\">Trinidad and Tobago": "Trinidad and Tobago",
        "Soviet Union": "Russia",
        "Germany FR": "Germany",
    }
    value = clean_cell(value)
    return replacements.get(value, value)


def parse_int(value: str) -> int:
    text = clean_cell(value)
    if not text:
        raise ValueError("empty integer")
    return int(float(text))


def parse_match_date(year: int, value: str) -> date:
    text = clean_cell(value)
    if " - " in text:
        text = text.split(" - ", 1)[0].strip()
    for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return date(year, 1, 1)


def parse_host_countries(world_cups_csv: str | Path | None) -> Dict[int, Set[str]]:
    if not world_cups_csv:
        return {}
    path = Path(world_cups_csv)
    if not path.exists():
        return {}

    hosts: Dict[int, Set[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                year = parse_int(row.get("Year", ""))
            except ValueError:
                continue
            country = clean_cell(row.get("Country", ""))
            if not country:
                continue
            host_names = {clean_team(part.strip()) for part in country.replace("/", ",").split(",")}
            hosts[year] = {name for name in host_names if name}
    return hosts


def infer_neutral(year: int, home_team: str, away_team: str, hosts: Dict[int, Set[str]]) -> bool:
    host_names = hosts.get(year, set())
    if not host_names:
        return True
    return home_team not in host_names and away_team not in host_names


def convert_world_cup_matches(
    matches_csv: str | Path,
    output_csv: str | Path,
    world_cups_csv: str | Path | None = None,
) -> int:
    matches_path = Path(matches_csv)
    output_path = Path(output_csv)
    hosts = parse_host_countries(world_cups_csv)
    rows: List[dict] = []
    seen = set()

    with matches_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = MATCH_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing Kaggle match columns: {sorted(missing)}")

        for row in reader:
            try:
                year = parse_int(row["Year"])
                home_goals = parse_int(row["Home Team Goals"])
                away_goals = parse_int(row["Away Team Goals"])
            except ValueError:
                continue

            home_team = clean_team(row["Home Team Name"])
            away_team = clean_team(row["Away Team Name"])
            if not home_team or not away_team:
                continue

            match_date = parse_match_date(year, row["Datetime"])
            key = (
                match_date.isoformat(),
                home_team,
                away_team,
                home_goals,
                away_goals,
                clean_cell(row["Stage"]),
            )
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "date": match_date.isoformat(),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "neutral": "true" if infer_neutral(year, home_team, away_team, hosts) else "false",
                    "tournament": "World Cup",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "neutral",
                "tournament",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: item["date"]))
    return len(rows)


def default_kaggle_paths(data_dir: str | Path) -> tuple[Path, Optional[Path]]:
    root = Path(data_dir)
    matches = root / "WorldCupMatches.csv"
    cups = root / "WorldCups.csv"
    return matches, cups if cups.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kaggle FIFA World Cup CSV files into predictor training data.")
    parser.add_argument("--matches", help="Path to Kaggle WorldCupMatches.csv")
    parser.add_argument("--world-cups", help="Optional path to Kaggle WorldCups.csv for host-country neutral-site inference")
    parser.add_argument("--data-dir", default="data/kaggle", help="Directory containing Kaggle CSVs when --matches is omitted")
    parser.add_argument("--output", default="data/world_cup_matches.csv")
    args = parser.parse_args()

    if args.matches:
        matches = Path(args.matches)
        cups = Path(args.world_cups) if args.world_cups else None
    else:
        matches, cups = default_kaggle_paths(args.data_dir)

    count = convert_world_cup_matches(matches, args.output, cups)
    print(f"Wrote {count} matches to {args.output}")


if __name__ == "__main__":
    main()
