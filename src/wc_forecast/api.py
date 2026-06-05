from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from wc_forecast.data import parse_bool
from wc_forecast.data import Match, load_matches
from wc_forecast.models import ForecastModel, load_model


class PredictionHandler(BaseHTTPRequestHandler):
    model: ForecastModel
    matches: list[Match]
    static_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.write_json({"status": "ok"})
            return
        if parsed.path in {"/", "/dashboard"}:
            self.serve_static("index.html")
            return
        if parsed.path.startswith("/static/"):
            self.serve_static(parsed.path.removeprefix("/static/"))
            return
        if parsed.path == "/api/teams":
            self.write_json({"teams": sorted(self.model.team_states)})
            return
        if parsed.path == "/api/summary":
            self.write_json(summary_payload(self.matches))
            return
        if parsed.path == "/api/matches":
            params = parse_qs(parsed.query)
            team = params.get("team", [""])[0]
            limit = int(params.get("limit", ["50"])[0])
            self.write_json({"matches": match_rows(self.matches, team=team, limit=limit)})
            return
        if parsed.path not in {"/predict", "/api/predict"}:
            self.write_json({"error": "Use /api/predict?home=Argentina&away=France&neutral=true"}, status=404)
            return

        params = parse_qs(parsed.query)
        home = params.get("home", [""])[0]
        away = params.get("away", [""])[0]
        neutral = parse_bool(params.get("neutral", ["true"])[0])
        tournament = params.get("tournament", ["World Cup"])[0]
        if not home or not away:
            self.write_json({"error": "home and away are required query parameters"}, status=400)
            return

        probs = self.model.predict_proba(home, away, neutral=neutral, tournament=tournament)
        self.write_json(
            {
                "home_team": home,
                "away_team": away,
                "prediction": max(probs.items(), key=lambda item: item[1])[0],
                "probabilities": probs,
            }
        )

    def serve_static(self, relative_path: str) -> None:
        if relative_path in {"", "/"}:
            relative_path = "index.html"
        root = self.static_dir.resolve()
        target = (root / relative_path).resolve()
        if root not in target.parents and target != root:
            self.write_json({"error": "invalid static path"}, status=400)
            return
        if not target.exists() or not target.is_file():
            self.write_json({"error": "static asset not found"}, status=404)
            return

        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return

    def write_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def match_rows(matches: list[Match], team: str = "", limit: int = 50) -> list[dict]:
    selected = matches
    if team:
        selected = [match for match in matches if team in {match.home_team, match.away_team}]
    rows = []
    for match in sorted(selected, key=lambda item: item.date, reverse=True)[:limit]:
        rows.append(
            {
                "date": match.date.isoformat(),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "score": f"{match.home_goals}-{match.away_goals}",
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
                "neutral": match.neutral,
                "tournament": match.tournament,
                "outcome": "home_win" if match.home_goals > match.away_goals else "draw" if match.home_goals == match.away_goals else "away_win",
            }
        )
    return rows


def summary_payload(matches: list[Match]) -> dict:
    teams = sorted({team for match in matches for team in (match.home_team, match.away_team)})
    years = sorted({match.date.year for match in matches})
    goals = sum(match.home_goals + match.away_goals for match in matches)
    outcome_counts = {"home_win": 0, "draw": 0, "away_win": 0}
    team_records: dict[str, dict[str, int]] = {}

    for match in matches:
        if match.home_goals > match.away_goals:
            outcome_counts["home_win"] += 1
        elif match.home_goals == match.away_goals:
            outcome_counts["draw"] += 1
        else:
            outcome_counts["away_win"] += 1

        for team in (match.home_team, match.away_team):
            team_records.setdefault(team, {"matches": 0, "goals_for": 0, "goals_against": 0})
        team_records[match.home_team]["matches"] += 1
        team_records[match.home_team]["goals_for"] += match.home_goals
        team_records[match.home_team]["goals_against"] += match.away_goals
        team_records[match.away_team]["matches"] += 1
        team_records[match.away_team]["goals_for"] += match.away_goals
        team_records[match.away_team]["goals_against"] += match.home_goals

    top_attack = sorted(
        (
            {
                "team": team,
                "matches": record["matches"],
                "goals_per_match": record["goals_for"] / record["matches"],
                "goal_diff": record["goals_for"] - record["goals_against"],
            }
            for team, record in team_records.items()
            if record["matches"] >= 3
        ),
        key=lambda item: item["goals_per_match"],
        reverse=True,
    )[:8]

    return {
        "match_count": len(matches),
        "team_count": len(teams),
        "first_year": min(years) if years else None,
        "last_year": max(years) if years else None,
        "total_goals": goals,
        "goals_per_match": goals / len(matches) if matches else 0,
        "outcomes": outcome_counts,
        "top_attack": top_attack,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve World Cup match predictions over HTTP")
    parser.add_argument("--model", required=True)
    parser.add_argument("--matches", default="data/sample_matches.csv")
    parser.add_argument("--static-dir", default="static")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    PredictionHandler.model = load_model(args.model)
    PredictionHandler.matches = load_matches(args.matches)
    PredictionHandler.static_dir = Path(args.static_dir)
    server = HTTPServer((args.host, args.port), PredictionHandler)
    print(f"Serving predictions at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
