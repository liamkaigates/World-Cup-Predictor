from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from wc_forecast.data import Match


FORM_WINDOW = 5

FEATURE_NAMES = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "home_form_points",
    "away_form_points",
    "form_points_diff",
    "home_goal_diff_form",
    "away_goal_diff_form",
    "goal_diff_form_diff",
    "home_matches_played",
    "away_matches_played",
    "neutral",
    "is_world_cup",
]


def _form_window() -> Deque[float]:
    return deque(maxlen=FORM_WINDOW)


@dataclass
class TeamState:
    elo: float = 1500.0
    recent_points: Deque[float] = field(default_factory=_form_window)
    recent_goal_diff: Deque[float] = field(default_factory=_form_window)
    matches_played: int = 0


def match_points(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def update_elo(home: TeamState, away: TeamState, home_goals: int, away_goals: int, k: float = 32.0) -> None:
    home_expected = expected_score(home.elo, away.elo)
    home_actual = actual_score(home_goals, away_goals)
    delta = k * (home_actual - home_expected)
    home.elo += delta
    away.elo -= delta


def mean_last(values: Sequence[float], default: float = 0.0, window: int = FORM_WINDOW) -> float:
    if not values:
        return default
    if len(values) > window:
        values = list(values)[-window:]
    return sum(values) / len(values)


def get_state(states: Dict[str, TeamState], team: str) -> TeamState:
    if team not in states:
        states[team] = TeamState()
    return states[team]


def feature_vector(home: TeamState, away: TeamState, neutral: bool, tournament: str) -> List[float]:
    home_form = mean_last(home.recent_points)
    away_form = mean_last(away.recent_points)
    home_goal_form = mean_last(home.recent_goal_diff)
    away_goal_form = mean_last(away.recent_goal_diff)
    return [
        home.elo - away.elo,
        home.elo,
        away.elo,
        home_form,
        away_form,
        home_form - away_form,
        home_goal_form,
        away_goal_form,
        home_goal_form - away_goal_form,
        float(home.matches_played),
        float(away.matches_played),
        1.0 if neutral else 0.0,
        1.0 if tournament.lower() == "world cup" else 0.0,
    ]


def update_team_history(home: TeamState, away: TeamState, match: Match) -> None:
    update_elo(home, away, match.home_goals, match.away_goals)
    home.recent_points.append(match_points(match.home_goals, match.away_goals))
    away.recent_points.append(match_points(match.away_goals, match.home_goals))
    home.recent_goal_diff.append(float(match.home_goals - match.away_goals))
    away.recent_goal_diff.append(float(match.away_goals - match.home_goals))
    home.matches_played += 1
    away.matches_played += 1


def build_training_matrix(matches: Iterable[Match]) -> Tuple[np.ndarray, np.ndarray, Dict[str, TeamState]]:
    states: Dict[str, TeamState] = {}
    rows: List[List[float]] = []
    labels: List[int] = []

    for match in sorted(matches, key=lambda item: item.date):
        home = get_state(states, match.home_team)
        away = get_state(states, match.away_team)
        rows.append(feature_vector(home, away, match.neutral, match.tournament))
        labels.append(match.outcome)
        update_team_history(home, away, match)

    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), states


def build_prediction_vector(states: Dict[str, TeamState], home_team: str, away_team: str, neutral: bool = True, tournament: str = "World Cup") -> np.ndarray:
    home = states.get(home_team, TeamState())
    away = states.get(away_team, TeamState())
    return np.asarray([feature_vector(home, away, neutral, tournament)], dtype=float)

