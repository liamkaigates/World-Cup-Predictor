from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc_forecast.data import Match
from wc_forecast.features import (
    FEATURE_NAMES,
    TeamState,
    build_prediction_vector,
    build_training_matrix,
)

OUTCOME_LABELS = {
    0: "away_win",
    1: "draw",
    2: "home_win",
}

Matchup = Tuple[str, str, bool, str]

# Bump whenever the feature layout changes so stale pickles fail at load
# time with a clear message instead of at predict time with a shape error.
MODEL_SCHEMA_VERSION = 2


@dataclass
class ForecastModel:
    pipeline: Pipeline
    team_states: Dict[str, TeamState]
    feature_names: List[str]
    schema_version: int = MODEL_SCHEMA_VERSION

    def predict_proba(self, home_team: str, away_team: str, neutral: bool = True, tournament: str = "World Cup") -> Dict[str, float]:
        row = build_prediction_vector(self.team_states, home_team, away_team, neutral, tournament)
        raw = self.pipeline.predict_proba(row)[0]
        probs = {OUTCOME_LABELS[int(cls)]: float(prob) for cls, prob in zip(self.pipeline.classes_, raw)}
        for label in OUTCOME_LABELS.values():
            probs.setdefault(label, 0.0)
        return dict(sorted(probs.items()))

    def predict_proba_batch(self, matchups: Iterable[Matchup]) -> np.ndarray:
        """Predict many matchups with one pipeline call.

        Each matchup is (home_team, away_team, neutral, tournament). Returns an
        array of shape (n, 3) with columns [away_win, draw, home_win].
        """
        rows = np.vstack(
            [
                build_prediction_vector(self.team_states, home, away, neutral, tournament)
                for home, away, neutral, tournament in matchups
            ]
        )
        raw = self.pipeline.predict_proba(rows)
        probs = np.zeros((rows.shape[0], len(OUTCOME_LABELS)))
        for column, cls in enumerate(self.pipeline.classes_):
            probs[:, int(cls)] = raw[:, column]
        return probs

    def predict_label(self, home_team: str, away_team: str, neutral: bool = True, tournament: str = "World Cup") -> str:
        probs = self.predict_proba(home_team, away_team, neutral, tournament)
        return max(probs.items(), key=lambda item: item[1])[0]


def train_model(matches: Iterable[Match]) -> ForecastModel:
    x, y, states = build_training_matrix(matches)
    if len(set(y.tolist())) < 2:
        raise ValueError("Training data must contain at least two outcome classes.")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=7,
                ),
            ),
        ]
    )
    pipeline.fit(x, y)
    return ForecastModel(pipeline=pipeline, team_states=states, feature_names=FEATURE_NAMES)


def save_model(model: ForecastModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str | Path) -> ForecastModel:
    with Path(path).open("rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, ForecastModel):
        raise TypeError("Loaded object is not a ForecastModel.")
    saved_version = getattr(model, "schema_version", 1)
    if saved_version != MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"Model at {path} uses schema version {saved_version}, but this code expects "
            f"{MODEL_SCHEMA_VERSION}. Retrain it (e.g. `make train`)."
        )
    return model


def backtest(matches: List[Match], holdout_year: int) -> Dict[str, float]:
    train = [match for match in matches if match.date.year < holdout_year]
    test = [match for match in matches if match.date.year >= holdout_year]
    if not train or not test:
        raise ValueError("Backtest requires matches on both sides of the holdout year.")

    model = train_model(train)
    y_true = np.asarray([match.outcome for match in test], dtype=int)
    y_prob_arr = model.predict_proba_batch(
        (match.home_team, match.away_team, match.neutral, match.tournament) for match in test
    )
    y_pred = np.argmax(y_prob_arr, axis=1)
    return {
        "train_matches": float(len(train)),
        "test_matches": float(len(test)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob_arr, labels=[0, 1, 2])),
        "home_win_brier": float(brier_score_loss((y_true == 2).astype(int), y_prob_arr[:, 2])),
    }
