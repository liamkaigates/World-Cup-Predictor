import unittest
from datetime import date, timedelta

from wc_forecast.data import Match
from wc_forecast.models import train_model
from wc_forecast.tournament import (
    infer_spec,
    knockout_bracket_size,
    simulate_tournament,
    world_cup_matches,
)

TEAMS = ["Alphaland", "Betaland", "Gammaland", "Deltaland", "Epsilonia", "Zetaland", "Etaland", "Thetaland"]
GROUPS = [TEAMS[:4], TEAMS[4:]]


def make_match(day: date, home: str, away: str, home_goals: int, away_goals: int, tournament: str, neutral: bool = True) -> Match:
    return Match(
        date=day,
        home_team=home,
        away_team=away,
        home_goals=home_goals,
        away_goals=away_goals,
        neutral=neutral,
        tournament=tournament,
    )


def build_history_and_tournament():
    """A synthetic history (friendlies) plus a mini 8-team, 2-group World Cup."""
    matches = []
    day = date(2024, 1, 1)
    # Earlier teams in TEAMS beat later teams, giving the model a clear ordering.
    for _ in range(4):
        for strong_pos, strong in enumerate(TEAMS):
            for weak in TEAMS[strong_pos + 1 :]:
                matches.append(make_match(day, strong, weak, 2, 0, "Friendly"))
                day += timedelta(days=1)
    matches.append(make_match(day, TEAMS[0], TEAMS[1], 1, 1, "Friendly"))

    wc_day = date(2025, 6, 1)
    wc = []
    for members in GROUPS:
        for a_pos in range(4):
            for b_pos in range(a_pos + 1, 4):
                neutral = members[a_pos] != "Alphaland"
                wc.append(
                    make_match(wc_day, members[a_pos], members[b_pos], 1, 0, "FIFA World Cup", neutral=neutral)
                )
                wc_day += timedelta(days=1)
    # Knockout: semifinals, third place, final.
    wc.append(make_match(wc_day, "Alphaland", "Zetaland", 2, 0, "FIFA World Cup"))
    wc.append(make_match(wc_day + timedelta(days=1), "Epsilonia", "Betaland", 0, 1, "FIFA World Cup"))
    wc.append(make_match(wc_day + timedelta(days=2), "Zetaland", "Epsilonia", 2, 1, "FIFA World Cup"))
    wc.append(make_match(wc_day + timedelta(days=3), "Alphaland", "Betaland", 3, 1, "FIFA World Cup"))
    return matches, wc


class TournamentTest(unittest.TestCase):
    def test_bracket_sizes(self):
        self.assertEqual(knockout_bracket_size(2), 4)
        self.assertEqual(knockout_bracket_size(6), 16)
        self.assertEqual(knockout_bracket_size(8), 16)
        self.assertEqual(knockout_bracket_size(12), 32)

    def test_world_cup_matches_filters_and_sorts(self):
        history, wc = build_history_and_tournament()
        selected = world_cup_matches(history + wc, 2025)
        self.assertEqual(len(selected), len(wc))
        dates = [match.date for match in selected]
        self.assertEqual(dates, sorted(dates))

    def test_infer_spec_reconstructs_groups_and_hosts(self):
        _, wc = build_history_and_tournament()
        spec = infer_spec(wc)
        self.assertEqual(len(spec.groups), 2)
        self.assertEqual(sorted(spec.groups["A"]), sorted(GROUPS[0]))
        self.assertEqual(sorted(spec.groups["B"]), sorted(GROUPS[1]))
        self.assertEqual(spec.hosts, {"Alphaland"})

    def test_simulation_probabilities_are_coherent(self):
        history, wc = build_history_and_tournament()
        model = train_model(history)
        spec = infer_spec(wc)
        forecast = simulate_tournament(model, spec, runs=400, seed=3)
        again = simulate_tournament(model, spec, runs=400, seed=3)
        self.assertEqual(forecast, again)
        self.assertAlmostEqual(sum(row["champion"] for row in forecast), 1.0, places=9)
        self.assertAlmostEqual(sum(row["finalist"] for row in forecast), 2.0, places=9)
        self.assertAlmostEqual(sum(row["semifinalist"] for row in forecast), 4.0, places=9)
        by_team = {row["team"]: row for row in forecast}
        # The strongest team in the synthetic history should be the favorite.
        self.assertEqual(forecast[0]["team"], "Alphaland")
        for row in forecast:
            self.assertGreaterEqual(row["finalist"], row["champion"])
            self.assertGreaterEqual(row["semifinalist"], row["finalist"])
        self.assertGreater(by_team["Alphaland"]["champion"], by_team["Thetaland"]["champion"])


if __name__ == "__main__":
    unittest.main()
