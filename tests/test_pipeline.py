import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wc_forecast.api import match_rows
from wc_forecast.data import load_matches
from wc_forecast.kaggle import convert_international_results, convert_world_cup_matches
from wc_forecast.models import backtest, load_model, save_model, train_model
from wc_forecast.simulate import simulate_fixtures


class ForecastPipelineTest(unittest.TestCase):
    def test_training_and_prediction(self):
        matches = load_matches("data/sample_matches.csv")
        model = train_model(matches)
        probs = model.predict_proba("Argentina", "France", neutral=True)
        self.assertEqual(set(probs), {"away_win", "draw", "home_win"})
        self.assertLess(abs(sum(probs.values()) - 1.0), 1e-6)

    def test_backtest_metrics(self):
        matches = load_matches("data/sample_matches.csv")
        metrics = backtest(matches, holdout_year=2022)
        self.assertGreater(metrics["train_matches"], 0)
        self.assertGreater(metrics["test_matches"], 0)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_batch_prediction_matches_single(self):
        matches = load_matches("data/sample_matches.csv")
        model = train_model(matches)
        matchups = [
            ("Argentina", "France", True, "World Cup"),
            ("Brazil", "Germany", False, "Friendly"),
        ]
        batch = model.predict_proba_batch(matchups)
        for row, (home, away, neutral, tournament) in zip(batch, matchups):
            single = model.predict_proba(home, away, neutral, tournament)
            self.assertAlmostEqual(row[0], single["away_win"], places=9)
            self.assertAlmostEqual(row[1], single["draw"], places=9)
            self.assertAlmostEqual(row[2], single["home_win"], places=9)

    def test_simulation_is_deterministic_and_normalized(self):
        matches = load_matches("data/sample_matches.csv")
        model = train_model(matches)
        fixtures = [
            {"home_team": "Argentina", "away_team": "France", "neutral": True},
            {"home_team": "Brazil", "away_team": "Croatia", "neutral": True},
        ]
        first = simulate_fixtures(model, fixtures, runs=500, seed=11)
        second = simulate_fixtures(model, fixtures, runs=500, seed=11)
        self.assertEqual(first, second)
        for row in first:
            total = row["home_win"] + row["draw"] + row["away_win"]
            self.assertAlmostEqual(total, 1.0, places=9)

    def test_match_rows_filters_and_limits(self):
        matches = sorted(
            load_matches("data/sample_matches.csv"),
            key=lambda match: match.date,
            reverse=True,
        )
        rows = match_rows(matches, limit=5)
        self.assertEqual(len(rows), 5)
        dates = [row["date"] for row in rows]
        self.assertEqual(dates, sorted(dates, reverse=True))
        team_rows = match_rows(matches, team="Argentina", limit=100)
        for row in team_rows:
            self.assertIn("Argentina", {row["home_team"], row["away_team"]})

    def test_model_roundtrip_and_schema_guard(self):
        matches = load_matches("data/sample_matches.csv")
        model = train_model(matches)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            save_model(model, path)
            loaded = load_model(path)
            self.assertEqual(loaded.feature_names, model.feature_names)

            stale = train_model(matches)
            stale.schema_version = 1
            stale_path = Path(tmp) / "stale.pkl"
            save_model(stale, stale_path)
            with self.assertRaises(ValueError):
                load_model(stale_path)

    def test_international_results_importer(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            results_csv = tmp_path / "results.csv"
            output_csv = tmp_path / "matches.csv"
            results_csv.write_text(
                "\n".join(
                    [
                        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral",
                        "2022-12-18,Argentina,France,3,3,FIFA World Cup,Lusail,Qatar,TRUE",
                        "2023-03-23,Germany,Peru,2,0,Friendly,Mainz,Germany,FALSE",
                        "2026-06-11,Mexico,TBD,,,FIFA World Cup,Mexico City,Mexico,FALSE",
                    ]
                ),
                encoding="utf-8",
            )
            count = convert_international_results(results_csv, output_csv)
            imported = load_matches(output_csv)
            self.assertEqual(count, 2)
            self.assertEqual(imported[0].tournament, "FIFA World Cup")
            self.assertTrue(imported[0].neutral)
            self.assertFalse(imported[1].neutral)

    def test_kaggle_importer(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matches_csv = tmp_path / "WorldCupMatches.csv"
            cups_csv = tmp_path / "WorldCups.csv"
            output_csv = tmp_path / "world_cup_matches.csv"

            matches_csv.write_text(
                "\n".join(
                    [
                        "Year,Datetime,Stage,Stadium,City,Home Team Name,Home Team Goals,Away Team Goals,Away Team Name",
                        "2022,18 Dec 2022 - 18:00,Final,Lusail,Lusail,Argentina,3,3,France",
                        "2022,20 Nov 2022 - 19:00,Group A,Al Bayt,Al Khor,Qatar,0,2,Ecuador",
                    ]
                ),
                encoding="utf-8",
            )
            cups_csv.write_text("Year,Country\n2022,Qatar\n", encoding="utf-8")

            count = convert_world_cup_matches(matches_csv, output_csv, cups_csv)
            imported = load_matches(output_csv)

            self.assertEqual(count, 2)
            self.assertEqual(imported[0].home_team, "Qatar")
            self.assertFalse(imported[0].neutral)
            self.assertEqual(imported[1].home_team, "Argentina")
            self.assertTrue(imported[1].neutral)


if __name__ == "__main__":
    unittest.main()
