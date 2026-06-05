import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wc_forecast.data import load_matches
from wc_forecast.kaggle import convert_world_cup_matches
from wc_forecast.models import backtest, train_model


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
