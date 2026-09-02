import json
import threading
import unittest
import urllib.error
import urllib.request

from wc_forecast.api import build_server
from wc_forecast.data import load_matches
from wc_forecast.models import train_model


class ApiEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        matches = load_matches("data/sample_matches.csv")
        model = train_model(matches)
        cls.server = build_server(model, matches, static_dir="static", host="127.0.0.1", port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get_json(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as response:
            return response.status, json.loads(response.read())

    def test_health(self):
        status, payload = self.get_json("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})

    def test_predict(self):
        status, payload = self.get_json("/api/predict?home=Argentina&away=France&neutral=true")
        self.assertEqual(status, 200)
        self.assertEqual(set(payload["probabilities"]), {"away_win", "draw", "home_win"})
        self.assertAlmostEqual(sum(payload["probabilities"].values()), 1.0, places=6)

    def test_predict_requires_teams(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get_json("/api/predict?home=Argentina")
        self.assertEqual(ctx.exception.code, 400)

    def test_matches_rejects_bad_limit(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get_json("/api/matches?limit=notanumber")
        self.assertEqual(ctx.exception.code, 400)

    def test_matches_and_summary(self):
        status, payload = self.get_json("/api/matches?team=Argentina&limit=3")
        self.assertEqual(status, 200)
        self.assertLessEqual(len(payload["matches"]), 3)
        status, summary = self.get_json("/api/summary")
        self.assertEqual(status, 200)
        self.assertGreater(summary["match_count"], 0)

    def test_dashboard_and_static(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/html", response.headers["Content-Type"])

    def test_static_traversal_blocked(self):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/static/../data/sample_matches.csv")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(request)
        self.assertIn(ctx.exception.code, {400, 404})


if __name__ == "__main__":
    unittest.main()
