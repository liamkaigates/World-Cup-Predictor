import io
import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from wc_forecast.download import (
    CredentialsError,
    download_dataset,
    extract_csvs,
    load_credentials,
)


def build_zip(path: Path, members: dict) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


class DownloadTest(unittest.TestCase):
    def test_credentials_from_environment(self):
        env = {"KAGGLE_USERNAME": "someone", "KAGGLE_KEY": "secret"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(load_credentials(), ("someone", "secret"))

    def test_credentials_from_kaggle_json(self):
        with TemporaryDirectory() as tmp:
            config = Path(tmp) / "kaggle.json"
            config.write_text(json.dumps({"username": "someone", "key": "secret"}), encoding="utf-8")
            with patch.dict("os.environ", {"KAGGLE_CONFIG_DIR": tmp}, clear=True):
                self.assertEqual(load_credentials(), ("someone", "secret"))

    def test_missing_credentials_raise(self):
        with TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"KAGGLE_CONFIG_DIR": tmp}, clear=True):
                with self.assertRaises(CredentialsError):
                    load_credentials()

    def test_extract_csvs_flattens_and_filters(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "dataset.zip"
            build_zip(
                zip_path,
                {
                    "WorldCupMatches.csv": "a,b\n1,2\n",
                    "nested/WorldCups.csv": "c,d\n3,4\n",
                    "notes.txt": "ignore me",
                    "folder/": "",
                },
            )
            out_dir = tmp_path / "out"
            out_dir.mkdir()
            extracted = extract_csvs(zip_path, out_dir)
            self.assertEqual(sorted(extracted), ["WorldCupMatches.csv", "WorldCups.csv"])
            self.assertEqual((out_dir / "WorldCups.csv").read_text(encoding="utf-8"), "c,d\n3,4\n")
            self.assertFalse((out_dir / "notes.txt").exists())

    def test_extract_csvs_rejects_empty_archive(self):
        with TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "dataset.zip"
            build_zip(zip_path, {"notes.txt": "no csvs here"})
            with self.assertRaises(ValueError):
                extract_csvs(zip_path, Path(tmp))

    def test_download_dataset_extracts_fetched_zip(self):
        def fake_fetch(dataset, destination, timeout=60.0):
            build_zip(destination, {"WorldCupMatches.csv": "a,b\n1,2\n", "WorldCups.csv": "c,d\n3,4\n"})

        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "kaggle"
            with patch("wc_forecast.download.fetch_dataset_zip", side_effect=fake_fetch):
                files = download_dataset("abecklas/fifa-world-cup", data_dir)
            self.assertEqual(sorted(files), ["WorldCupMatches.csv", "WorldCups.csv"])
            self.assertTrue((data_dir / "WorldCupMatches.csv").is_file())
            leftovers = list(data_dir.glob("*.zip"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
