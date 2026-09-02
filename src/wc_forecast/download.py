from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"
DEFAULT_DATASET = "abecklas/fifa-world-cup"
INTERNATIONAL_DATASET = "martj42/international-football-results-from-1872-to-2017"


class CredentialsError(RuntimeError):
    """Raised when no Kaggle API credentials can be found."""


def _config_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR", "") or Path.home() / ".kaggle")


def load_access_token() -> Optional[str]:
    """Return a Kaggle access token (KGAT_...) from KAGGLE_API_TOKEN or
    $KAGGLE_CONFIG_DIR/access_token, or None when neither is configured."""
    token = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token
    token_path = _config_dir() / "access_token"
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def load_credentials() -> Tuple[str, str]:
    """Return (username, key) from KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    config_path = _config_dir() / "kaggle.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        username = payload.get("username")
        key = payload.get("key")
        if username and key:
            return username, key

    raise CredentialsError(
        "Kaggle API credentials not found. Either save an access token to ~/.kaggle/access_token "
        "(or set KAGGLE_API_TOKEN), or set KAGGLE_USERNAME and KAGGLE_KEY, or save an API token "
        "to ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token)."
    )


def auth_header() -> str:
    """Prefer a bearer access token; fall back to basic auth with username/key."""
    token = load_access_token()
    if token:
        return f"Bearer {token}"
    username, key = load_credentials()
    basic = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
    return f"Basic {basic}"


class _CrossHostAuthStripper(urllib.request.HTTPRedirectHandler):
    """Drop the Authorization header when Kaggle redirects to signed storage URLs,
    which reject requests carrying a second authentication mechanism."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urlparse(newurl).hostname != urlparse(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def fetch_dataset_zip(dataset: str, destination: Path, timeout: float = 60.0) -> None:
    request = urllib.request.Request(
        f"{KAGGLE_API_BASE}/datasets/download/{dataset}",
        headers={"Authorization": auth_header(), "User-Agent": "wc-forecast"},
    )
    opener = urllib.request.build_opener(_CrossHostAuthStripper)
    with opener.open(request, timeout=timeout) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_csvs(zip_path: Path, data_dir: Path) -> List[str]:
    """Extract every CSV in the archive into data_dir, flattened to its basename."""
    extracted = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if member.endswith("/") or not name.lower().endswith(".csv"):
                continue
            target = data_dir / name
            with archive.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted.append(name)
    if not extracted:
        raise ValueError(f"No CSV files found in the downloaded archive {zip_path}")
    return extracted


def download_dataset(
    dataset: str = DEFAULT_DATASET,
    data_dir: str | Path = "data/kaggle",
    timeout: float = 60.0,
) -> List[str]:
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", dir=directory, delete=False) as handle:
        zip_path = Path(handle.name)
    try:
        fetch_dataset_zip(dataset, zip_path, timeout=timeout)
        return extract_csvs(zip_path, directory)
    finally:
        zip_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle FIFA World Cup dataset with the Kaggle API")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug (owner/name)")
    parser.add_argument("--data-dir", default="data/kaggle", help="Directory to place the extracted CSVs in")
    parser.add_argument("--force", action="store_true", help="Re-download even if the expected CSVs already exist")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    directory = Path(args.data_dir)
    if not args.force and any(directory.glob("*.csv")):
        print(f"Dataset already present in {args.data_dir}; use --force to re-download")
        return

    try:
        files = download_dataset(args.dataset, args.data_dir, timeout=args.timeout)
    except CredentialsError as error:
        raise SystemExit(str(error))
    except urllib.error.HTTPError as error:
        if error.code in {401, 403}:
            raise SystemExit(
                f"Kaggle rejected the request ({error.code}). Check that the API key is valid and "
                f"that your account has accepted the terms for {args.dataset} on kaggle.com."
            )
        raise SystemExit(f"Download failed with HTTP {error.code}: {error.reason}")

    print(f"Downloaded {len(files)} files to {args.data_dir}: {', '.join(sorted(files))}")


if __name__ == "__main__":
    main()
