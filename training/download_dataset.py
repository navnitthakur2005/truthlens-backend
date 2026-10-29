"""Locate (or try to download) the WELFake dataset.

Usage:  python -m training.download_dataset [--url URL]

WELFake is hosted on Kaggle and Zenodo, which usually need a browser or credentials. This script:
(1) checks data/WELFake_Dataset.csv, (2) downloads from a URL you supply (--url or WELFAKE_URL),
(3) otherwise exits non-zero with exact manual instructions. It never fabricates data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = DATA_DIR / "WELFake_Dataset.csv"

INSTRUCTIONS = f"""
WELFake dataset not found at: {TARGET}

Get it in one of these ways, then re-run training:
  1. Kaggle:  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
       kaggle datasets download -d saurabhshahane/fake-news-classification -p backend/data --unzip
  2. Zenodo:  https://zenodo.org/records/4561253  (download WELFake_Dataset.csv)
  3. Any mirror you trust:  python -m training.download_dataset --url <direct-csv-or-zip-url>
Place the file at backend/data/WELFake_Dataset.csv (columns: title, text, label).
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("WELFAKE_URL"))
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 1000:
        print(f"Found dataset: {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
        return 0
    if not args.url:
        print(INSTRUCTIONS)
        return 2
    try:
        print(f"Downloading {args.url} ...")
        tmp = TARGET.with_suffix(".download")
        with requests.get(args.url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        if args.url.lower().endswith(".zip"):
            import zipfile

            with zipfile.ZipFile(tmp) as z:
                name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
                with z.open(name) as src, open(TARGET, "wb") as dst:
                    dst.write(src.read())
            tmp.unlink()
        else:
            tmp.replace(TARGET)
        print(f"Saved {TARGET}")
        return 0
    except Exception as exc:  # network blocked, 403, bad zip...
        print(f"Download failed: {type(exc).__name__}: {exc}")
        print(INSTRUCTIONS)
        return 3


if __name__ == "__main__":
    sys.exit(main())
