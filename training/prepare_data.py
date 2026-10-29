"""Clean the raw dataset and write stratified train/val/test splits.

    python -m training.prepare_data                       # real WELFake in data/WELFake_Dataset.csv
    python -m training.prepare_data --input X.csv --out-dir data/processed_x

Outputs (in --out-dir): clean.csv, train.csv, val.csv, test.csv, cleaning_report.json
Splitting happens AFTER de-duplication so identical records cannot appear in two splits.
Learned preprocessing (TF-IDF) is fitted later, on the training split only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.cleaning import clean_dataframe

BACKEND = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BACKEND / "data" / "WELFake_Dataset.csv"
DEFAULT_OUT = BACKEND / "data" / "processed"


def split(df: pd.DataFrame, val_size=0.15, test_size=0.15, seed=42):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def assert_no_leakage(train, val, test) -> None:
    from app.ml.cleaning import dedup_key

    keys = [set(dedup_key(t, x) for t, x in zip(d["title"], d["text"])) for d in (train, val, test)]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise AssertionError("Duplicate records found across splits")


def run(input_path: Path, out_dir: Path, source_fake_value: int, seed: int = 42) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run `python -m training.download_dataset` for instructions."
        )
    raw = pd.read_csv(input_path)
    clean, report = clean_dataframe(raw, source_fake_value=source_fake_value)
    if report.rows_after < 100 or clean["label"].nunique() < 2:
        raise ValueError(f"Too few usable rows after cleaning: {report.to_dict()}")
    train, val, test = split(clean, seed=seed)
    assert_no_leakage(train, val, test)

    out_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_dir / "clean.csv", index=False)
    for name, d in (("train", train), ("val", val), ("test", test)):
        d.to_csv(out_dir / f"{name}.csv", index=False)
    info = report.to_dict()
    info["input_file"] = str(input_path)
    info["splits"] = {n: int(len(d)) for n, d in (("train", train), ("val", val), ("test", test))}
    info["label_mapping"] = {"0": "real", "1": "fake"}
    info["seed"] = seed
    (out_dir / "cleaning_report.json").write_text(json.dumps(info, indent=2))
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--source-fake-value",
        type=int,
        choices=[0, 1],
        default=1,
        help="Which numeric value in the SOURCE label column means 'fake' (see app/ml/cleaning.py).",
    )
    args = ap.parse_args(argv)
    try:
        info = run(args.input, args.out_dir, args.source_fake_value)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}")
        return 2
    print(f"Rows before cleaning : {info['rows_before']}")
    for k in (
        "dropped_invalid_label",
        "dropped_empty_text",
        "dropped_too_short",
        "dropped_conflicting_duplicates",
        "dropped_duplicates",
        "dropped_duplicate_text",
    ):
        print(f"  {k:32s}: {info[k]}")
    print(f"Rows after cleaning  : {info['rows_after']}  {info['label_counts_after']}")
    print(f"Splits               : {info['splits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
