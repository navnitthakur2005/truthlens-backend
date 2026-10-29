"""Sanity-check label direction: print tokens most associated with each class.

If 'real' tokens look like unreliable/sensational language (or vice versa), re-run
prepare_data.py with --source-fake-value flipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BACKEND = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=BACKEND / "data" / "processed")
    ap.add_argument("-k", type=int, default=25)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.data_dir / "train.csv", keep_default_na=False)
    X = (df["title"] + " . " + df["text"]).tolist()
    vec = TfidfVectorizer(stop_words="english", min_df=5, max_features=50000, sublinear_tf=True)
    M = vec.fit_transform(X)
    lr = LogisticRegression(max_iter=500).fit(M, df["label"])
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    print("Top tokens for label=0 (REAL in our convention):", ", ".join(names[order[: args.k]]))
    print("Top tokens for label=1 (FAKE in our convention):", ", ".join(names[order[-args.k :]]))
    print("\nVerify these look right (real: wire-service / agency style; fake: sensational / partisan).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
