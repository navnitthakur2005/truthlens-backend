"""Generate a SYNTHETIC dataset in WELFake's schema, only to smoke-test the pipeline code.

It is NOT real news and any model trained on it is meaningless. Artifacts built from it are marked
``smoke_test: true`` and the UI shows a warning. Includes deliberately dirty rows (missing values,
duplicates, bad labels) so the cleaning code is exercised.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent

REAL_T = ["Officials announce {x} budget review for {y}", "Committee publishes report on {x} in {y}",
          "Agency confirms schedule for {x} programme in {y}", "Researchers release data on {x} across {y}"]
REAL_B = ["The ministry said on Tuesday that the review would cover {x} and be published after consultation.",
          "According to the statement, the figures were compiled by the statistics office and verified independently.",
          "A spokesperson told reporters that the schedule remains subject to approval by the council.",
          "The report, released after a six month study, cites data from regional departments."]
FAKE_T = ["SHOCKING: they are hiding the truth about {x} in {y}", "You won't BELIEVE what {x} did in {y}!!!",
          "Secret plot EXPOSED: {x} behind massive cover-up in {y}", "BREAKING!!! {x} miracle cure banned in {y}"]
FAKE_B = ["Share this before they delete it! Wake up people, the mainstream media will never tell you this.",
          "Anonymous insiders reveal the shocking truth that elites do not want you to know!!!",
          "This one weird trick is being censored. Doctors hate it. Spread the word now before it is too late.",
          "Everything you were told is a lie and the globalist agenda is finally exposed."]
X = ["energy", "health", "transport", "education", "water", "housing", "tax", "trade"]
Y = ["the capital", "the northern region", "the coastal districts", "three states", "rural areas"]


def make(n_per_class=700, seed=7) -> pd.DataFrame:
    rnd = random.Random(seed)
    rows = []
    for label, T, B in ((0, REAL_T, REAL_B), (1, FAKE_T, FAKE_B)):
        for i in range(n_per_class):
            x, y = rnd.choice(X), rnd.choice(Y)
            body = " ".join(rnd.sample(B, 3)).format(x=x, y=y) + f" Reference {rnd.randint(1, 10**6)}."
            rows.append({"title": rnd.choice(T).format(x=x, y=y), "text": body, "label": label})
    # dirty rows
    rows += [{"title": None, "text": rows[0]["text"] + " variant", "label": 0}]          # missing title
    rows += [{"title": "No body", "text": None, "label": 1}]                                # missing text (too short)
    rows += [{"title": None, "text": None, "label": 0}]                                     # empty
    rows += [dict(rows[5])] * 3                                                              # exact duplicates
    rows += [{"title": rows[7]["title"], "text": rows[7]["text"], "label": 1 - rows[7]["label"]}]  # conflict
    rows += [{"title": "Bad label row for cleaning test", "text": "x" * 80, "label": "maybe"}]
    rnd.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BACKEND / "data" / "smoke" / "smoke.csv")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(a.out, index=False)
    print(f"Wrote SYNTHETIC smoke-test data to {a.out}")
