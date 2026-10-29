"""Data cleaning and label normalisation for WELFake-style datasets.

Output schema: ``title`` (str), ``text`` (str), ``label`` (int: 0 = real, 1 = fake).

Label mapping
-------------
Our internal convention is **1 = fake, 0 = real**. The WELFake documentation describes its own
column as ``0 = fake, 1 = real``; several independent analyses report that the *actual* data is the
other way round. Because we cannot silently guess, the mapping is an explicit, recorded parameter
(``source_fake_value``) and ``training/check_labels.py`` prints the most discriminative tokens so a
human can verify the direction before trusting the results.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import pandas as pd

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

LABEL_TEXT = {0: "real", 1: "fake"}


@dataclass
class CleaningReport:
    rows_before: int = 0
    dropped_invalid_label: int = 0
    dropped_empty_text: int = 0
    dropped_too_short: int = 0
    dropped_conflicting_duplicates: int = 0
    dropped_duplicates: int = 0
    dropped_duplicate_text: int = 0
    rows_after: int = 0
    missing_title_before: int = 0
    missing_text_before: int = 0
    label_counts_after: dict = field(default_factory=dict)
    source_fake_value: int = 1
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def clean_str(value) -> str:
    """Normalise whitespace / control characters; NaN -> empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = _CTRL.sub(" ", str(value))
    return _WS.sub(" ", s).strip()


def dedup_key(title: str, text: str) -> str:
    return _WS.sub(" ", f"{title} {text}".lower()).strip()


def normalize_label(value, source_fake_value: int = 1):
    """Map a raw source label to 1 (fake) / 0 (real). Returns ``None`` when unrecognised.

    ``source_fake_value`` says which *numeric* source value means "fake".
    """
    if source_fake_value not in (0, 1):
        raise ValueError("source_fake_value must be 0 or 1")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s in ("0", "0.0", "1", "1.0"):
        raw = int(float(s))
        return 1 if raw == source_fake_value else 0
    if s in ("fake", "false", "unreliable"):
        return 1
    if s in ("real", "true", "reliable"):
        return 0
    return None


def clean_dataframe(
    df: pd.DataFrame, source_fake_value: int = 1, min_chars: int = 20
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a raw dataframe. Never mutates the input."""
    report = CleaningReport(rows_before=len(df), source_fake_value=source_fake_value)

    cols = {c.lower().strip(): c for c in df.columns}
    if "label" not in cols or ("title" not in cols and "text" not in cols):
        raise ValueError(f"Dataset needs a 'label' column and 'title' and/or 'text'; found {list(df.columns)}")

    out = pd.DataFrame(
        {
            "title": df[cols["title"]] if "title" in cols else "",
            "text": df[cols["text"]] if "text" in cols else "",
            "label_raw": df[cols["label"]],
        }
    )
    out["title"] = out["title"].map(clean_str)
    out["text"] = out["text"].map(clean_str)
    report.missing_title_before = int((out["title"] == "").sum())
    report.missing_text_before = int((out["text"] == "").sum())

    out["label"] = out["label_raw"].map(lambda v: normalize_label(v, source_fake_value))
    n = len(out)
    out = out[out["label"].notna()].copy()
    report.dropped_invalid_label = n - len(out)
    out["label"] = out["label"].astype(int)

    n = len(out)
    out = out[(out["title"] != "") | (out["text"] != "")]
    report.dropped_empty_text = n - len(out)
    n = len(out)
    out = out[(out["title"].str.len() + out["text"].str.len()) >= min_chars]
    report.dropped_too_short = n - len(out)

    # identical content with different labels is unusable: drop all copies
    out["_key"] = [dedup_key(t, x) for t, x in zip(out["title"], out["text"])]
    n = len(out)
    nunique = out.groupby("_key")["label"].transform("nunique")
    out = out[nunique == 1]
    report.dropped_conflicting_duplicates = n - len(out)

    n = len(out)
    out = out.drop_duplicates(subset="_key", keep="first")
    report.dropped_duplicates = n - len(out)

    # same article body under a different headline would leak across splits: keep the first
    n = len(out)
    long_text = out["text"].str.len() >= 50
    dup_text = out["text"].str.lower().duplicated(keep="first") & long_text
    out = out[~dup_text]
    report.dropped_duplicate_text = n - len(out)

    out = out[["title", "text", "label"]].reset_index(drop=True)
    report.rows_after = len(out)
    report.label_counts_after = {
        LABEL_TEXT[int(k)]: int(v) for k, v in out["label"].value_counts().sort_index().items()
    }
    if report.rows_after == 0:
        report.notes.append("No rows left after cleaning.")
    return out, report


def build_text(title: str, text: str, mode: str) -> str:
    """The single place where input mode -> model input string is decided."""
    title, text = clean_str(title), clean_str(text)
    if mode == "headline":
        return title
    if mode == "article":
        return text
    if mode == "both":
        return f"{title} . {text}".strip(" .") if (title or text) else ""
    raise ValueError(f"Unknown mode: {mode}")
