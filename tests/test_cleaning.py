import numpy as np
import pandas as pd
import pytest

from app.ml.cleaning import build_text, clean_dataframe, clean_str, normalize_label


def test_normalize_label_both_mappings():
    assert normalize_label(1, source_fake_value=1) == 1
    assert normalize_label(0, source_fake_value=1) == 0
    assert normalize_label(1, source_fake_value=0) == 0
    assert normalize_label("0", source_fake_value=0) == 1
    assert normalize_label("fake") == 1 and normalize_label("REAL") == 0
    for bad in (None, np.nan, "maybe", "2", ""):
        assert normalize_label(bad) is None
    with pytest.raises(ValueError):
        normalize_label(1, source_fake_value=2)


def test_clean_str_handles_nan_and_whitespace():
    assert clean_str(np.nan) == "" and clean_str(None) == ""
    assert clean_str("  a \n\t b\x00c ") == "a b c"


def _raw():
    return pd.DataFrame({
        "Unnamed: 0": range(9),
        "title": ["Good headline one", None, "Dup", "Dup", "Conflict title here", "Conflict title here", "x", "Valid headline two", "Bad label"],
        "text": ["Body text long enough to survive cleaning. " * 2, "Body without a title but long enough here.", "Same body " * 5,
                 "Same body " * 5, "Conflicting body text " * 4, "Conflicting body text " * 4, None, "Another body text that is fine. " * 2, "z" * 60],
        "label": [0, 1, 1, 1, 0, 1, 0, 1, "??"],
    })


def test_clean_dataframe_counts_and_schema():
    raw = _raw()
    out, rep = clean_dataframe(raw)
    assert list(out.columns) == ["title", "text", "label"]
    assert rep.rows_before == 9
    assert rep.dropped_invalid_label == 1
    assert rep.dropped_too_short == 1              # "x" with empty text
    assert rep.dropped_conflicting_duplicates == 2  # both conflicting copies removed
    assert rep.dropped_duplicates == 1
    assert rep.rows_after == len(out) == 4
    assert set(out["label"]) <= {0, 1}
    assert rep.label_counts_after["real"] + rep.label_counts_after["fake"] == rep.rows_after
    assert len(raw) == 9  # input not mutated


def test_clean_dataframe_label_direction_flips():
    raw = pd.DataFrame({"title": ["A valid headline here"], "text": ["body " * 20], "label": [1]})
    assert clean_dataframe(raw, source_fake_value=1)[0]["label"].iloc[0] == 1
    assert clean_dataframe(raw, source_fake_value=0)[0]["label"].iloc[0] == 0


def test_clean_dataframe_missing_columns():
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame({"foo": [1]}))


def test_build_text_modes():
    assert build_text("T", "Body", "headline") == "T"
    assert build_text("T", "Body", "article") == "Body"
    assert build_text("T", "Body", "both") == "T . Body"
    assert build_text("", "Body", "both") == "Body"
    with pytest.raises(ValueError):
        build_text("a", "b", "nope")
