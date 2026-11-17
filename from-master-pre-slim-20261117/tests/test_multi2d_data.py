from pathlib import Path

import pandas as pd

from molgap.multi2d_data import parse_family_values, parse_source_patterns, validate_labels


def test_parse_family_values() -> None:
    assert parse_family_values(["general=3", "hard=2"]) == {"general": 3, "hard": 2}
    assert parse_family_values(["general=*.csv"], value_type=str) == {"general": "*.csv"}
    assert parse_source_patterns(["broad=a/*.csv", "broad=b/*.csv"]) == {
        "broad": ["a/*.csv", "b/*.csv"]
    }


def test_validate_labels() -> None:
    frame = pd.DataFrame(
        {
            "cid": [1],
            "smiles": ["CC"],
            "canonical_smiles": ["CC"],
            "homo": [-5.0],
            "lumo": [-1.0],
            "gap": [4.0],
        }
    )
    assert validate_labels(frame, str(Path("fixture.csv"))) == 0.0
