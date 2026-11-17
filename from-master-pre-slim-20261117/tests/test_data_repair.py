from pathlib import Path

import numpy as np
import pandas as pd

from molgap.data_repair import SourceSpec, enrich_chunk, infer_source_group


def test_infer_source_group() -> None:
    assert infer_source_group(Path("phase8_2m_round08_conjugated_da.csv")) == (
        "complementary",
        "conjugated_da",
    )
    assert infer_source_group(Path("phase8_2m_round04_very_large.csv")) == (
        "hard",
        "very_large",
    )


def test_enrich_chunk_quality_and_buckets() -> None:
    frame = pd.DataFrame(
        {
            "cid": ["1", "2"],
            "mw": [78.1, 40.0],
            "formula": ["C6H6", "CH4.He"],
            "smiles": ["c1ccccc1", "[He].C"],
            "canonical_smiles": ["c1ccccc1", "[He].C"],
            "homo": [-6.0, -5.0],
            "lumo": [-1.0, -1.0],
            "gap": [5.0, 4.0],
        }
    )
    spec = SourceSpec("test", "test.csv", "core", "test", immutable=True)
    result = enrich_chunk(frame, spec, row_offset=10, workers=1)
    assert result["source_row"].tolist() == [10, 11]
    assert bool(result.loc[0, "quality_ok"])
    assert not bool(result.loc[1, "quality_ok"])
    assert result.loc[0, "scaffold"] == "c1ccccc1"
    assert np.isclose(result.loc[0, "gap_identity_error"], 0.0)
