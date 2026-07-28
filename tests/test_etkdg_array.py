import gzip
import hashlib
import json

import numpy as np

from molgap.etkdg_array import build_secondary_raw_shard


def test_raw_secondary_shard_is_aligned_and_resumable(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_path = input_dir / "rows_0000000_0000003.csv.gz"
    with gzip.open(input_path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("source_idx,canonical_smiles,homo,lumo,gap\n")
        handle.write("0,CCO,-5.0,-1.0,4.0\n")
        handle.write("1,CCN,-5.1,-1.1,4.0\n")
        handle.write("2,CCC,-5.2,-1.2,4.0\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "molgap-secondary-etkdg-array-v1",
                "seed": 314159,
                "source_csv_sha256": "a" * 64,
                "primary_acceptance_sha256": "b" * 64,
                "shards": [
                    {
                        "shard_index": 0,
                        "start": 0,
                        "stop": 3,
                        "input_path": "inputs/rows_0000000_0000003.csv.gz",
                        "input_sha256": hashlib.sha256(
                            input_path.read_bytes()
                        ).hexdigest(),
                        "output_path": "graphs_0000000_0000003.pt",
                        "primary_graphs": 2,
                        "primary_failure_source_idx": [1],
                        "primary_graph_sha256": "c" * 64,
                        "primary_sidecar_sha256": "d" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    first = build_secondary_raw_shard(
        manifest_path=manifest_path,
        output_dir=output_dir,
        shard_index=0,
        workers=1,
    )
    second = build_secondary_raw_shard(
        manifest_path=manifest_path,
        output_dir=output_dir,
        shard_index=0,
        workers=1,
    )
    assert first == second
    assert first["requested"] == first["built"] == 2
    with np.load(output_dir / "graphs_0000000_0000003.npz") as arrays:
        assert arrays["source_idx"].tolist() == [0, 2]
        assert arrays["y"].shape == (2, 3)
        assert arrays["atom_ptr"][-1] == len(arrays["z"])
