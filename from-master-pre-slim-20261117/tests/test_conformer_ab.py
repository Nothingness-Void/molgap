import json

import torch

from molgap.conformer_ab import analyze_tradeoff, select_aligned_rows


def test_select_aligned_rows_is_role_stratified(tmp_path):
    split = tmp_path / "split.csv"
    split.write_text(
        "source_idx,split,cid,smiles,canonical_smiles,homo,lumo,gap\n"
        "0,train,10,C,C,-1,1,2\n"
        "1,train,11,CC,CC,-2,1,3\n"
        "2,validation,12,CCC,CCC,-3,1,4\n"
        "3,test,13,CCCC,CCCC,-4,1,5\n",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.pt"
    torch.save(
        {
            role: {"source_idx": torch.tensor(indices)}
            for role, indices in {
                "train": [0, 1],
                "validation": [2],
                "test": [3],
            }.items()
        },
        payload,
    )

    rows = select_aligned_rows(
        split, payload, {"train": 2, "validation": 1, "test": 1}
    )

    assert [row["source_idx"] for row in rows] == [0, 1, 2, 3]
    assert [row["role"] for row in rows] == [
        "train",
        "train",
        "validation",
        "test",
    ]


def test_tradeoff_reports_time_and_accuracy_delta(tmp_path):
    builds = {
        "bare_etkdg": {"wall_s": 100.0, "succeeded": 50_000},
        "etkdgv3_mmff200": {"wall_s": 300.0, "succeeded": 50_000},
    }
    evaluation = {
        "protocols": {
            "bare_etkdg": {
                "route_b_equal_seed_ensemble": {
                    "homo": 0.2,
                    "lumo": 0.2,
                    "gap": 0.3,
                    "average": 0.233333,
                }
            },
            "etkdgv3_mmff200": {
                "route_b_equal_seed_ensemble": {
                    "homo": 0.19,
                    "lumo": 0.18,
                    "gap": 0.27,
                    "average": 0.213333,
                }
            },
        }
    }
    output = tmp_path / "tradeoff.json"

    result = analyze_tradeoff(builds, evaluation, output)

    assert result["speed"]["mmff_to_bare_ratio"] == 3.0
    assert result["accuracy"]["mmff_minus_bare_mae_eV"]["gap"] < 0
    assert json.loads(output.read_text(encoding="utf-8")) == result
