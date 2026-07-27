import csv

import pytest

from molgap.ensemble_evaluation import evaluate_equal_ensemble


def _write(path, rows, pcqm=False):
    fields = (
        ["cid", "smiles", "gap", "model_gap"]
        if pcqm
        else [
            "eval_set", "cid", "smiles", "homo", "lumo", "gap",
            "model_homo", "model_lumo", "model_gap",
        ]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{field: row[field] for field in fields} for row in rows]
        )


def test_equal_ensemble_aligns_and_averages(tmp_path):
    base = {
        "eval_set": "ood1000", "cid": "1", "smiles": "C",
        "homo": 0, "lumo": 1, "gap": 1,
    }
    common = []
    pcqm = []
    for seed, delta in enumerate((1.0, -1.0)):
        c = tmp_path / f"c{seed}.csv"
        p = tmp_path / f"p{seed}.csv"
        predicted = {
            **base,
            "model_homo": delta,
            "model_lumo": 1 + delta,
            "model_gap": 1 + delta,
        }
        _write(c, [predicted])
        pcqm_row = dict(predicted)
        pcqm_row.pop("eval_set")
        _write(p, [pcqm_row], pcqm=True)
        common.append(c)
        pcqm.append(p)
    result = evaluate_equal_ensemble(common, pcqm, model_hint="model")
    assert result["common"]["common"]["metrics"]["average"]["mae_eV"] == 0
    assert result["pcqm"]["gap"]["mae_eV"] == 0
    assert result["compute"]["gps7_encoder_passes_per_molecule"] == 2


def test_equal_ensemble_rejects_identity_mismatch(tmp_path):
    row = {
        "eval_set": "ood1000", "cid": "1", "smiles": "C",
        "homo": 0, "lumo": 1, "gap": 1,
        "model_homo": 0, "model_lumo": 1, "model_gap": 1,
    }
    paths = []
    pcqm = []
    for index in range(2):
        c = tmp_path / f"c{index}.csv"
        p = tmp_path / f"p{index}.csv"
        changed = dict(row)
        changed["cid"] = str(index + 1)
        _write(c, [changed])
        changed.pop("eval_set")
        _write(p, [changed], pcqm=True)
        paths.append(c)
        pcqm.append(p)
    with pytest.raises(ValueError, match="not aligned"):
        evaluate_equal_ensemble(paths, pcqm, model_hint="model")
