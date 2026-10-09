"""Bind the predeclared diagnostic to accepted artifacts; no model execution."""
import argparse
import json
from pathlib import Path

from molgap.k1_bn_diagnostic import ARMS, SETTINGS, validate_inputs
from molgap.training_reproducibility import atomic_json, sha256_file


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=here / "inputs.json")
    args = parser.parse_args()
    root = here.parents[2]
    question = here.parent
    package = root / "platforms/_records/kaggle/staging/pcqm_k1_consistency_ablation_500k/kaggle3_v1/package"
    frozen = package.parent / "frozen_source"
    stages = question / "submission_kaggle3_v1/terminal_inspection_20261009/evidence/stages"

    def bind(path):
        return {"path": path.as_posix(), "sha256": sha256_file(path)}

    inputs = {
        "format": "molgap-k1-clean-bn-pair-v1", "settings": SETTINGS,
        "protocol": bind(question / "protocol.md"),
        "source_archive": bind(package / "source.tar.gz"),
        "source_inventory": bind(package / "SOURCE_FILES.json"),
        "frozen_source_root": frozen.as_posix(),
        "cache_root": "D:/文档/molgap-exp/molgap-500k-v4-evidence/data/cache/pcqm4mv2_500k_v4",
        "manifest_sha256": "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751",
        "executed_source_files": {
            (root / name).as_posix(): sha256_file(root / name) for name in (
                "src/molgap/k1_bn_diagnostic.py", "src/molgap/k1_frozen_inference.py",
                "src/molgap/k1_bn_calibration.py",
                "experiments/pcqm_k1_consistency_ablation_500k/clean_bn/run.py",
                "experiments/pcqm_k1_consistency_ablation_500k/clean_bn/prepare.py",
                "experiments/pcqm_k1_consistency_ablation_500k/clean_bn/AUTHORIZATION.md",
            )
        },
        "arms": {},
    }
    for arm in ARMS:
        trajectory = question / "kaggle1_v1" / arm / "trajectory.json"
        inputs["arms"][arm] = {
            "trajectory": bind(trajectory),
            "trajectory_id": json.loads(trajectory.read_text(encoding="utf-8"))["trajectory_id"],
            "checkpoint": bind(stages / arm / "best_model.pt"),
            "saved_predictions": bind(stages / arm / "best_predictions.pt"),
        }
    validate_inputs(inputs)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Frozen diagnostic inputs already exist")
    atomic_json(output, inputs)
    print(json.dumps({"inputs": str(output), "sha256": sha256_file(output)}))


if __name__ == "__main__":
    main()
