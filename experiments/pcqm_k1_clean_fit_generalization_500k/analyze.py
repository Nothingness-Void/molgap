"""Analyze saved predictions only; no checkpoints, model or cache execution."""
from pathlib import Path
from molgap.k1_clean_fit_diagnostic import analyze
from molgap.training_reproducibility import atomic_json, sha256_file

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    result = analyze(here)
    result["gap_changes_second_minus_first_eV"] = {
        name.removeprefix("train_"): result["metrics"][value["second"]]["dev_minus_train_gap_eV"] -
            result["metrics"][value["first"]]["dev_minus_train_gap_eV"]
        for name, value in result["contrasts"].items() if name.startswith("train_")}
    result["gap_change_uncertainty"] = "Point difference only; train/dev rows are disjoint, not paired across roles. Per-role paired row intervals are retained separately."
    result["analysis_source_sha256"] = {str(Path(__file__).resolve()): sha256_file(Path(__file__)),
        str(here.parents[1] / "src/molgap/k1_clean_fit_diagnostic.py"): sha256_file(here.parents[1] / "src/molgap/k1_clean_fit_diagnostic.py")}
    atomic_json(here / "analysis.json", result)
