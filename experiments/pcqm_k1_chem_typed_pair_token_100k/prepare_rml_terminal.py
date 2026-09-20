"""Configure the shared no-inference terminal adapter for this experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from molgap.constants import REPO_ROOT


def main() -> None:
    shared_path = (
        REPO_ROOT
        / "experiments/pcqm_k1_functional_group_token_100k/prepare_rml_terminal.py"
    )
    spec = importlib.util.spec_from_file_location("shared_rml_terminal_adapter", shared_path)
    shared = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(shared)

    root_rel = "experiments/pcqm_k1_chem_typed_pair_token_100k"
    record_rel = "platforms/_records/kaggle/training/pcqm_k1_chem_typed_pair_token_s42_v1"
    shared.ROOT = REPO_ROOT / root_rel
    shared.ROOT_REL = root_rel
    shared.RESULTS = shared.ROOT / "results"
    shared.RECORD_ROOT = REPO_ROOT / record_rel
    shared.CANDIDATE_ROOT = (
        shared.RECORD_ROOT
        / "pcqm_k1_chem_typed_pair_token/neural_atom_k1_chem_typed_pair_token"
    )
    shared.RAW_ACCEPTANCE = shared.RESULTS / "raw_acceptance.json"
    shared.TRAJECTORY_ID = "TC-k1-chem-typed-pair-token-100k-s42"
    shared.RUN_ID = "kaseichou/molgap-k1-chemistry-typed-pairtoken-s42:v1"
    shared.CONTRACT_RUN_ID = (
        "pcqm-k1-variants-100k-s42-v1-neural_atom_k1_chem_typed_pair_token"
    )
    shared.MODE = "neural_atom_k1_chem_typed_pair_token"
    shared.ACTION_ID = "A001"
    shared.EVIDENCE_ID = "pcqm-k1-chem-typed-pair-token-100k-s42"
    shared.LOCAL_RECORD_URI = f"external://local-platform-record/{record_rel}/pcqm_k1_chem_typed_pair_token/neural_atom_k1_chem_typed_pair_token"
    shared.DECISION_OUTCOME = "POSITIVE_BELOW_GATE"
    shared.NEXT_ALLOWED_ACTIONS = []
    shared.REOPEN_CONDITIONS = [
        "a future route must change relation values or pretraining rather than retry chemistry-role selection"
    ]
    shared.SCIENTIFIC_STATUS = "positive_below_gate"
    shared.FINALIZED_AT = "2026-09-20T20:44:22.1596800+09:00"
    shared.main()


if __name__ == "__main__":
    main()
