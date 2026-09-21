"""Configure the shared no-inference terminal adapter for one-shot triplet PairToken."""
from __future__ import annotations

import importlib.util

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

    root_rel = "experiments/pcqm_k1_oneshot_triplet_pair_token_100k"
    record_rel = (
        "platforms/_records/kaggle/training/"
        "pcqm_k1_oneshot_triplet_pair_token_s42_v1"
    )
    shared.ROOT = REPO_ROOT / root_rel
    shared.ROOT_REL = root_rel
    shared.RESULTS = shared.ROOT / "results"
    shared.RECORD_ROOT = REPO_ROOT / record_rel
    shared.CANDIDATE_ROOT = (
        shared.RECORD_ROOT
        / "pcqm_k1_oneshot_triplet_pair_token/"
        "neural_atom_k1_pair_token_oneshot_triplet"
    )
    shared.RAW_ACCEPTANCE = shared.RESULTS / "acceptance_v1.json"
    shared.SOURCE_ACCEPTANCE_REL = "results/acceptance_v1.json"
    shared.TRAJECTORY_ID = "TC-k1-oneshot-triplet-pair-token-100k-s42"
    shared.RUN_ID = "nvoid912/molgap-k1-one-shot-triplet-pairtoken-s42:v1"
    shared.CONTRACT_RUN_ID = (
        "pcqm-k1-variants-100k-s42-v1-"
        "neural_atom_k1_pair_token_oneshot_triplet"
    )
    shared.MODE = "neural_atom_k1_pair_token_oneshot_triplet"
    shared.ACTION_ID = "A001"
    shared.EVIDENCE_ID = "pcqm-k1-oneshot-triplet-pair-token-100k-s42"
    shared.LOCAL_RECORD_URI = (
        f"external://local-platform-record/{record_rel}/"
        "pcqm_k1_oneshot_triplet_pair_token/"
        "neural_atom_k1_pair_token_oneshot_triplet"
    )
    shared.DECISION_OUTCOME = "POSITIVE_BELOW_GATE"
    shared.NEXT_ALLOWED_ACTIONS = [
        "close the authorized three-round sequence and retain original PairToken as the relation-mechanism incumbent"
    ]
    shared.REOPEN_CONDITIONS = [
        "a future experiment requires new user authority and a distinct non-triplet, non-SPD information-flow hypothesis"
    ]
    shared.SCIENTIFIC_STATUS = "positive_below_gate"
    shared.FINALIZED_AT = "2026-09-22T02:47:02+09:00"
    shared.PLATFORM = "kaggle3"
    shared.HARDWARE = "Tesla_T4_16GB"
    shared.main()


if __name__ == "__main__":
    main()
