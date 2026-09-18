"""Static and contract-only tests; no local model execution."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from molgap.constants import REPO_ROOT
from molgap.pcqm_k1_sparse_pair_100k import (
    ROW_ORDER_FINGERPRINT,
    row_order_fingerprint,
)


ROOT = REPO_ROOT / "experiments/pcqm_k1_sparse_pair_100k"


class SparsePairContractTests(unittest.TestCase):
    def test_python_syntax(self):
        paths = (
            REPO_ROOT / "src/molgap/k1_sparse_pair.py",
            REPO_ROOT / "src/molgap/pcqm_k1_sparse_pair_100k.py",
            ROOT / "run.py",
            ROOT / "accept.py",
            ROOT / "accept_profiles.py",
            ROOT / "kaggle_profile.py",
            ROOT / "kaggle_train.py",
        )
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"))

    def test_contract_identity(self):
        contract = json.loads(
            (ROOT / "training_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["model"]["parameters"], 3_681_233)
        self.assertEqual(contract["model"]["target_layer"], 6)
        self.assertEqual(contract["training"]["physical_batch_per_device"], 128)
        self.assertEqual(contract["training"]["epochs"], 40)
        self.assertFalse(contract["model"]["persistent_dense_pair_state"])
        self.assertFalse(contract["roles"]["official_validation_role_read"])

    def test_single_mechanism_source(self):
        source = (REPO_ROOT / "src/molgap/k1_sparse_pair.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "TARGET_LAYER = 6",
            "MAX_PAIR_DISTANCE = 3",
            "PATH_ORDERS = 5",
            "self.pair_norm",
            "nn.init.zeros_(self.return_projection.weight)",
            "receiver_has_pair",
        ):
            self.assertIn(token, source)
        self.assertNotIn("edge_distance", source)
        self.assertNotIn("wedge_angle", source)

    def test_row_order_matches_frozen_k1(self):
        self.assertEqual(row_order_fingerprint(), ROW_ORDER_FINGERPRINT)


if __name__ == "__main__":
    unittest.main()
