"""Static contract checks; no local model construction or execution."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
MODE = "neural_atom_k1_pair_token_value_decoupled"


class ContractTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in (
            REPO_ROOT / "src/molgap/k1_pair_token.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py",
            ROOT / "run_candidate.py",
            ROOT / "package_source.py",
            ROOT / "kaggle_candidate/run_candidate.py",
        ):
            ast.parse(path.read_text(encoding="utf-8"))

    def test_frozen_contract(self):
        contract = json.loads((ROOT / "training_contract.json").read_text())
        self.assertEqual(contract["arm"], MODE)
        self.assertEqual(contract["physical_batch_per_device"], 128)
        self.assertEqual(contract["total_optimizer_steps"], 31240)
        self.assertEqual(contract["architecture"]["parameters"], 3_682_689)
        self.assertFalse(contract["roles"]["official_validation_role_read"])
        self.assertFalse(contract["roles"]["test_dev_role_read"])

    def test_isolated_value_intervention(self):
        source = (REPO_ROOT / "src/molgap/k1_pair_token.py").read_text()
        for token in (
            "VALUE_DECOUPLED_MODE",
            "self.value_projection",
            "nn.init.eye_(self.value_projection.weight)",
            "selection_value_decoupled",
        ):
            self.assertIn(token, source)
        self.assertNotIn("edge_distance", source)
        self.assertNotIn("shortest_path", source)

    def test_prospective_rml(self):
        trajectory = json.loads((ROOT / "trajectory.json").read_text())
        self.assertEqual(trajectory["record_mode"], "prospective")
        self.assertEqual(trajectory["owner"], "desktop")
        self.assertEqual(trajectory["decision"]["outcome"], "ACTIVE")
        self.assertEqual(trajectory["result"]["evidence_ids"], [])


if __name__ == "__main__":
    unittest.main()
