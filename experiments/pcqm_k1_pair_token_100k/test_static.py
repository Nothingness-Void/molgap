"""Static contract checks; no local model construction or execution."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from molgap.constants import REPO_ROOT


ROOT = REPO_ROOT / "experiments/pcqm_k1_pair_token_100k"


class ContractTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in (
            REPO_ROOT / "src/molgap/k1_pair_token.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py",
            ROOT / "run_candidate.py",
            ROOT / "accept.py",
        ):
            ast.parse(path.read_text(encoding="utf-8"))

    def test_contract_preserves_v4(self):
        contract = json.loads(
            (ROOT / "training_contract.json").read_text(encoding="utf-8")
        )
        self.assertEqual(contract["arms"], ["neural_atom_k1_pair_token"])
        self.assertEqual(contract["physical_batch_per_device"], 128)
        self.assertEqual(contract["seed"], 42)
        self.assertEqual(contract["architecture"]["parameters"], 3_681_601)
        self.assertEqual(contract["architecture"]["target_layer"], 6)
        self.assertFalse(contract["architecture"]["dense_atom_attention"])
        self.assertFalse(contract["architecture"]["geometry_model_input"])
        self.assertFalse(contract["roles"]["official_validation_role_read"])
        self.assertFalse(contract["roles"]["test_dev_role_read"])

    def test_one_relation_path_is_declared(self):
        source = (REPO_ROOT / "src/molgap/k1_pair_token.py").read_text(
            encoding="utf-8"
        )
        for token in (
            "PAIR_CHANNELS = 32",
            "TARGET_LAYER = 6",
            "self.pair_norm",
            "nn.init.zeros_(self.return_projection.weight)",
            "all-ordered-pairs-of-current-layer6-node-states",
        ):
            self.assertIn(token, source)
        self.assertNotIn("shortest_path", source)
        self.assertNotIn("edge_distance", source)

    def test_remote_job_is_bounded(self):
        slurm = (ROOT / "run_kunshan.slurm").read_text(encoding="utf-8")
        self.assertIn("--time=05:00:00", slurm)
        self.assertIn("--gres=dcu:Hygon:1", slurm)
        self.assertIn("MOLGAP_FIXED_CACHE_ROOT", (
            REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py"
        ).read_text(encoding="utf-8"))

    def test_no_local_model_execution_is_promised(self):
        self.assertIn(
            "never constructs or executes a model",
            (ROOT / "README.md").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "Saved-artifact acceptance performs no model inference",
            (ROOT / "protocol.md").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()

