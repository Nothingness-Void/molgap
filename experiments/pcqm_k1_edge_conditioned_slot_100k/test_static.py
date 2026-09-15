"""Static contract checks only; never import or execute a model locally."""
from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from molgap.constants import REPO_ROOT


ROOT = REPO_ROOT / "experiments/pcqm_k1_edge_conditioned_slot_100k"


class ContractTests(unittest.TestCase):
    def test_source_and_launcher_syntax(self):
        paths = [
            REPO_ROOT / "src/molgap/k1_edge_conditioned_slot.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py",
            ROOT / "accept.py",
            ROOT / "package_source.py",
            ROOT / "p100_candidate/run_candidate.py",
        ]
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"))

    def test_candidate_contract_is_exactly_one_mechanism(self):
        contract = json.loads((ROOT / "training_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["arms"], ["neural_atom_k1_edge_conditioned_slot"])
        self.assertEqual(contract["physical_batch_per_device"], 128)
        self.assertEqual(contract["seed"], 42)
        self.assertEqual(contract["architecture"]["parameters"], 3_671_105)
        self.assertFalse(contract["architecture"]["geometry_model_input"])
        self.assertFalse(contract["roles"]["official_validation_role_read"])
        self.assertFalse(contract["roles"]["test_dev_role_read"])

    def test_mode_dispatch_and_zero_edge_path_are_declared(self):
        source = (REPO_ROOT / "src/molgap/k1_edge_conditioned_slot.py").read_text(encoding="utf-8")
        variants = (REPO_ROOT / "src/molgap/pcqm_k1_variants.py").read_text(encoding="utf-8")
        runner = (REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py").read_text(encoding="utf-8")
        for token in (
            "mean-incident-directed-real-bond-state",
            "nn.init.zeros_(projection.weight)",
            "edge_conditioned_keys",
            "edge_context_norms",
        ):
            self.assertIn(token, source)
        self.assertIn("k1_edge_conditioned_slot", variants)
        self.assertIn("EDGE_CONDITIONED_MODES", runner)
        self.assertIn("resume_two_step_bitwise_equal", runner)

    def test_remote_job_is_private_single_p100(self):
        metadata = json.loads((ROOT / "p100_candidate/kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaP100")
        self.assertEqual(metadata["kernel_sources"], [])
        self.assertEqual(metadata["model_sources"], [])
        launcher = (ROOT / "p100_candidate/run_candidate.py").read_text(encoding="utf-8")
        self.assertIn("source_payload.bin", launcher)
        self.assertIn("MODE = \"neural_atom_k1_edge_conditioned_slot\"", launcher)
        self.assertNotIn("test", launcher.lower().replace("test_candidate", ""))

    def test_no_local_model_execution_is_promised(self):
        self.assertIn("no model", (ROOT / "README.md").read_text(encoding="utf-8").lower())
        self.assertIn("without constructing or executing a model", (ROOT / "protocol.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
