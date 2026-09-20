import ast
import json
import unittest
from pathlib import Path

from molgap.constants import REPO_ROOT

EXP = REPO_ROOT / "experiments/pcqm_gptrans_conditional_flow_100k"


class ConditionalFlowTests(unittest.TestCase):
    def test_sources_parse(self):
        paths = list(EXP.rglob("*.py")) + [
            REPO_ROOT / "src/molgap/gptrans_conditional_flow.py",
            REPO_ROOT / "src/molgap/gptrans_variant_checks.py",
            REPO_ROOT / "src/molgap/gptrans_kaggle_runtime.py",
            REPO_ROOT / "src/molgap/pcqm_gptrans_v4.py",
        ]
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8-sig"))

    def test_contract(self):
        contract = json.loads((EXP / "training_contract.json").read_text())
        self.assertEqual(contract["candidates"], [
            "conditional_pair_readback", "conditional_pair_recurrence"
        ])
        self.assertEqual(contract["physical_batch_per_device"], 128)
        self.assertEqual(contract["epochs"], 60)
        self.assertEqual(contract["sample_presentations"], 5998080)

    def test_kernel_binding(self):
        metadata = json.loads((EXP / "kaggle_t4x2/kernel-metadata.json").read_text())
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["is_private"], "true")
        self.assertEqual(len(metadata["dataset_sources"]), 2)

    def test_two_arms(self):
        source = (EXP / "kaggle_t4x2/run_candidates.py").read_text()
        self.assertIn("conditional_pair_readback", source)
        self.assertIn("conditional_pair_recurrence", source)
        self.assertIn("kaggle1-t4x2", source)


if __name__ == "__main__":
    unittest.main()
