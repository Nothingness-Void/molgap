"""Static contract tests; no encoder construction or local model execution."""
import ast
import json
import unittest
from pathlib import Path
from molgap.constants import REPO_ROOT


ROOT = Path(__file__).parent


class ContractTests(unittest.TestCase):
    def test_all_sources_parse(self):
        for path in (
            REPO_ROOT / "src/molgap/k1_edge_slot_interaction.py",
            REPO_ROOT / "src/molgap/k1_edge_kaggle_runtime.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py",
            ROOT / "accept.py", ROOT / "package_source.py",
            ROOT / "t4x2_candidates/run_candidates.py",
        ):
            ast.parse(path.read_text(), filename=str(path))

    def test_two_isolated_modes(self):
        source = (REPO_ROOT / "src/molgap/k1_edge_slot_interaction.py").read_text()
        self.assertIn("neural_atom_k1_edge_context_no_slot_attention", source)
        self.assertIn("neural_atom_k1_edge_context_uniform_return", source)
        self.assertIn("normalize_context=True", source)
        for forbidden in ("nn.Linear", "nn.LayerNorm", "nn.Parameter", "pos", "edge_distance"):
            self.assertNotIn(forbidden, source)

    def test_v4_contract_is_unchanged(self):
        tree = ast.parse((REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py").read_text())
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in {"SEED", "TRAIN_ROWS", "DEVELOPMENT_ROWS", "BATCH_SIZE", "EPOCHS", "LEARNING_RATE", "WEIGHT_DECAY"}}
        self.assertEqual(values, {"SEED":42,"TRAIN_ROWS":100000,"DEVELOPMENT_ROWS":50000,"BATCH_SIZE":128,"EPOCHS":40,"LEARNING_RATE":4e-4,"WEIGHT_DECAY":1e-5})

    def test_kaggle_is_private_t4x2(self):
        metadata = json.loads((ROOT / "t4x2_candidates/kernel-metadata.json").read_text())
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["kernel_sources"], [])
        launcher = (ROOT / "t4x2_candidates/run_candidates.py").read_text()
        self.assertIn("MOLGAP_SCREEN_MODES", launcher)
        self.assertNotIn("src.zip", launcher)

    def test_acceptance_has_no_model_construction(self):
        source = (ROOT / "accept.py").read_text()
        for forbidden in ("make_encoder", ".to(\"cuda\")", "load_state_dict"):
            self.assertNotIn(forbidden, source)
        self.assertIn("initialization_policy=\"identical-tensors-altered-edge-dataflow\"", source)

    def test_round_three_is_conditional(self):
        protocol = (ROOT / "protocol.md").read_text(encoding="utf-8")
        self.assertIn("within `0.0005 eV`", protocol)
        self.assertIn("Official validation, shadow and all test roles stay", protocol)


if __name__ == "__main__":
    unittest.main()
