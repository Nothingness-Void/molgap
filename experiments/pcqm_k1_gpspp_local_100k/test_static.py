"""Static contract checks only; no encoder construction or local training."""
import ast
import json
import unittest
from pathlib import Path

from molgap.constants import REPO_ROOT


ROOT = Path(__file__).parent


class ContractTests(unittest.TestCase):
    def test_sources_parse(self):
        for path in (
            REPO_ROOT / "src/molgap/k1_gpspp_local.py",
            REPO_ROOT / "src/molgap/k1_edge_kaggle_runtime.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants.py",
            REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py",
            ROOT / "accept.py",
            ROOT / "package_source.py",
            ROOT / "t4x2_candidates/run_candidates.py",
        ):
            ast.parse(path.read_text(), filename=str(path))

    def test_directional_design_and_parameter_identity(self):
        source = (REPO_ROOT / "src/molgap/k1_gpspp_local.py").read_text()
        tree = ast.parse(source)
        module_imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "qm9_neural_atom"
            for alias in node.names
        ]
        self.assertIn("MIXER_LAYERS", module_imports)
        for token in (
            "neural_atom_k1_gpspp_sender",
            "neural_atom_k1_gpspp_bidirectional",
            "source(normalized[source])",
            "target(normalized[target])",
            "receiver, sender",
            "nn.init.zeros_(self.return_projection.weight)",
            "ADAPTER_PARAMETERS_PER_LAYER = 95_232",
            "4_515_905",
        ):
            self.assertIn(token, source)
        for forbidden in ("pos", "edge_distance", "wedge_angle", "teacher"):
            self.assertNotIn(forbidden, source)

    def test_v4_contract_is_unchanged(self):
        tree = ast.parse(
            (REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py").read_text()
        )
        names = {
            "SEED",
            "TRAIN_ROWS",
            "DEVELOPMENT_ROWS",
            "BATCH_SIZE",
            "EPOCHS",
            "LEARNING_RATE",
            "WEIGHT_DECAY",
        }
        values = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names
        }
        self.assertEqual(
            values,
            {
                "SEED": 42,
                "TRAIN_ROWS": 100000,
                "DEVELOPMENT_ROWS": 50000,
                "BATCH_SIZE": 128,
                "EPOCHS": 40,
                "LEARNING_RATE": 4e-4,
                "WEIGHT_DECAY": 1e-5,
            },
        )

    def test_private_t4x2_submission(self):
        metadata = json.loads(
            (ROOT / "t4x2_candidates/kernel-metadata.json").read_text()
        )
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["kernel_sources"], [])
        launcher = (ROOT / "t4x2_candidates/run_candidates.py").read_text()
        self.assertIn("MOLGAP_SCREEN_MODES", launcher)
        self.assertIn("MOLGAP_K1_OUTPUT_ROOT", launcher)
        self.assertNotIn("ambiguous source roots", launcher)
        self.assertIn("hash-verified neutral payload", launcher)
        self.assertLess(
            launcher.index("Source identity mismatch"),
            launcher.index("from molgap.k1_edge_kaggle_runtime"),
        )

    def test_acceptance_has_no_model_execution(self):
        source = (ROOT / "accept.py").read_text()
        for forbidden in ("make_encoder", '.to("cuda")', "load_state_dict"):
            self.assertNotIn(forbidden, source)
        for required in (
            'initialization_policy="nested-function"',
            "shared_k1_initial_state_sha256",
            "resume_two_step_bitwise_equal",
            "recovery_epoch_",
            "model_inference_executed",
        ):
            self.assertIn(required, source)

    def test_protocol_keeps_sealed_roles_closed(self):
        protocol = (ROOT / "protocol.md").read_text(encoding="utf-8")
        self.assertIn("physical\nBS128", protocol)
        self.assertIn("strict deterministic FP32", protocol)
        self.assertIn("Official validation, shadow and every test role remain", protocol)
        self.assertIn("third and final round", protocol)

    def test_registry_contains_only_declared_new_modes(self):
        source = (REPO_ROOT / "src/molgap/pcqm_k1_variants.py").read_text()
        for mode in (
            "neural_atom_k1_gpspp_sender",
            "neural_atom_k1_gpspp_bidirectional",
        ):
            self.assertEqual(source.count(f'"{mode}"'), 1)


if __name__ == "__main__":
    unittest.main()
