"""AST/contract checks only: no model imports or local GPU use."""
import ast
import json
import unittest
from pathlib import Path
from molgap.constants import REPO_ROOT


class ContractTests(unittest.TestCase):
    def test_source_syntax(self):
        for name in ("k1_edge_memory.py", "pcqm_k1_variants.py", "pcqm_k1_variants_runner.py"):
            ast.parse((REPO_ROOT / "src/molgap" / name).read_text(encoding="utf-8"))

    def test_frozen_training_literals(self):
        tree = ast.parse((REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py").read_text())
        assignments = {node.targets[0].id: node.value for node in tree.body
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        for key, value in {"SEED":42,"TRAIN_ROWS":100000,"DEVELOPMENT_ROWS":50000,"BATCH_SIZE":128,"EPOCHS":40,"LEARNING_RATE":4e-4,"WEIGHT_DECAY":1e-5}.items():
            self.assertEqual(ast.literal_eval(assignments[key]), value)

    def test_no_new_trainable_modules(self):
        source = (REPO_ROOT / "src/molgap/k1_edge_memory.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, ("Linear", "LayerNorm", "Parameter", "Embedding"))
        self.assertIn("memory = memory + update.update(context)", source)
        self.assertIn("return memory, update.output_norm(memory)", source)

    def test_initialization_exception_scoped(self):
        text = (REPO_ROOT / "experiments/pcqm_k1_variants_100k/accept.py").read_text()
        self.assertIn('initialization_policy: str = "nested-function"', text)
        self.assertIn("mode not in modes", text)
        self.assertIn('reference["preflight"]["shared_k1_initial_state_sha256"]', text)

    def test_recovery_rng_and_chunks(self):
        text = (REPO_ROOT / "src/molgap/pcqm_k1_variants_runner.py").read_text()
        for token in ("capture_rng_state()", "restore_rng_state(checkpoint", "resume_from", "best_artifact_sha256", "recovery_epoch_", "os.replace(temporary, chunk)"):
            self.assertIn(token, text)

    def test_three_round_scope(self):
        facts = json.loads((Path(__file__).parent / "results/authorization.json").read_text())
        self.assertEqual(facts["authorized_new_rounds"], 3)
        self.assertFalse(facts["extra_seeds_authorized"])
        self.assertFalse(facts["scale_up_authorized"])

    def test_infrastructure_syntax_and_isolation(self):
        root = Path(__file__).parent
        for path in (root / "accept.py", root / "package_source.py", root / "t4x2_candidates/run_candidates.py", REPO_ROOT / "src/molgap/k1_edge_kaggle_runtime.py"):
            ast.parse(path.read_text())
        metadata = json.loads((root / "t4x2_candidates/kernel-metadata.json").read_text())
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["kernel_sources"], [])
        self.assertEqual(metadata["model_sources"], [])
        runtime = (REPO_ROOT / "src/molgap/k1_edge_kaggle_runtime.py").read_text()
        self.assertIn("CUDA_VISIBLE_DEVICES=str(device)", runtime)
        self.assertIn("torch==2.4.1", runtime)
        self.assertIn("MOLGAP_K1_OUTPUT_ROOT", runtime)
        self.assertIn("ALLOWED_MODE_PAIRS", runtime)
        launcher = (root / "t4x2_candidates/run_candidates.py").read_text()
        self.assertNotIn("src.zip", launcher)
        self.assertLess(launcher.index("Source identity mismatch"), launcher.index("from molgap"))

if __name__ == "__main__":
    unittest.main()
