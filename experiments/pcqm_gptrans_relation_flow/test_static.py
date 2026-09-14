"""Executable contract/orchestration tests: no torch or model execution."""
import ast
import importlib.util
import json
import subprocess
import unittest
from unittest.mock import patch

from molgap.constants import REPO_ROOT

EXP = REPO_ROOT / "experiments/pcqm_gptrans_relation_flow"


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def function_ast(source, name):
    return ast.dump(next(node for node in ast.parse(source).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name), include_attributes=False)


class ContractTests(unittest.TestCase):
    def test_all_sources_parse(self):
        paths = list(EXP.rglob("*.py")) + [REPO_ROOT / "src/molgap" / name for name in ("gptrans_variants.py", "gptrans_variant_checks.py", "pcqm_gptrans_v4.py")]
        for path in paths:
            tree(path)

    def test_frozen_core_unchanged(self):
        old = subprocess.check_output(["git", "show", "0a88b34:src/molgap/gptrans.py"], cwd=REPO_ROOT, text=True)
        self.assertEqual(ast.dump(ast.parse(old)), ast.dump(tree(REPO_ROOT / "src/molgap/gptrans.py")))

    def test_training_contract_unchanged(self):
        old = subprocess.check_output(["git", "show", "0a88b34:src/molgap/pcqm_gptrans_v4.py"], cwd=REPO_ROOT, text=True)
        new = (REPO_ROOT / "src/molgap/pcqm_gptrans_v4.py").read_text()
        for name in ("_scientific_fields", "_optimizer_step", "_evaluate", "FrozenEpochScheduler", "ExponentialMovingAverage", "DeterministicEpochBatchSampler", "validate_fixed_assets", "_verify_model_identity"):
            self.assertEqual(function_ast(old, name), function_ast(new, name), name)
        def constants(source):
            return {node.targets[0].id: ast.dump(node.value) for node in ast.parse(source).body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        self.assertEqual(constants(old), constants(new))

    def test_variant_no_new_parameter_layers(self):
        source = tree(REPO_ROOT / "src/molgap/gptrans_variants.py")
        calls = [node for node in ast.walk(source) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr in ("Parameter", "Linear", "LayerNorm", "Embedding") for node in calls))
        text = (REPO_ROOT / "src/molgap/gptrans_variants.py").read_text()
        self.assertIn("torch.random.fork_rng(devices=[])", text)
        self.assertIn("load_state_dict(original.state_dict(), strict=True)", text)

    def test_metadata_fixed_assets_only(self):
        meta = json.loads((EXP / "kaggle_t4x2/kernel-metadata.json").read_text())
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(meta["dataset_sources"], ["kaseichou/molgap-gptrans-relation-flow-source", "kaseichou/pcqm4mv2-ogb-fixed-100k-v1"])
        for key in ("competition_sources", "kernel_sources", "model_sources"):
            self.assertEqual(meta[key], [])
        self.assertEqual(meta["is_private"], "true")

    def test_parent_launches_only_two_variants(self):
        spec = importlib.util.spec_from_file_location("relation_entry", EXP / "kaggle_t4x2/run_candidates.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        self.assertEqual(entry.MODES, ("pair_prenorm", "centered_logits"))
        self.assertEqual(entry.MAX_WALL_SECONDS, 36000)
        with patch.object(entry.os, "environ", {"MOLGAP_STAGE": "training", "MOLGAP_VARIANT": "reference", "MOLGAP_CONTEXT": "{}"}):
            with self.assertRaises(ValueError):
                entry.main()

    def test_no_model_packaging(self):
        source = (EXP / "package_source.py").read_text()
        self.assertNotIn("import torch", source)
        self.assertNotIn("_make_model", source)
        self.assertIn("shutil.copyfile(args.initial_state", source)

    def test_resume_and_output_variant_bound(self):
        source = (REPO_ROOT / "src/molgap/pcqm_gptrans_v4.py").read_text()
        self.assertIn('checkpoint.get("variant", "reference") != variant', source)
        self.assertIn('preflight.get("variant", "reference") != variant', source)
        self.assertGreaterEqual(source.count('"variant": variant'), 4)
        self.assertIn('checkpoint_epoch_{epoch:02d}.pt', source)


if __name__ == "__main__":
    unittest.main()
