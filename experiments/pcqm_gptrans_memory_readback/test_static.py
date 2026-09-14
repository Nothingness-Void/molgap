"""Static scientific-contract and bootstrap tests; no local models."""
import ast
import importlib.util
import json
import subprocess
import unittest
from unittest.mock import patch
from molgap.constants import REPO_ROOT

EXP = REPO_ROOT / "experiments/pcqm_gptrans_memory_readback"


class MemoryTests(unittest.TestCase):
    def test_sources_parse(self):
        for path in [*EXP.rglob("*.py"), *(REPO_ROOT / "src/molgap" / n for n in ("gptrans_memory.py", "gptrans_kaggle_runtime.py", "gptrans_variant_checks.py", "pcqm_gptrans_v4.py"))]:
            ast.parse(path.read_text(encoding="utf-8-sig"))

    def test_frozen_scientific_functions(self):
        before = ast.parse(subprocess.check_output(["git", "show", "978af90:src/molgap/pcqm_gptrans_v4.py"], cwd=REPO_ROOT, text=True))
        after = ast.parse((REPO_ROOT / "src/molgap/pcqm_gptrans_v4.py").read_text())
        names = ("_scientific_fields", "_optimizer_step", "_evaluate", "FrozenEpochScheduler", "ExponentialMovingAverage", "DeterministicEpochBatchSampler", "validate_fixed_assets", "_verify_model_identity")
        for name in names:
            get = lambda t: ast.dump(next(n for n in t.body if getattr(n, "name", None) == name))
            self.assertEqual(get(before), get(after), name)
        constants = lambda t: [ast.dump(n) for n in t.body if isinstance(n, ast.Assign)]
        self.assertEqual(constants(before), constants(after))

    def test_no_stack_or_parameters(self):
        source = (REPO_ROOT / "src/molgap/gptrans_memory.py").read_text()
        self.assertNotIn("normalize_pair", source)
        self.assertNotIn("center_valid_logits", source)
        self.assertIn("memory = pair + delta", source)
        self.assertIn('routing = delta if variant == "memory_value" else memory', source)
        self.assertIn("torch.random.fork_rng(devices=[])", source)
        self.assertNotIn("nn.Parameter", source)
        self.assertNotIn("nn.Linear", source)

    def test_metadata(self):
        meta = json.loads((EXP / "kaggle_t4x2/kernel-metadata.json").read_text())
        self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(meta["dataset_sources"], ["kaseichou/molgap-gptrans-memory-readback-source", "kaseichou/pcqm4mv2-ogb-fixed-100k-v1"])
        self.assertEqual(meta["kernel_sources"], [])
        self.assertEqual(meta["competition_sources"], [])

    def test_invalid_worker_rejected_before_import(self):
        spec = importlib.util.spec_from_file_location("runtime", REPO_ROOT / "src/molgap/gptrans_kaggle_runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.MODES, ("memory_value", "memory_message"))
        with patch.object(module.os, "environ", {"MOLGAP_STAGE": "training", "MOLGAP_VARIANT": "reference"}):
            with self.assertRaises(ValueError):
                module.main()

    def test_bootstrap_no_torch_before_pin(self):
        wrapper = (EXP / "kaggle_t4x2/run_candidates.py").read_text()
        self.assertNotIn("import torch", wrapper)
        runtime = (REPO_ROOT / "src/molgap/gptrans_kaggle_runtime.py").read_text()
        self.assertIn("torch==2.4.1", runtime)
        self.assertIn('env.update(CUDA_VISIBLE_DEVICES=str(device)', runtime)
        self.assertIn('for stage in ("checks", "preflight", "training")', runtime)


if __name__ == "__main__":
    unittest.main()
