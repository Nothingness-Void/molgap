import ast
import importlib.util
import json
import unittest
from pathlib import Path

from molgap.constants import REPO_ROOT

EXP = REPO_ROOT / "experiments/pcqm_gptrans_flow_ablation_100k"


class FlowAblationTests(unittest.TestCase):
    def test_sources_parse(self):
        paths = list(EXP.rglob("*.py")) + [
            REPO_ROOT / "src/molgap/gptrans_flow_ablation.py",
            REPO_ROOT / "src/molgap/gptrans_variant_checks.py",
            REPO_ROOT / "src/molgap/gptrans_kaggle_runtime.py",
            REPO_ROOT / "src/molgap/pcqm_gptrans_v4.py",
        ]
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8-sig"))

    def test_contract_is_reference_matched(self):
        contract = json.loads((EXP / "training_contract.json").read_text())
        self.assertEqual(contract["candidates"], ["no_pair_to_node", "no_pair_recurrence"])
        self.assertEqual(contract["parameter_count"], 5246817)
        self.assertEqual(contract["physical_batch_per_device"], 128)
        self.assertEqual(contract["epochs"], 60)
        self.assertEqual(contract["sample_exposure"], 5998080)

    def test_kernel_binding(self):
        metadata = json.loads((EXP / "kaggle_t4x2/kernel-metadata.json").read_text())
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(
            metadata["dataset_sources"],
            [
                "nothingnessvoid/molgap-gptrans-flow-ablation-source",
                "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1",
            ],
        )
        self.assertEqual(metadata["is_private"], "true")

    def test_entrypoint_binds_two_arms(self):
        spec = importlib.util.spec_from_file_location(
            "flow_entry", EXP / "kaggle_t4x2/run_candidates.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(
            json.loads(module.os.environ["MOLGAP_SCREEN_MODES"]),
            ["no_pair_to_node", "no_pair_recurrence"],
        )
        self.assertEqual(module.os.environ["MOLGAP_PLATFORM_ID"], "kaggle1-t4")

    def test_flow_module_adds_no_parameter_definitions(self):
        source = ast.parse(
            (REPO_ROOT / "src/molgap/gptrans_flow_ablation.py").read_text()
        )
        calls = [node for node in ast.walk(source) if isinstance(node, ast.Call)]
        forbidden = {"Parameter", "Embedding"}
        self.assertFalse(
            any(
                isinstance(node.func, ast.Attribute) and node.func.attr in forbidden
                for node in calls
            )
        )


if __name__ == "__main__":
    unittest.main()
