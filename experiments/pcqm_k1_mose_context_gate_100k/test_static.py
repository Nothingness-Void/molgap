import json
import unittest
from pathlib import Path

import torch

from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS, make_encoder


ROOT = Path(__file__).resolve().parent
MODE = "neural_atom_k1_rwse_mose_context_gate"


class MoSEContextGateContractTest(unittest.TestCase):
    def test_contract(self):
        contract = json.loads((ROOT / "training_contract.json").read_text())
        self.assertEqual(contract["physical_batch_per_device"], 128)
        self.assertEqual(contract["precision"], "fp32")
        self.assertFalse(contract["tf32"])
        self.assertEqual(contract["minimum_material_gain_eV"], 0.003)

    def test_architecture_identity(self):
        config = ARCHITECTURE_CONFIGS[MODE]
        self.assertEqual(config["expected_parameters"], 3_687_682)
        self.assertEqual(config["initialization_policy"], "nested-function")
        self.assertIn("graph-mean-and-second-moment", config["gate"])
        self.assertFalse(config["geometry"])
        self.assertFalse(config["teacher"])

    def test_parameter_count_and_zero_residual(self):
        torch.manual_seed(42)
        model = make_encoder(MODE)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3_687_682)
        self.assertEqual(model.gate_scope, "molecule-context")
        self.assertEqual(torch.count_nonzero(model.mose_residual[-1].weight), 0)
        self.assertEqual(torch.count_nonzero(model.mose_residual[-1].bias), 0)

    def test_exact_k1_initialization(self):
        torch.manual_seed(42)
        reference = make_encoder("neural_atom_k1_v4").eval()
        torch.manual_seed(42)
        candidate = make_encoder(MODE).eval()
        x = torch.zeros((4, 9), dtype=torch.long)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        edge_attr = torch.zeros((4, 3), dtype=torch.long)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        rwse = torch.rand((4, 16))
        mose = torch.rand((4, 31))
        with torch.no_grad():
            expected = reference(x, edge_index, edge_attr, batch, rwse)
            actual = candidate(x, edge_index, edge_attr, batch, torch.cat((rwse, mose), dim=-1))
        self.assertTrue(torch.equal(expected, actual))

    def test_kaggle_binding(self):
        metadata = json.loads((ROOT / "gpu_candidate" / "kernel-metadata.json").read_text())
        self.assertEqual(metadata["id"], "nothingnessvoid/molgap-pcqm-k1-mose-context-gate-s42")
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")


if __name__ == "__main__":
    unittest.main()

