import json
import unittest
from pathlib import Path

from molgap.pcqm_k1_variants import ARCHITECTURE_CONFIGS, make_encoder


ROOT = Path(__file__).resolve().parent
MODE = "neural_atom_k1_mose_hidden_bn"


class HiddenBatchNormContractTest(unittest.TestCase):
    def test_frozen_contract(self):
        contract = json.loads((ROOT / "training_contract.json").read_text())
        self.assertEqual(contract["physical_batch_per_device"], 128)
        self.assertEqual(contract["precision"], "fp32")
        self.assertFalse(contract["tf32"])
        self.assertEqual(contract["minimum_material_gain_eV"], 0.003)
        self.assertFalse(contract["structural_encoding"]["concatenate_rwse"])

    def test_single_architecture_change(self):
        config = ARCHITECTURE_CONFIGS[MODE]
        self.assertEqual(config["expected_parameters"], 3_662_081)
        self.assertEqual(
            config["structural_mlp"],
            "linear192-batchnorm192-silu-linear192",
        )
        self.assertFalse(config["raw_input_batchnorm"])
        self.assertFalse(config["geometry"])
        self.assertFalse(config["teacher"])

    def test_encoder_contains_one_hidden_batchnorm(self):
        import torch

        model = make_encoder(MODE)
        normalizers = [
            module
            for module in model.rwse_encoder.modules()
            if isinstance(module, torch.nn.BatchNorm1d)
        ]
        self.assertEqual(len(normalizers), 1)
        self.assertEqual(normalizers[0].num_features, 192)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3_662_081)

    def test_kaggle_binding(self):
        metadata = json.loads(
            (ROOT / "gpu_candidate" / "kernel-metadata.json").read_text()
        )
        self.assertEqual(
            metadata["id"], "nothingnessvoid/molgap-pcqm-k1-mose-hidden-bn-s42"
        )
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertIn(
            "nothingnessvoid/molgap-pcqm-k1-mose-cache-v1",
            metadata["dataset_sources"],
        )

    def test_runtime_contract(self):
        runtime = (ROOT / "gpu_candidate" / "run_candidate.py").read_text()
        self.assertIn('MODE = "neural_atom_k1_mose_hidden_bn"', runtime)
        self.assertIn("torch.backends.cuda.matmul.allow_tf32 = False", runtime)
        self.assertIn("torch.backends.cudnn.allow_tf32 = False", runtime)
        self.assertIn('"torch==2.4.1"', runtime)


if __name__ == "__main__":
    unittest.main()
