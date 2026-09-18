import json
import unittest
from pathlib import Path

from molgap.pcqm_mose import MOSE_DIM, MOSE_PATTERNS, rooted_homomorphism_counts


ROOT = Path(__file__).resolve().parent


class MoSEContractTest(unittest.TestCase):
    def test_pattern_inventory(self):
        self.assertEqual(MOSE_DIM, 31)
        self.assertEqual(len(MOSE_PATTERNS), 31)
        self.assertEqual(max(max(edge) for edge in MOSE_PATTERNS[-1]), 5)

    def test_rooted_counts_on_path(self):
        counts = rooted_homomorphism_counts(
            3,
            [[0, 1, 1, 2], [1, 0, 2, 1]],
        )
        self.assertEqual([row[0] for row in counts], [1, 2, 1])
        self.assertTrue(all(len(row) == 31 for row in counts))
        self.assertTrue(all(value >= 0 for row in counts for value in row))

    def test_frozen_contract(self):
        contract = json.loads((ROOT / "training_contract.json").read_text())
        self.assertEqual(contract["physical_batch_per_device"], 128)
        self.assertEqual(contract["precision"], "fp32")
        self.assertFalse(contract["tf32"])
        self.assertFalse(contract["structural_encoding"]["concatenate_rwse"])
        self.assertEqual(contract["minimum_material_gain_eV"], 0.003)

    def test_kaggle_identity(self):
        cpu = json.loads((ROOT / "cpu_cache" / "kernel-metadata.json").read_text())
        gpu = json.loads((ROOT / "gpu_candidate" / "kernel-metadata.json").read_text())
        self.assertFalse(cpu["enable_gpu"])
        self.assertTrue(gpu["enable_gpu"])
        self.assertEqual(gpu["machine_shape"], "NvidiaTeslaP100")
        self.assertTrue(cpu["id"].startswith("nothingnessvoid/"))
        self.assertTrue(gpu["id"].startswith("nothingnessvoid/"))


if __name__ == "__main__":
    unittest.main()

