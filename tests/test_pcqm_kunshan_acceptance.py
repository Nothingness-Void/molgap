"""Mechanical acceptance regression tests; synthetic CSV/bytes, no torch."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("kunshan_accept", ROOT / "experiments/pcqm_gap_architecture/accept_kunshan_vector_screen.py")
ACCEPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPT)
SOURCE = "a" * 40


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.counts = {ACCEPT.BASELINE: 3665809, ACCEPT.CANDIDATE: 3700000}
        self.write(self.root / "preflight.json", {"accepted": True, "parameter_counts": self.counts})
        self.completion = {
            "format": "molgap-kunshan-vector-screen-v1", "complete": True,
            "source_commit": SOURCE, "candidates": list(self.counts),
            "geometry_cache_aggregate_sha256": ACCEPT.CACHE,
            "contract": ACCEPT.EXPECTED_CONTRACT, "platform": "SCNet Kunshan",
            "device_count": 1, "official_validation_role_read": False,
            "train_graphs": 100000, "validation_graphs": 10000,
            "test_dev_role_read": False,
            "preflight_sha256": ACCEPT.sha256(self.root / "preflight.json"), "runs": [],
        }
        for candidate, count in self.counts.items():
            folder = self.root / "results" / candidate
            folder.mkdir(parents=True)
            artifacts = {}
            for key in ("best_model", "checkpoint", "validation_payload"):
                path = folder / (key + ".pt")
                path.write_bytes(b"synthetic artifact, not a model")
                artifacts[key] = str(path.relative_to(self.root))
                artifacts[key + "_sha256"] = ACCEPT.sha256(path)
            path = folder / "validation.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row_index", "target_eV", "prediction_eV"])
                writer.writerows((i, 1.0, 1.125) for i in range(10000))
            artifacts["validation_csv"] = str(path.relative_to(self.root))
            artifacts["validation_csv_sha256"] = ACCEPT.sha256(path)
            trace = [{"epoch": i, "train_mae_eV": .1, "validation_mae_eV": .125 + (39-i)*.01, "elapsed_s": 300.0, "graphs_per_s": 333.0, "learning_rate": .00001} for i in range(40)]
            path = folder / "trace.json"
            self.write(path, {"epochs": trace})
            artifacts["trace"] = str(path.relative_to(self.root))
            artifacts["trace_sha256"] = ACCEPT.sha256(path)
            metrics = {"candidate": candidate, "complete": True, "source_commit": SOURCE, "input_cache_aggregate_sha256": ACCEPT.CACHE, "seed": 42, "platform_contract": ACCEPT.EXPECTED_CONTRACT, "official_validation_role_read": False, "test_dev_role_read": False, "parameter_count": count, "validation_gap_mae_eV": .125, "epochs_completed": 40, "best_epoch": 39, "artifacts": artifacts}
            self.write(folder / "metrics.json", metrics)
            self.completion["runs"].append(metrics)
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def write(path, payload):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def save(self):
        self.write(self.root / "completion.json", self.completion)
        for metrics in self.completion["runs"]:
            self.write(self.root / "results" / metrics["candidate"] / "metrics.json", metrics)

    def test_valid_artifacts(self):
        report = ACCEPT.accept(self.root, SOURCE)
        self.assertTrue(report["accepted"], report["errors"])
        self.assertFalse(report["model_inference_executed"])

    def test_corrupt_checkpoint_rejected(self):
        path = self.root / self.completion["runs"][0]["artifacts"]["checkpoint"]
        path.write_bytes(b"corrupt")
        self.assertFalse(ACCEPT.accept(self.root, SOURCE)["accepted"])

    def test_role_change_rejected(self):
        self.completion["test_dev_role_read"] = True
        self.save()
        self.assertFalse(ACCEPT.accept(self.root, SOURCE)["accepted"])

    def test_early_incomplete_rejected(self):
        row = self.completion["runs"][1]
        path = self.root / row["artifacts"]["trace"]
        trace = json.loads(path.read_text())["epochs"][:3]
        self.write(path, {"epochs": trace})
        row["artifacts"]["trace_sha256"] = ACCEPT.sha256(path)
        row["epochs_completed"] = 3
        self.save()
        self.assertFalse(ACCEPT.accept(self.root, SOURCE)["accepted"])

    def test_false_score_rejected(self):
        self.completion["runs"][1]["validation_gap_mae_eV"] = .01
        self.save()
        self.assertFalse(ACCEPT.accept(self.root, SOURCE)["accepted"])

    def test_source_mismatch_rejected(self):
        self.assertFalse(ACCEPT.accept(self.root, "b" * 40)["accepted"])

    def test_trainer_durability_source(self):
        source = (ROOT / "src/molgap/pcqm_local_global_runner.py").read_text(encoding="utf-8")
        ast.parse(source)
        for key in ("python", "numpy", "torch_cpu", "torch_device", "loader"):
            self.assertIn(f'rng["{key}"]', source)
        self.assertIn('scheduler = torch.optim.lr_scheduler.CosineAnnealingLR', source)
        self.assertIn('normalized_target = (batch.y.view(-1, 1) - mean_tensor) / std_tensor', source)


if __name__ == "__main__":
    unittest.main()
