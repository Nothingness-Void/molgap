"""Synthetic CPU graphs only; no real data, A100 allocation or role access."""
import copy
import json
import math
import time

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import global_mean_pool

from molgap import colab_k1_screen as screen
from molgap import pcqm_k1_scale_runner as data_owner


class ToyK1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)
        self.dropout = torch.nn.Dropout(0.2)
        self.head = torch.nn.Linear(2, 1)

    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
        return self.head(global_mean_pool(self.dropout(self.bn(x)), batch))


def graphs(start, rows):
    return [Data(x=torch.tensor([[i / 100 + 1, 0.2], [0.4, i / 100 + 2]]),
                 edge_index=torch.tensor([[0, 1], [1, 0]]),
                 edge_attr=torch.zeros(2, 3, dtype=torch.long),
                 random_walk_pe=torch.zeros(2, 16),
                 y=torch.tensor([i / 100 + 0.5]), source_idx=torch.tensor([i]))
            for i in range(start, start + rows)]


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setenv("MOLGAP_V4_LOADER_WORKERS", "0")
    monkeypatch.setattr(screen, "TRAIN_ROWS", 256)
    monkeypatch.setattr(screen, "DEV_ROWS", 128)
    monkeypatch.setattr(screen, "CALIBRATION_ROWS", 128)
    monkeypatch.setattr(screen.owner, "EPOCHS", 2)
    monkeypatch.setattr(screen.owner, "schedule", lambda epoch:
                        1e-6 + (4e-4 - 1e-6) * (1 + math.cos(math.pi * epoch / 59)) / 2)
    roles = {"train": graphs(0, 256), "validation": graphs(256, 128)}
    torch.manual_seed(9)
    initial = copy.deepcopy(ToyK1().state_dict())

    def arms():
        values = {}
        for name in screen.ARMS:
            model = ToyK1()
            model.load_state_dict(initial)
            screen.configure_fp32_determinism(42)
            values[name] = screen._arm(model)
            if name == "ema999":
                values[name]["ema"] = screen.make_ema(model)
        return values
    return roles, arms


def run_pair(arms, roles, output, deadline=float("inf"), checkpoint_steps=1):
    return screen._paired(arms, roles, 1.0, 0.8, output, {"toy": True}, "toy-runtime",
                          "cpu", deadline, time.time(), checkpoint_steps)


def test_first_last_step_repeat_and_exact_atomic_resume(toy, tmp_path):
    roles, make = toy
    arms = make()
    before = {name: screen._snapshot(arm) for name, arm in arms.items()}
    rng = screen.capture_rng_state()
    report = screen._qualify(arms, roles["train"], 1.0, 0.8, tmp_path, "cpu", float("inf"))
    assert screen._exact(rng, screen.capture_rng_state())
    for name, arm in arms.items():
        assert screen._exact(before[name], screen._snapshot(arm))
        assert report[name]["exact_resume"]
        assert report[name]["formal_samples"] == 0
        assert len(report[name]["fixture_sha256"]) == 2
        assert all(math.isfinite(v) for v in report[name]["repeat"]["losses"])


def test_parameter_ema_copies_not_averages_buffers(toy):
    _, make = toy
    arm = make()["ema999"]
    before = dict(arm["ema"].named_parameters())["head.weight"].clone()
    with torch.no_grad():
        arm["model"].head.weight.add_(1)
        arm["model"].bn.running_mean.fill_(7)
        arm["model"].bn.num_batches_tracked.fill_(9)
    screen.update_ema(arm["ema"], arm["model"])
    assert torch.allclose(arm["ema"].head.weight, before + 0.001)
    assert torch.equal(arm["ema"].bn.running_mean, arm["model"].bn.running_mean)
    assert int(arm["ema"].bn.num_batches_tracked) == 9


def test_clean_calibration_snapshot_inside_context_rng_and_live_unchanged(toy):
    roles, make = toy
    arm = make()["reference"]
    batch = next(iter(screen._train_loader(roles["train"], 0)))
    screen._step(arm, batch, 1, 0.8, "cpu", float("inf"))
    before = screen._snapshot(arm)
    rng = screen.capture_rng_state()
    raw, calibrated, selected, report = screen._endpoint(arm, roles, 1, 0.8, "cpu", float("inf"))
    assert screen._exact(before, screen._snapshot(arm))
    assert screen._exact(rng, screen.capture_rng_state())
    assert not torch.equal(selected["bn.running_mean"], before["model"]["bn.running_mean"])
    assert report["rows"] == 128 and report["dropout_disabled"]
    assert report["parameters_unchanged"] and report["non_bn_buffers_unchanged"]
    assert report["buffers_restored"]
    assert raw["source_idx"].tolist() == calibrated["source_idx"].tolist() == list(range(256, 384))
    assert int(selected["bn.num_batches_tracked"]) == 1


def test_pair_same_live_training_order_initial_rng_and_full_lr_horizon(toy, tmp_path):
    roles, make = toy
    arms = make()
    result = run_pair(arms, roles, tmp_path)
    assert result["status"] == "COMPLETE" and result["complete"]
    assert result["matched_completed_epochs"] == 2
    assert all(item is not None and item["epoch"] < 2 for item in result["matched_prefix_selection"].values())
    assert screen._exact(arms["reference"]["model"].state_dict(), arms["ema999"]["model"].state_dict())
    assert screen._exact(arms["reference"]["rng"], arms["ema999"]["rng"])
    for name, arm in arms.items():
        assert arm["steps"] == 4 and arm["samples"] == 512
        assert [r["lr"] for r in arm["trace"]] == [screen.owner.schedule(0), screen.owner.schedule(1)]
        selected = torch.load(tmp_path / name / "best.pt", weights_only=False)
        assert selected["selection"] == ("live-calibrated" if name == "reference" else "ema-calibrated")
        assert int(selected["model"]["bn.num_batches_tracked"]) == 1
    assert [r["order_sha256"] for r in arms["reference"]["trace"]] == [r["order_sha256"] for r in arms["ema999"]["trace"]]
    assert not list(tmp_path.rglob("*.tmp"))


def test_mid_epoch_deadline_checkpoint_resumes_exactly(toy, tmp_path, monkeypatch):
    roles, make = toy
    continuous = make()
    run_pair(continuous, roles, tmp_path / "continuous")
    interrupted = make()
    actual_step = screen._step
    calls = []

    def stop_after_one(*args, **kwargs):
        if calls:
            raise TimeoutError("synthetic watchdog")
        calls.append(True)
        return actual_step(*args, **kwargs)

    monkeypatch.setattr(screen, "_step", stop_after_one)
    result = run_pair(interrupted, roles, tmp_path / "partial")
    assert result["status"] == "STOP_FOR_COST" and not result["complete"]
    assert result["matched_completed_epochs"] == 0
    assert result["arms"]["reference"]["offset"] == 1
    path = tmp_path / "partial" / "last.pt"
    resumed = make()
    digest = screen.sha256_file(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        screen._resume(path, "0" * 64, resumed, {"toy": True}, "toy-runtime", 256)
    with pytest.raises(ValueError, match="identity changed"):
        screen._resume(path, digest, resumed, {"toy": False}, "toy-runtime", 256)
    screen._resume(path, digest, resumed, {"toy": True}, "toy-runtime", 256)
    monkeypatch.setattr(screen, "_step", actual_step)
    run_pair(resumed, roles, tmp_path / "partial")
    for name in screen.ARMS:
        for key in ("model", "optimizer", "ema", "rng", "steps", "samples", "best", "best_epoch"):
            assert screen._exact(screen._snapshot(continuous[name])[key], screen._snapshot(resumed[name])[key]), key


def test_timeout_during_calibration_preserves_full_train_cursor(toy, tmp_path, monkeypatch):
    roles, make = toy
    arms = make()
    endpoint = screen._endpoint
    monkeypatch.setattr(screen, "_endpoint", lambda *args: (_ for _ in ()).throw(TimeoutError("calibration")))
    result = run_pair(arms, roles, tmp_path)
    assert result["status"] == "STOP_FOR_COST" and not result["complete"]
    assert result["arms"]["reference"]["offset"] == 2
    assert result["arms"]["reference"]["epoch"] == 0
    assert result["arms"]["reference"]["samples"] == 256
    assert result["matched_completed_epochs"] == 0
    path = tmp_path / "last.pt"
    resumed = make()
    screen._resume(path, screen.sha256_file(path), resumed, {"toy": True}, "toy-runtime", 256)
    assert list(screen._train_loader(roles["train"], 0, 2)) == []
    monkeypatch.setattr(screen, "_endpoint", endpoint)
    assert run_pair(resumed, roles, tmp_path)["complete"]
    assert resumed["reference"]["steps"] == resumed["ema999"]["steps"] == 4


def test_expired_pair_saves_stop_without_training(toy, tmp_path, monkeypatch):
    roles, make = toy
    monkeypatch.setattr(screen, "_step", lambda *args: pytest.fail("expired pair trained"))
    result = run_pair(make(), roles, tmp_path, deadline=float("-inf"))
    assert result["status"] == "STOP_FOR_COST"
    assert all(item["samples"] == 0 for item in result["arms"].values())


def test_leading_reference_not_eligible_until_candidate_matches(toy, tmp_path, monkeypatch):
    roles, make = toy
    endpoint = screen._endpoint

    def stop_candidate(arm, *args):
        if arm["ema"] is not None:
            raise TimeoutError("budget during candidate evaluation")
        return endpoint(arm, *args)

    monkeypatch.setattr(screen, "_endpoint", stop_candidate)
    result = run_pair(make(), roles, tmp_path)
    assert result["arms"]["reference"]["epoch"] == 1
    assert result["matched_completed_epochs"] == 0
    assert result["matched_prefix_selection"] == dict.fromkeys(screen.ARMS)


def test_resume_repairs_best_mirror_from_last_checkpoint(toy, tmp_path, monkeypatch):
    roles, make = toy
    arms = make()
    run_pair(arms, roles, tmp_path)
    path = tmp_path / "last.pt"
    resumed = make()
    screen._resume(path, screen.sha256_file(path), resumed, {"toy": True}, "toy-runtime", 256)
    screen.atomic_torch_save(tmp_path / "reference" / "best.pt", {"uncommitted": True})
    run_pair(resumed, roles, tmp_path)
    repaired = torch.load(tmp_path / "reference" / "best.pt", weights_only=False)
    assert repaired["epoch"] == resumed["reference"]["best_epoch"]
    assert screen._exact(repaired["model"], resumed["reference"]["best_payload"]["model"])


def test_raw_metric_never_selects_best(toy, tmp_path, monkeypatch):
    roles, make = toy
    endpoint = screen._endpoint
    calls = []

    def modified(*args):
        raw, cal, selected, report = endpoint(*args)
        epoch = args[0]["epoch"]
        raw["mae_eV"] = 10 - epoch
        cal["mae_eV"] = 1 + epoch
        calls.append(args[0]["epoch"])
        return raw, cal, selected, report

    monkeypatch.setattr(screen, "_endpoint", modified)
    arms = make()
    run_pair(arms, roles, tmp_path)
    assert calls == [0, 0, 1, 1]
    assert all(a["best_epoch"] == 0 and a["best"] == 1 for a in arms.values())


def test_real_data_owner_reused_with_synthetic_packed_shards(toy, tmp_path, monkeypatch):
    import hashlib
    from torch.utils.data import ConcatDataset
    roles, _ = toy
    shards = []
    fixed, scnet = hashlib.sha256(), hashlib.sha256()
    for role, graphs_ in (("train", roles["train"]), ("development", roles["validation"])):
        for graph in graphs_:
            graph.pos = torch.zeros(graph.num_nodes, 3)
        path = tmp_path / f"{role}.pt"
        screen.atomic_torch_save(path, InMemoryDataset.collate(graphs_))
        digest = screen.sha256_file(path)
        shards.append({"file": path.name, "sha256": digest, "rows": len(graphs_), "role": role})
        fixed.update(f"{role}\tstore/geometry/{path.name}\t{digest}\n".encode())
        scnet.update(f"{role}\t{path.name}\t{digest}\n".encode())
    manifest = {"geometry_shards": shards}
    monkeypatch.setattr(data_owner, "SCALE_TRAIN_ROWS", 256)
    monkeypatch.setattr(data_owner, "VALIDATION_ROWS", 128)
    monkeypatch.setattr(data_owner, "FIXED_500K_GEOMETRY_SHA256", fixed.hexdigest())
    monkeypatch.setattr(data_owner, "SCNET_REFERENCE_CACHE_SHA256", scnet.hexdigest())
    monkeypatch.setattr(data_owner, "find_cache", lambda digest: (tmp_path, manifest))
    accepted, mean, std, _ = screen._data(tmp_path)
    assert isinstance(accepted["train"], ConcatDataset)
    assert "pos" not in accepted["train"][0]
    expected = torch.cat([g.y for g in roles["train"]]).double()
    assert mean == float(expected.mean()) and std == float(expected.std(unbiased=True))
    manifest["geometry_shards"][0]["role"] = "official_validation"
    with pytest.raises(ValueError, match="internal roles"):
        screen._data(tmp_path)


def test_nonfinite_forward_and_geometry_rejected(toy):
    roles, make = toy
    arm = make()["reference"]
    batch = next(iter(screen._train_loader(roles["train"], 0)))
    batch.pos = torch.zeros(batch.num_nodes, 3)
    with pytest.raises(ValueError, match="Geometry"):
        screen._step(arm, batch, 1, 1, "cpu", float("inf"))
    del batch.pos
    with torch.no_grad():
        arm["model"].head.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        screen._step(arm, batch, 1, 1, "cpu", float("inf"))


def test_full_sixty_epoch_schedule_unchanged():
    assert screen.owner.EPOCHS == 60
    assert screen.owner.schedule(0) == pytest.approx(4e-4)
    assert screen.owner.schedule(59) == pytest.approx(1e-6)


def test_atomic_retention_does_not_replace_changed_provenance(tmp_path):
    source, target = tmp_path / "source", tmp_path / "retained"
    source.write_bytes(b"synthetic original source")
    digest = screen.sha256_file(source)
    screen._retain(source, target, digest)
    assert screen.sha256_file(target) == digest
    source.write_bytes(b"different source")
    with pytest.raises(ValueError, match="Retained input hash"):
        screen._retain(source, target, screen.sha256_file(source))
    assert screen.sha256_file(target) == digest


def config_inputs(tmp_path, **changes):
    source, initial, config_path = (tmp_path / name for name in ("source.bin", "initial.pt", "config.json"))
    source.write_bytes(b"toy source archive")
    initial.write_bytes(b"toy initialization; never deserialized")
    config = {"format": screen.FORMAT, "source_commit": "a" * 40,
              "source_package_sha256": screen.sha256_file(source), "job_id": "synthetic-job",
              "cpu_accepted_dataset_manifest_sha256": screen.FIXED_500K_MANIFEST_SHA256,
              "initial_format": "synthetic", "initial_state_sha256": "b" * 64,
              "allocation_started_unix": time.time()}
    config.update(changes)
    config_path.write_text(json.dumps(config))
    return dict(mode="preflight", dataset_root=tmp_path / "not-read", initial_path=initial,
                initial_sha256=screen.sha256_file(initial), runconfig_path=config_path,
                runconfig_sha256=screen.sha256_file(config_path), source_archive=source,
                output=tmp_path / "out", deadline=config["allocation_started_unix"] + 14400)


@pytest.mark.parametrize("change,message", [
    ({"cpu_accepted_dataset_manifest_sha256": "0" * 64}, "CPU acceptance"),
    ({"source_commit": "unknown"}, "Git SHA"),
    ({"extra": "not allowed"}, "schema"),
])
def test_cli_config_fails_closed_before_data_or_gpu(tmp_path, change, message, monkeypatch):
    monkeypatch.setattr(screen, "_data", lambda *args: pytest.fail("real data must not be read"))
    with pytest.raises(ValueError, match=message):
        screen.execute(**config_inputs(tmp_path, **change))


def test_original_total_allocation_deadline_not_reset(tmp_path):
    inputs = config_inputs(tmp_path)
    inputs["deadline"] += 1
    with pytest.raises(ValueError, match="four-hour total"):
        screen.execute(**inputs)


def test_expired_preflight_never_reads_dataset(tmp_path, monkeypatch):
    inputs = config_inputs(tmp_path, allocation_started_unix=time.time() - 14400)
    inputs["deadline"] = time.time() - 1
    monkeypatch.setattr(screen, "_data", lambda *args: pytest.fail("expired preflight data read"))
    result = screen.execute(**inputs)
    assert result["status"] == "STOP_FOR_COST" and not result["complete"]
    assert not result["official_validation_role_read"]


def test_preflight_failure_is_no_train(tmp_path, monkeypatch):
    inputs = config_inputs(tmp_path)
    monkeypatch.setattr(screen, "_data", lambda *args: (_ for _ in ()).throw(ValueError("synthetic invalid cache")))
    with pytest.raises(ValueError, match="invalid cache"):
        screen.execute(**inputs)
    terminal = json.loads((inputs["output"] / "terminal.json").read_text())
    assert terminal["status"] == "NO_TRAIN" and not terminal["complete"]


def test_explicit_source_allowlist_import_and_cli_smoke(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    modules = (
        "__init__ constants colab_k1_screen edge_state_training_core edge_state_adapter "
        "experiment_spec pcqm_topology k1_bn_calibration k1_screen_training k1_weight_ema "
        "pcqm_500k_v4_evidence pcqm_k1_scale_runner pcqm_k1_scale pcqm_gap_data "
        "pcqm_gptrans_v4 qm9_neural_atom pcqm_gap_architecture gps futility_gate "
        "screen_policy training_reproducibility v4_runtime pcqm_wedge"
    ).split()
    package = tmp_path / "src" / "molgap"
    package.mkdir(parents=True)
    for name in modules:
        shutil.copy2(Path(screen.__file__).parent / f"{name}.py", package / f"{name}.py")
    environment = {**os.environ, "PYTHONPATH": str(tmp_path / "src")}
    result = subprocess.run([sys.executable, "-m", "molgap.colab_k1_screen", "--help"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--runconfig-sha256" in result.stdout and "--deadline" in result.stdout
    result = subprocess.run([sys.executable, "-c", "import molgap.pcqm_gap_architecture; import molgap.pcqm_wedge"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
