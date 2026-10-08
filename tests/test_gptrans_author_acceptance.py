"""Fail-closed saved-input gates using tiny JSON fixtures only."""
import json

import pytest

from molgap.gptrans_author_acceptance import accept_prepared_inputs, accept_training_outputs, NO_READ


def fixture_inputs(tmp_path):
    output, package = tmp_path / "output", tmp_path / "package"
    output.mkdir()
    package.mkdir()
    release = {"status": "LOCAL_RELEASE_INPUTS_VERIFIED", "errors": [],
               "source_commit": "abc", "source_archive_sha256": "a" * 64}
    startup = {**{key: False for key in NO_READ}, "source_commit": "abc",
               "source_archive_sha256": "a" * 64,
               "manifest_sha256": "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d",
               "initial_state_file_sha256": "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"}
    cost = {"status": "COMPLETE", "allocated_gpu_count": 0,
            "wall_seconds": 1, "process_cpu_seconds": 1}
    for path, record in ((package / "release.json", release), (output / "startup.json", startup),
                         (output / "preparation_result.json", {key: False for key in NO_READ}),
                         (output / "native_cost.json", cost)):
        path.write_text(json.dumps(record))
    return output, package, startup, cost


def test_rejects_unqualified_source(tmp_path):
    output, package, _, _ = fixture_inputs(tmp_path)
    (package / "release.json").write_text(json.dumps({"status": "FAILED", "errors": ["source"]}))
    with pytest.raises(ValueError, match="qualified"):
        accept_prepared_inputs(output, package)


def test_rejects_wrong_actual_source(tmp_path):
    output, package, startup, _ = fixture_inputs(tmp_path)
    startup["source_commit"] = "wrong"
    (output / "startup.json").write_text(json.dumps(startup))
    with pytest.raises(ValueError, match="identities"):
        accept_prepared_inputs(output, package)


def test_rejects_label_access(tmp_path):
    output, package, startup, _ = fixture_inputs(tmp_path)
    startup["labels_read"] = True
    (output / "startup.json").write_text(json.dumps(startup))
    with pytest.raises(ValueError, match="forbidden"):
        accept_prepared_inputs(output, package)


@pytest.mark.parametrize("wall", [float("nan"), 10801])
def test_rejects_invalid_or_excess_cost(tmp_path, wall):
    output, package, _, cost = fixture_inputs(tmp_path)
    cost["wall_seconds"] = wall
    (output / "native_cost.json").write_text(json.dumps(cost))
    with pytest.raises(ValueError, match="CPU"):
        accept_prepared_inputs(output, package)


def test_training_acceptance_rejects_incomplete_actual_source_package(tmp_path, monkeypatch):
    from pathlib import Path
    import shutil
    root = Path(__file__).resolve().parents[1]
    gpu = tmp_path / "experiments/pcqm_gptrans_author_alignment/gpu"
    gpu.mkdir(parents=True)
    for name in ("spec.json", "submission_v1.json", "screen_config.json"):
        shutil.copyfile(root / "experiments/pcqm_gptrans_author_alignment/gpu" / name, gpu / name)
    package = tmp_path / "incomplete_package"
    package.mkdir()
    monkeypatch.setattr("torch.load", lambda *a, **k: pytest.fail("No tensor read before package acceptance"))
    with pytest.raises(ValueError, match="six declared files"):
        accept_training_outputs(tmp_path, tmp_path / "records", package)


def test_terminal_adapter_rejects_unaccepted_outputs_before_closure(tmp_path, monkeypatch):
    from molgap.gptrans_author_terminal import close_author_outputs
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text('{"accepted":false}')
    monkeypatch.setattr("molgap.gptrans_author_terminal.close_terminal_arm",
        lambda *a, **k: pytest.fail("Unaccepted outputs cannot finalize"))
    with pytest.raises(ValueError, match="Independent saved-output"):
        close_author_outputs(tmp_path, tmp_path / "records", acceptance)


@pytest.mark.parametrize("mode", [
    "degree_bond_local_cap_ema999", "degree_local_connected_pair_ema999",
    "degree_local_bond_return_ema999",
])
def test_native_control_diagnostics_do_not_compete_with_canonical_replay_trace(mode):
    from molgap.gptrans_author_acceptance import DIAGNOSTIC_TRACE_MODES
    from molgap.research_memory.terminal_wiring import inspect_trace_retention_evidence
    assert mode in DIAGNOSTIC_TRACE_MODES
    prefix = f"screen/{mode}/training/"
    canonical = {"name": "canonical_trace.json", "locator": prefix + "canonical_trace.json"}
    diagnostic = {"name": "trace.json", "locator": prefix + "trace.json",
                  "purpose": "analysis_diagnostics"}
    assert inspect_trace_retention_evidence({"artifacts": [canonical, diagnostic]},
        mode) == (True, canonical)
    del diagnostic["purpose"]
    with pytest.raises(ValueError, match="ambiguous trace retention"):
        inspect_trace_retention_evidence({"artifacts": [canonical, diagnostic]},
            mode)
