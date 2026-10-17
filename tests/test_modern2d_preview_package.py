"""Committed-source isolation checks for the Modern2D preview model closures."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from molgap.source_dependencies import discover_source_dependencies
from molgap.v4_bundle import _payload, build_v4_source_bundle, committed_payloads


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SOURCE_TOKENS = ("vendor", "lightning", "xformers", "trainer")
FORBIDDEN_EXTERNAL_MODULES = {
    "admin_torch", "flash_attn", "lightning", "pytorch_lightning", "wandb", "xformers",
}


MODEL_CASES = (
    pytest.param(
        "ppgt_author",
        {
            "hidden": 16,
            "heads": 2,
            "blocks": 1,
            "pair_dim": 8,
            "pe_mlp_hidden": 16,
            "stem_blocks": 1,
            "rrwp_steps": 3,
            "spe_bases": 1,
        },
        id="ppgt-author",
    ),
    pytest.param(
        "esa_native",
        {
            "hidden": 16,
            "heads": 2,
            "layer_types": ("M", "S", "P"),
            "graph_dim": 16,
            "mlp_hidden": 32,
            "seeds": 3,
        },
        id="esa-native",
    ),
)


def _isolated_cpu_script(module: str, config: dict) -> str:
    config_json = json.dumps(config, separators=(",", ":"))
    return f"""
import sys
sys.path.insert(0, sys.argv[1])
import torch
from torch_geometric.data import Batch, Data
from molgap.{module} import make_model

torch.set_num_threads(1)
torch.manual_seed(42)
model = make_model({{"model_config": {config_json}}}).eval()
assert all(parameter.device.type == "cpu" for parameter in model.parameters())

def graph(source):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.zeros((3, 9), dtype=torch.long)
    x[:, 0] = torch.tensor([5, 6, 7], dtype=torch.long)
    edge_attr = torch.zeros((4, 3), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                source_idx=torch.tensor([source], dtype=torch.long))

batch = Batch.from_data_list([graph(index) for index in range(128)])
with torch.no_grad():
    output = model(batch)
assert output.shape == (128,)
assert torch.isfinite(output).all()

loaded = set(sys.modules)
forbidden_loaded = sorted(
    name for name in loaded
    if name == "wandb"
    or name.startswith(("admin_torch", "flash_attn", "lightning", "pytorch_lightning", "xformers"))
    or name.startswith(("molgap.training", "molgap.family_training", "molgap.experiment_training_worker"))
)
assert not forbidden_loaded, forbidden_loaded
print("ISOLATED_CPU_128_OK", output.shape[0], output.dtype, "forbidden_loaded=", forbidden_loaded)
"""


def _meta_default_parameter_count(module: str) -> int:
    """Measure the default model shape without allocating default CPU tensors."""
    import torch

    with torch.device("meta"):
        model = importlib.import_module(f"molgap.{module}").make_model({})
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    return sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize("module,config", MODEL_CASES)
def test_committed_modern2d_source_bundle_isolated_import_and_cpu128(
    tmp_path, module, config, record_property
):
    closure = discover_source_dependencies(
        ROOT, [f"src/molgap/{module}.py"], require_static=True
    )
    assert not any(
        any(token in path.lower() for token in FORBIDDEN_SOURCE_TOKENS)
        for path in closure.source_files
    )
    assert FORBIDDEN_EXTERNAL_MODULES.isdisjoint(closure.external_modules)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output_dir = tmp_path / "bundle"
    bundle = build_v4_source_bundle(
        repo_root=ROOT,
        relative_paths=closure.source_files,
        output_dir=output_dir,
        source_commit=commit,
    )
    archive_path = output_dir / "source.tar.gz"
    archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert archive_sha256 == bundle["archive_sha256"]

    committed = committed_payloads(ROOT, commit, closure.source_files)
    with tarfile.open(archive_path) as archive:
        assert archive.getnames() == list(closure.source_files)
        for member in archive.getmembers():
            assert member.isfile() and member.name in closure.source_files
            payload = archive.extractfile(member).read()
            expected = _payload(ROOT / member.name)
            assert payload == expected, member.name
            assert payload == committed[member.name], member.name

    extracted = tmp_path / "isolated"
    with tarfile.open(archive_path) as archive:
        archive.extractall(extracted)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _isolated_cpu_script(module, config),
            str(extracted / "src"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    record_property("source_commit", commit)
    record_property("source_archive_sha256", archive_sha256)
    record_property("source_files", json.dumps(closure.source_files))
    record_property("default_parameter_count", str(_meta_default_parameter_count(module)))
    record_property("default_parameter_count_mode", "meta_only_measured")
    record_property("subprocess_returncode", str(result.returncode))
    record_property("subprocess_stdout", result.stdout.strip())
    record_property("subprocess_stderr", result.stderr.strip())
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ISOLATED_CPU_128_OK")
