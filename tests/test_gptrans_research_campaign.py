"""Tiny elementary kernels and metadata, not molecular/native qualification."""
from types import MethodType
from pathlib import Path
import json
import torch
from torch import nn
import pytest

from molgap import gptrans_research_campaign as research
from molgap.gptrans_capacity import BondLocalBlock
from molgap.gptrans_local_relation import INSERTIONS


def test_bounded_topology_layout_does_not_read_labels():
    graph = torch.tensor([0] * 40 + [1] * 4)
    source = torch.arange(44)
    target = source + 1
    layout = research.correspondence_layout({"batch_size": 2, "edges": (graph, source, target)})
    g, s, t = layout["correspondence_edges"]
    assert len(g) == 12 and (g == 0).sum() == 8 and (g == 1).sum() == 4
    assert torch.equal(t, s + 1)


def test_correspondence_loss_is_finite_symmetric_and_differentiable():
    atom, bond = torch.randn(8, 32, requires_grad=True), torch.randn(8, 32, requires_grad=True)
    loss = research.correspondence_loss(atom, bond)
    assert torch.equal(loss, research.correspondence_loss(bond, atom))
    loss.backward()
    assert torch.isfinite(atom.grad).all() and atom.grad.norm() > 0
    assert torch.isfinite(bond.grad).all() and bond.grad.norm() > 0
    assert research.correspondence_loss(atom[:0], bond[:0]) == 0


class TinyCore(nn.Module):
    def forward(self, node, pair, padding):
        return node + node.mean(dim=1, keepdim=True) * .1, pair


class Wrapped(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
    def forward(self, *args):
        return self.core(*args)


def tiny_factory(_mode, _state):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = nn.Module()
        model.node_channels = 2
        blocks = []
        for depth in range(1, 13):
            block = BondLocalBlock(TinyCore(), channels=2)
            with torch.no_grad():
                block.output.weight.fill_(.01)
            blocks.append(Wrapped(block) if depth in INSERTIONS else block)
        model.blocks = nn.ModuleList(blocks)
        model.readout = nn.Linear(34, 1)
        def dense(self, *args):
            return torch.arange(8).float().reshape(1, 4, 2) / 8, torch.arange(512).float().reshape(1, 32, 4, 4) / 512, None
        model._dense_inputs = MethodType(dense, model)
        model._local_edges = MethodType(lambda *args: (torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])), model)
        model._execution_layout = {"cycle_categories": torch.ones(1, 4, 4, dtype=torch.long),
            "correspondence_edges": (torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]), torch.tensor([2, 3, 1]))}
        return model


@pytest.mark.parametrize("mode", [research.CYCLE, research.CARRY, research.AUX])
def test_candidate_disabled_forward_matches_parent_and_batch_memory_resets(monkeypatch, mode):
    monkeypatch.setattr(research, "aggregate_factory", tiny_factory)
    model = research.construct(mode, {})
    reference = research.construct(research.BASE, {})
    args = (torch.empty(3, 1), None, None, None)
    model.eval()
    expected = reference(*args)
    assert torch.equal(model(*args), expected)
    assert torch.equal(model(*args), expected)
    assert model._carry_previous is None
    if mode == research.CARRY:
        model.train()
        model(*args).sum().backward()
        assert (model.bond_carry.logits.grad.abs() > 0).all()
    if mode == research.AUX:
        model.train()
        prediction = model(*args)
        assert torch.equal(prediction, expected)
        model._correspondence_loss.backward()
        assert model.atom_projection.weight.grad.norm() > 0
        assert model.bond_projection.weight.grad.norm() > 0
        model.eval()
        model(*args)
        assert model._correspondence_loss is None


def test_campaign_covers_two_slots_and_rejects_unauthorized_budget(tmp_path):
    from molgap.gptrans_campaign_release import BASE, check_campaign
    from molgap.experiment_spec import ExperimentSpec
    from molgap.training_reproducibility import atomic_json
    from molgap.v4_runtime import normalized_source_sha256
    root = Path(__file__).resolve().parents[1]
    campaign_path = root / BASE / "campaign.json"
    if not campaign_path.exists():
        pytest.skip("Frozen local campaign metadata absent")
    campaign = json.loads(campaign_path.read_bytes())
    states = {name: root / item["initial_state"]["path"] for name, item in campaign["arms"].items()}
    if not all(p.exists() for p in states.values()):
        pytest.skip("Ignored local initial tensors absent")
    plan = json.loads((root / BASE / "slot2/acceptance_plan.json").read_bytes())
    spec = ExperimentSpec.from_json((root / BASE / "slot2/scientific_spec.json").read_text())
    assert check_campaign(spec, root, plan)["reference_status"] == "PLANNED_NOT_ACCEPTED"
    from molgap.experiment_source_inventory import registered_source_files
    workflow = json.loads((root / BASE / "slot2/workflow_plan.json").read_bytes())
    registered_source_files(spec, [*workflow["source_files"], *(a["recipe"] for a in workflow["arms"])])
    pins = [campaign["authority"], campaign["audit"], campaign["cpu_acceptance"], *campaign["slots"],
            *(item["contract"] for item in campaign["arms"].values()), plan["campaign"]]
    for pin in pins:
        path = tmp_path / pin["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / pin["path"]).read_bytes().replace(b"\r\n", b"\n"))
    # Run the actual authority/recipe/parent validator on transported LF bytes,
    # not only on the original Windows checkout. No molecular forward occurs.
    assert check_campaign(spec, tmp_path, plan, initial_states=states)["reference_status"] == "PLANNED_NOT_ACCEPTED"
    authority = tmp_path / campaign["authority"]["path"]
    value = json.loads(authority.read_bytes())
    value["maximum_allocated_t4_hours"] = 64
    atomic_json(authority, value)
    campaign["authority"]["sha256"] = normalized_source_sha256(authority)
    atomic_json(tmp_path / BASE / "campaign.json", campaign)
    plan["campaign"]["sha256"] = normalized_source_sha256(tmp_path / BASE / "campaign.json")
    with pytest.raises(ValueError, match="authority"):
        check_campaign(spec, tmp_path, plan, initial_states=states)


def test_bootstrap_cost_reserve_preserves_old_identity_and_rejects_duplicate():
    from molgap.gptrans_campaign_release import retired_bootstrap_cost
    native = {"status": "failed", "cost_status": "measured", "hardware": ["Tesla T4"] * 2,
              "allocation_device_count": 2, "wall_seconds": 60., "allocated_device_seconds": 120.}
    attempt = {"kernel": "owner/slot", "version_number": 1, "spec_identity": "old-spec",
               "bootstrap_allocation": native}
    reserve = {"format": "molgap-campaign-retired-bootstrap-cost-v1", "attempts": [attempt]}
    assert retired_bootstrap_cost(None, 1, source_text=json.dumps(reserve)) == {
        "wall_seconds": 60., "allocated_device_seconds": 120.}
    reserve["attempts"].append(attempt)
    with pytest.raises(ValueError, match="duplicated"):
        retired_bootstrap_cost(None, 1, source_text=json.dumps(reserve))


def test_release_inventory_covers_worker_bootstrap_dependency_closure():
    import ast
    from molgap.experiment_source_inventory import SHARED_SOURCE_FILES
    root = Path(__file__).resolve().parents[1]
    pending = ["molgap.gptrans_execution_tuning"]
    seen = set()
    while pending:
        module = pending.pop()
        paths = [root / "src" / (module.replace(".", "/") + ".py"),
                 root / "src" / module.replace(".", "/") / "__init__.py"]
        path = next((p for p in paths if p.is_file()), None)
        if path is None or path in seen:
            continue
        seen.add(path)
        assert path.relative_to(root).as_posix() in SHARED_SOURCE_FILES
        parts = module.split(".") + (["__init__"] if path.name == "__init__.py" else [])
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                pending.extend(a.name for a in node.names if a.name.startswith("molgap."))
            elif isinstance(node, ast.ImportFrom):
                prefix = ".".join(parts[:-node.level]) if node.level else ""
                target = prefix + (("." + node.module) if prefix and node.module else (node.module or ""))
                if target == "molgap" or target.startswith("molgap."):
                    pending.append(target)
                    if node.module is None:
                        pending.extend(target + "." + a.name for a in node.names if a.name != "*")
