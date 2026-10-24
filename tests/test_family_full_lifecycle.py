"""Synthetic same-runtime state/BN/EMA resume and portable output acceptance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import subprocess
import tarfile
import sys

import pytest
import torch

from molgap.family_training_core import optimizer_update
from molgap.family_training_kernel import FamilyKernel, make_family_optimizer
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_session import FamilyTrainingSession
from molgap.family_training_state import CalibrationPolicy, source_order_sha256
from molgap.gptrans_execution_hotpath import BoundForeachEMA, ReusedWeightView
from molgap.gptrans_execution_tuning import ExecutionTuning
from test_family_training_session import assert_equal, events, make_session
from test_family_training_runner import contract_file, context, PROFILES, SessionOutputs
from test_experiment_spec import payload


class StateKernel(FamilyKernel):
    def __init__(self, recipe):
        self.recipe, self.device = recipe, torch.device("cpu")
        self.model = self.make_clean_model()
        self.optimizer = make_family_optimizer(recipe, self.model.parameters())
        self.ema = BoundForeachEMA(self.model, recipe.ema_decay)
        self.weight_view = ReusedWeightView()

    def make_clean_model(self):
        return torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.BatchNorm1d(3),
                                   torch.nn.Dropout(.2), torch.nn.Linear(3, 1))

    def forward(self, batch, *, model=None):
        return (self.model if model is None else model)(batch.source_idx.float().reshape(-1, 1) / 257.).reshape(-1)

    def step(self, batch, *, check_finite):
        def objective():
            absolute = (self.forward(batch) - (batch.y.reshape(-1) - self.recipe.target_mean_eV) /
                        self.recipe.target_std_eV).abs()
            return absolute.mean(), (absolute.detach().sum(), int(batch.num_graphs))
        loss, (absolute, rows) = optimizer_update(self.model, self.optimizer, objective,
            clip_norm=self.recipe.gradient_clip, check_finite=check_finite)
        self.training_state.update_raw_ema()
        return loss, absolute, rows


def clean_session(family, development, log):
    template = make_session(family, development=development, schedule_kind="cosine_step", events=events())
    data = template.loaders.data
    policy = CalibrationPolicy(template.recipe.dataset.manifest_sha256, "3" * 64,
        source_order_sha256(torch.arange(128)), 1)
    recipe = replace(template.recipe, clean_bn=policy,
        selection="best-development-clean-ema" if development else "final-clean-ema")
    template.close()
    return FamilyTrainingSession(StateKernel(recipe), FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0)),
        checkpoint=lambda progress, state: log["checkpoint"].append(deepcopy(state)),
        selected=lambda *args: log["selected"].append(deepcopy(args)),
        pass_finished=lambda *args: log["passes"].append(deepcopy(args)))


@pytest.mark.parametrize("family", ["neural_atom_k1", "gptrans_t"])
@pytest.mark.parametrize("development", [False, True])
@pytest.mark.parametrize("pause_step", [1, 3])
def test_clean_state_midpass_resume_matches_all_state_and_metrics(family, development, pause_step):
    direct_log, split_log = events(), events()
    direct = clean_session(family, development, direct_log)
    direct.run()
    first = clean_session(family, development, split_log)
    first.run(should_pause=lambda: first.kernel.training_state.raw_revision >= pause_step)
    retained = split_log["checkpoint"][-1]
    if development and pause_step == 3:
        assert retained["cursor"]["weight_state"]["clean"]["stale"]
    restored = clean_session(family, development, split_log)
    restored.load_state_dict(retained)
    restored.run()
    assert_equal(direct.state_dict(), restored.state_dict())
    assert_equal(direct_log["passes"], split_log["passes"])
    assert_equal(direct_log["selected"], split_log["selected"])
    component = restored.state_dict()["cursor"]["weight_state"]
    assert component["raw_revision"] == restored.recipe.optimizer_steps
    assert component["clean"] is not None and not component["clean"]["stale"]
    assert all(selected[1] == "clean-ema" for selected in split_log["selected"])


@pytest.mark.parametrize("defect", ["revision", "identity", "buffer"])
def test_corrupt_clean_resume_rejects_without_partial_mutation(defect):
    log = events()
    first = clean_session("neural_atom_k1", True, log)
    first.run(should_pause=lambda: first.kernel.training_state.raw_revision >= 3)
    state = deepcopy(log["checkpoint"][-1])
    component = state["cursor"]["weight_state"]
    if defect == "revision":
        component["raw_revision"] += 1
    elif defect == "identity":
        component["clean"]["calibration"]["policy_identity"] = "0" * 64
    else:
        component["clean"]["model"]["1.running_mean"].add_(1)
    fresh = clean_session("neural_atom_k1", True, events())
    before = deepcopy(fresh.state_dict())
    try:
        with pytest.raises(ValueError):
            fresh.load_state_dict(state)
        assert_equal(before, fresh.state_dict())
    finally:
        fresh.close()


@pytest.mark.parametrize("development", [False, True])
def test_clean_endpoint_uses_existing_publication_and_inspector(tmp_path, development):
    session = clean_session("neural_atom_k1", development, events())
    path, record = contract_file(tmp_path, session)
    ctx = context(session.recipe.family, path)
    outputs = SessionOutputs(tmp_path / "output", ctx, contract=path,
        adapter=PROFILES[session.recipe.family], trajectory_id="synthetic-clean",
        train_role_identity=record["train_role_identity"])
    session.loaders.attach_role_audit(outputs.role_audit)
    session.checkpoint, session.selected, session.pass_finished = outputs.checkpoint, outputs.selected, outputs.pass_finished
    progress = session.run()
    report = outputs.complete(progress, runtime={"platform": ctx.platform, "account": ctx.account,
        "precision": "fp32", "source_commit": ctx.source_commit, "source_archive_sha256": ctx.source_archive_sha256},
        costs=[{"metric": "wall_seconds", "unit": "seconds", "status": "measured", "value": 1.,
                "semantics": "process_wall", "hardware": "synthetic-cpu"},
               {"metric": "device_seconds", "unit": "seconds", "status": "missing", "value": None,
                "semantics": "allocated_device", "hardware": "synthetic-cpu"}])
    assert report["status"] == "MECHANICALLY_VERIFIED", report
    assert report["observed"]["selected_weights"] == "clean-ema"
    assert report["scientific_acceptance"] == "NOT_EVALUATED"


def test_shared_package_imports_in_isolation_without_host_training_paths(tmp_path, payload):
    from molgap.experiment_source_inventory import runtime_source_files
    root = Path(__file__).resolve().parents[1]
    source_root = tmp_path / "repo"
    # The distributed test bundle has no Git metadata. Build the independent
    # fixture first, then select the production inventory from that Git tree.
    sources = [path for path in (root / "src/molgap").rglob("*.py")
               if "archive" not in path.relative_to(root).parts]
    sources.append(root / "platforms/kaggle/run_experiment.py")
    for source in sources:
        name = source.relative_to(root)
        destination = source_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    from molgap.experiment_spec import ExperimentSpec
    from molgap.experiment_package import build_experiment_source_package, verify_experiment_source_package
    for arguments in (("init",), ("config", "core.autocrlf", "false"),
                      ("config", "user.name", "Synthetic shared fixture"),
                      ("config", "user.email", "synthetic@example.invalid"), ("add", "."),
                      ("-c", "commit.gpgsign=false", "commit", "-m", "Synthetic shared source")):
        subprocess.run(["git", *arguments], cwd=source_root, check=True, capture_output=True)
    names = runtime_source_files(source_root)
    assert "src/molgap/training/__init__.py" in names
    assert "src/molgap/training/state.py" in names
    package = tmp_path / "package"
    manifest = build_experiment_source_package(ExperimentSpec(payload), source_root, names, package)
    assert verify_experiment_source_package(package) == manifest
    with tarfile.open(package / "source.tar.gz") as archive:
        for member in archive.getmembers():
            assert member.isfile() and member.name in names
            destination = tmp_path / "isolated" / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
    result = subprocess.run([sys.executable, "-I", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from molgap.family_training_runner import read_recipe; "
        "from molgap.common import runtime; "
        "from molgap.substrate_scale_addon import SlotScale; "
        "from molgap.family_training_state import FamilyTrainingState as LegacyState; "
        "from molgap.training.state import FamilyTrainingState as CanonicalState; "
        "assert LegacyState is CanonicalState is runtime.FamilyTrainingState; "
        "from pathlib import Path; "
        "assert all(Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()) "
        "for name, module in sys.modules.items() if name == 'molgap' or name.startswith('molgap.')); "
        "print('isolated shared package imported')", str(tmp_path / "isolated/src")],
        cwd=tmp_path, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "isolated shared package imported"
