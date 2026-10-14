"""Synthetic CPU coverage for optional Clean-BN inside the shared K1 session."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as functional
from torch_geometric.data import Data

from molgap.family_training_core import optimizer_update
from molgap.family_training_data import DatasetSelection, TrainingData
from molgap.family_training_kernel import FamilyKernel, make_family_optimizer
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_outputs import SessionOutputs
from molgap.family_training_recipe import FamilyRecipe, RECIPE_FORMAT, ScheduleSpec
from molgap.family_training_runner import PROFILES, read_recipe
from molgap.family_training_session import FamilyTrainingSession, validate_session_resume
from molgap.family_training_state import CalibrationPolicy, source_order_sha256
from molgap.gptrans_execution_hotpath import BoundForeachEMA, ReusedWeightView
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import capture_rng_state, configure_fp32_determinism
from molgap.v4_runtime import state_dict_sha256
from test_family_training_runner import context, contract_file


TRAIN_ROLE_IDENTITY = "3" * 64
MANIFEST_IDENTITY = "1" * 64
TRAIN_ROWS = 257
DEVELOPMENT_ROWS = 3
PHYSICAL_BATCH = 128


def clean_policy(*, role_identity=TRAIN_ROLE_IDENTITY, row_order=None):
    rows = torch.arange(PHYSICAL_BATCH, dtype=torch.int64) if row_order is None else row_order
    return CalibrationPolicy(
        input_identity=MANIFEST_IDENTITY,
        training_role_identity=role_identity,
        source_order_sha256=source_order_sha256(rows),
        batches=1,
        batch_size=PHYSICAL_BATCH,
    )


def clean_recipe(*, policy=None):
    return FamilyRecipe(
        "neural_atom_k1", "reference", 42, "2" * 64,
        DatasetSelection("synthetic.json", MANIFEST_IDENTITY, True),
        TRAIN_ROWS, DEVELOPMENT_ROWS, 4, .001, .01,
        ScheduleSpec("cosine_epoch", 2), "best-development-ema",
        "python-epoch-shuffle", 1,
        target_mean_eV=3., target_std_eV=2., ema_decay=.999,
        finite_check_every=1, clean_bn=policy if policy is not None else clean_policy(),
    )


def make_data(recipe):
    def graphs(start, rows):
        return [Data(
            x=torch.zeros(1, 9, dtype=torch.long),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_attr=torch.empty(0, 3, dtype=torch.long),
            random_walk_pe=torch.zeros(1, 16),
            source_idx=torch.tensor([index], dtype=torch.long),
            y=torch.tensor([float(index % 7)]),
        ) for index in range(start, start + rows)]

    train = graphs(0, TRAIN_ROWS)
    development = graphs(TRAIN_ROWS, recipe.development_rows) if recipe.dataset.development else None
    return TrainingData(
        train, development, recipe.dataset.manifest_sha256,
        TRAIN_ROWS, recipe.development_rows, 0, TRAIN_ROWS if development is not None else None, 3., 2.,
    )


def make_loaders(recipe):
    return FamilyLoaders(
        recipe, make_data(recipe), ExecutionTuning(training_workers=0, development_workers=0),
    )


def make_model():
    return torch.nn.Sequential(
        torch.nn.Linear(1, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Dropout(.2),
        torch.nn.Linear(4, 1),
    )


class SyntheticCleanKernel(FamilyKernel):
    """Real session/evaluator with a small independent BN model and no factory IO."""

    def __init__(self, recipe):
        from molgap.family_training_state import FamilyTrainingState

        self.recipe, self.device = recipe, torch.device("cpu")
        self.model = make_model()
        self.optimizer = make_family_optimizer(recipe, self.model.parameters())
        self.ema = BoundForeachEMA(self.model, recipe.ema_decay)
        self.weight_view = ReusedWeightView()
        self.training_state = FamilyTrainingState(
            self.model, self.ema, clean_factory=make_model,
        )

    def forward(self, cpu_batch, *, model=None):
        owner = self.model if model is None else model
        values = cpu_batch.source_idx.to(dtype=torch.float32).reshape(-1, 1) / TRAIN_ROWS
        return owner(values).reshape(-1)

    def step(self, cpu_batch, *, check_finite):
        target = (cpu_batch.y.view(-1) - self.recipe.target_mean_eV) / self.recipe.target_std_eV

        def loss_builder():
            prediction = self.forward(cpu_batch)
            absolute = (prediction.detach() - target).abs().sum()
            return functional.l1_loss(prediction, target), (absolute, int(target.numel()))

        loss, (absolute, rows) = optimizer_update(
            self.model, self.optimizer, loss_builder,
            clip_norm=self.recipe.gradient_clip, check_finite=check_finite,
        )
        self.training_state.update_raw_ema()
        return loss, absolute, rows


def make_session(recipe, *, root, contract, ctx):
    configure_fp32_determinism(recipe.seed)
    kernel = SyntheticCleanKernel(recipe)
    session = FamilyTrainingSession(
        kernel, make_loaders(recipe),
        checkpoint=lambda *args: None,
        selected=lambda *args: None,
        pass_finished=lambda *args: None,
    )
    outputs = SessionOutputs(
        root, ctx, contract=contract, adapter=PROFILES[recipe.family],
        trajectory_id="synthetic-clean-trajectory", train_role_identity=TRAIN_ROLE_IDENTITY,
    )
    session.checkpoint, session.selected, session.pass_finished = (
        outputs.checkpoint, outputs.selected, outputs.pass_finished,
    )
    return session, outputs


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            assert_tree_equal(first, second)
    else:
        assert left == right


def test_clean_none_keeps_legacy_recipe_identity_and_record_shape():
    recipe = replace(clean_recipe(), clean_bn=None)
    emitted = recipe.to_dict()
    old_field_names = (
        "family", "mode", "seed", "initialization_sha256", "dataset", "train_rows",
        "development_rows", "optimizer_steps", "learning_rate", "weight_decay", "schedule",
        "selection", "sampler", "checkpoint_every", "target_mean_eV", "target_std_eV",
        "ema_decay", "batch_size", "gradient_clip", "finite_check_every", "drop_last",
        "torch_version",
    )
    serialized = asdict(recipe)
    legacy_shape = {
        "format": RECIPE_FORMAT,
        **{name: serialized[name] for name in old_field_names},
        "execution_policy": recipe.execution_policy,
    }

    assert list(emitted) == ["format", *old_field_names, "execution_policy"]
    assert emitted == legacy_shape
    assert json.dumps(emitted, sort_keys=True, separators=(",", ":")) == json.dumps(
        legacy_shape, sort_keys=True, separators=(",", ":"),
    )
    assert recipe.identity == canonical_fingerprint(legacy_shape)
    restored = FamilyRecipe.from_dict(legacy_shape)
    assert restored == recipe and restored.identity == recipe.identity


def test_clean_selection_and_checkpoint_resume_match_uninterrupted_session(tmp_path):
    recipe = clean_recipe()
    template = SyntheticCleanKernel(recipe)
    template_loaders = make_loaders(recipe)
    template_session = FamilyTrainingSession(
        template, template_loaders, checkpoint=lambda *args: None,
        selected=lambda *args: None, pass_finished=lambda *args: None,
    )
    contract, _ = contract_file(tmp_path, template_session)
    ctx = context(recipe.family, contract)
    template_session.close()

    baseline, baseline_outputs = make_session(
        recipe, root=tmp_path / "baseline", contract=contract, ctx=ctx,
    )
    baseline_progress = baseline.run()
    baseline_state = deepcopy(baseline.state_dict())
    baseline_selected = torch.load(
        baseline_outputs.root / "selected_model.pt", map_location="cpu", weights_only=True,
    )

    paused, paused_outputs = make_session(
        recipe, root=tmp_path / "resumed", contract=contract, ctx=ctx,
    )
    pause_calls = 0

    def pause_after_one_batch_of_pass_two():
        nonlocal pause_calls
        pause_calls += 1
        return pause_calls == 3

    paused_progress = paused.run(should_pause=pause_after_one_batch_of_pass_two)
    assert paused_progress.status == "paused" and paused_progress.optimizer_step == 3
    retained = paused_outputs.retained_state()
    assert retained["owner_state"]["raw_revision"] == paused_progress.optimizer_step
    assert retained["owner_state"]["clean"]["stale"] is True

    selected_at_pause = torch.load(
        paused_outputs.root / "selected_model.pt", map_location="cpu", weights_only=True,
    )
    clean_at_pause = retained["owner_state"]["clean"]["model"]
    selected_calibration = selected_at_pause["calibration"]
    assert selected_at_pause["optimizer_step"] == selected_calibration["raw_revision"]
    assert selected_calibration["identity"] == retained["owner_state"]["clean"]["calibration"]["identity"]
    assert selected_at_pause["weights"] == "ema"
    assert selected_at_pause["calibration"]["clean_state_identity"] == state_dict_sha256(
        selected_at_pause["model"],
    )
    assert state_dict_sha256(selected_at_pause["model"]) == state_dict_sha256(clean_at_pause)

    missing_owner = deepcopy(retained)
    missing_owner.pop("owner_state")
    with pytest.raises(ValueError, match="Incomplete family resume state"):
        validate_session_resume(recipe, missing_owner)

    resumed, resumed_outputs = make_session(
        recipe, root=paused_outputs.root, contract=contract, ctx=ctx,
    )
    resumed.load_state_dict(resumed_outputs.retained_state())
    resumed_progress = resumed.run()
    resumed_state = deepcopy(resumed.state_dict())
    resumed_selected = torch.load(
        resumed_outputs.root / "selected_model.pt", map_location="cpu", weights_only=True,
    )

    assert resumed_progress == baseline_progress
    assert_tree_equal(resumed_state, baseline_state)
    assert resumed_selected["optimizer_step"] == baseline_selected["optimizer_step"]
    assert resumed_selected["development_mae_eV"] == baseline_selected["development_mae_eV"]
    assert resumed_selected["calibration"]["identity"] == baseline_selected["calibration"]["identity"]
    for selected in (baseline_selected, resumed_selected):
        assert selected["weights"] == "ema"
        assert selected["optimizer_step"] == selected["calibration"]["raw_revision"]
        assert selected["calibration"]["clean_state_identity"] == state_dict_sha256(selected["model"])


def test_calibration_rejects_wrong_contract_role_and_nonprefix_order(tmp_path):
    recipe = clean_recipe()
    session = FamilyTrainingSession(
        SyntheticCleanKernel(recipe), make_loaders(recipe),
        checkpoint=lambda *args: None, selected=lambda *args: None,
        pass_finished=lambda *args: None,
    )
    contract, record = contract_file(tmp_path, session)
    session.close()

    record["train_role_identity"] = "4" * 64
    contract.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="role identity"):
        read_recipe(contract)

    loaders = make_loaders(recipe)
    wrong_order = clean_policy(row_order=torch.arange(PHYSICAL_BATCH - 1, -1, -1, dtype=torch.int64))
    try:
        with pytest.raises(ValueError, match="exact training prefix"):
            loaders.calibration(wrong_order)
    finally:
        loaders.close()


def test_scratch_native_calibration_preserves_shared_session_state_and_rng():
    from molgap.family_training_state import FamilyTrainingState

    recipe = replace(
        clean_recipe(),
        dataset=DatasetSelection("synthetic.json", MANIFEST_IDENTITY, False),
        development_rows=0,
        optimizer_steps=4,
        selection="final-ema",
        clean_bn=None,
    )
    configure_fp32_determinism(recipe.seed)
    kernel = SyntheticCleanKernel(recipe)
    session = FamilyTrainingSession(
        kernel, make_loaders(recipe), checkpoint=lambda *args: None,
        selected=lambda *args: None, pass_finished=lambda *args: None,
    )
    try:
        assert session.run().status == "complete"
        before = deepcopy(session.state_dict())
        rng_before = capture_rng_state()
        pool = make_loaders(recipe)
        policy = clean_policy()
        scratch = FamilyTrainingState(kernel.model, kernel.ema, clean_factory=make_model)
        try:
            scratch.calibrate(
                policy, pool.calibration(policy),
                forward=lambda model, batch: kernel.forward(batch, model=model).reshape(-1),
                source_indices=lambda batch: batch.source_idx.reshape(-1),
            )
        finally:
            pool.close()
        assert_tree_equal(session.state_dict(), before)
        assert_tree_equal(capture_rng_state(), rng_before)
    finally:
        session.close()
