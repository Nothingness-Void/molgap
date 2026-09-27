"""Lazy joint Gap and atom-reconstruction objective for the K1 screen.

The public recipe/fingerprint functions deliberately have no torch dependency.
Remote training imports the tensor implementation only after the platform has
installed its pinned runtime.  The objective is an adapter around the existing
K1 encoder: it never copies or reimplements the encoder.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


OBJECTIVE_FORMAT = "molgap-k1-joint-objective-config-v1"
CHECKPOINT_FORMAT = "molgap-k1-joint-objective-checkpoint-v1"
MODEL_MODE = "neural_atom_k1_v4"
INFERENCE_MODEL_ID = "neural_atom_k1_v4"
SEED = 42
PHYSICAL_BATCH_SIZE = 128
EPOCHS = 40
STEPS_PER_EPOCH = 31_240 // EPOCHS
OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH
HIDDEN_CHANNELS = 192
CORRUPTION_RATE = 0.01
AUXILIARY_WEIGHT_GAP_ONLY = 0.0
AUXILIARY_WEIGHT_ATOM_AUX = 0.1
CLIP_NORM = 1.0
TARGET_TRANSFORM_ASSET_ID = (
    "a5ebd05f82f481020f2059c77a7bc7b9dee84081fdb9806fdc18d93a6b8d89d7"
)
TARGET_TRANSFORM_ASSET_SHA256 = (
    "1631a6842bcd0ed330eada408cc78b3abe05109a1266466eb2f2ded731857932"
)
TARGET_TRANSFORM_FILE_SHA256 = (
    "20e6730d57080b0bec9901a26a931034aad162848e0075940fd1e1273bf084b3"
)

# These are the official OGB atom feature cardinalities, in column order.
ATOM_FEATURE_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2)
ATOM_FEATURE_COUNT = len(ATOM_FEATURE_DIMS)
SUPPORTED_RECIPES = ("k1_corrupt_gap", "k1_corrupt_gap_atom_aux")

_COUNTER_DOMAIN = "molgap-k1-atom-categorical-corruption-v1"
_MASK = (1 << 63) - 1


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recipe_id(recipe: object) -> str:
    if isinstance(recipe, str):
        value = recipe
    elif isinstance(recipe, Mapping):
        value = recipe.get("recipe_id", recipe.get("recipe"))
    else:
        value = getattr(recipe, "recipe_id", None)
    if not isinstance(value, str) or value not in SUPPORTED_RECIPES:
        raise ValueError(
            f"objective recipe must be one of {SUPPORTED_RECIPES}, got {value!r}"
        )
    return value


def _assert_exact_recipe_overrides(recipe: object, config: Mapping[str, Any]) -> None:
    """Reject caller overrides that would silently change the scientific recipe."""
    if not isinstance(recipe, Mapping):
        return
    expected = {
        "recipe_id": config["recipe_id"],
        "model_mode": MODEL_MODE,
        "seed": SEED,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer_steps": OPTIMIZER_STEPS,
        "corruption_rate": CORRUPTION_RATE,
        "auxiliary_weight": config["auxiliary_loss"]["weight"],
        "clip_norm": CLIP_NORM,
    }
    for key, value in expected.items():
        if key in recipe and recipe[key] != value:
            raise ValueError(f"objective recipe override changes frozen field: {key}")


def objective_config(recipe: object) -> dict[str, Any]:
    """Return the immutable, recipe-only configuration used by both arms.

    The recipe is intentionally independent of the model mode.  The runner
    binds this config to ``neural_atom_k1_v4`` at execution time and rejects
    any other encoder mode.
    """
    recipe_id = _recipe_id(recipe)
    auxiliary_weight = (
        AUXILIARY_WEIGHT_GAP_ONLY
        if recipe_id == "k1_corrupt_gap"
        else AUXILIARY_WEIGHT_ATOM_AUX
    )
    config: dict[str, Any] = {
        "format": OBJECTIVE_FORMAT,
        "recipe_id": recipe_id,
        "model_mode": MODEL_MODE,
        "inference_model_id": INFERENCE_MODEL_ID,
        "model_initialization": {
            "factory": "molgap.pcqm_k1_variants.make_encoder",
            "mode": MODEL_MODE,
            "seed": SEED,
            "policy": "exact-existing-k1-v4-config-and-initial-state",
        },
        "precision": "fp32",
        "seed": SEED,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "optimizer_steps": OPTIMIZER_STEPS,
        "corruption_rate": CORRUPTION_RATE,
        "corruption": {
            "kind": "atom_categorical",
            "selection": "independent-bernoulli-per-atom-row",
            "probability": CORRUPTION_RATE,
            "replacement": "uniform-other-valid-category-per-field",
            "fields": "all-nine-ogb-atom-features",
            "feature_count": ATOM_FEATURE_COUNT,
            "category_dims": list(ATOM_FEATURE_DIMS),
            "edge_index_unchanged": True,
            "edge_attr_unchanged": True,
            "random_walk_pe_unchanged": True,
            "gap_target_unchanged": True,
        },
        "rng": {
            "generator": "torch.Generator",
            "device": "cpu",
            "seed": SEED,
            "scope": "epoch-step-counter",
            "derivation": "sha256(domain:seed:epoch:step) low-63-bits",
            "global_rng_draws": False,
            "same_counter_replays_identically": True,
        },
        "gap_loss": {
            "formula": "L1(predicted_normalized_gap, original_normalized_gap)",
            "target": "gap",
            "unit": "eV",
            "normalization": "immutable-target-transform-asset-mean-and-sample-std",
            "present_every_optimizer_step": True,
        },
        "target_transform_asset": {
            "asset_id": TARGET_TRANSFORM_ASSET_ID,
            "asset_sha256": TARGET_TRANSFORM_ASSET_SHA256,
            "file_sha256": TARGET_TRANSFORM_FILE_SHA256,
            "statistics": "asset-mean-and-sample-std-used-exactly",
            "computed_train_statistics_must_equal_asset": True,
        },
        "auxiliary_loss": {
            "formula": "mean(field_CE(original_x[field]) for all nine fields over selected rows)",
            "weight": auxiliary_weight,
            "coefficient_applies_to_total": True,
            "zero_selected_rows": "differentiable-zero",
            "labels": "original-clean-ogb-atom-categories",
        },
        "auxiliary_heads": {
            "count": ATOM_FEATURE_COUNT,
            "type": "linear",
            "input_units": HIDDEN_CHANNELS,
            "output_units": list(ATOM_FEATURE_DIMS),
            "state_source": "final-node-states-stable-final-k1-mixer-hook",
            "encoder_copy": False,
            "training_only": True,
            "initialization": "same-seed-under-torch.random.fork_rng",
            "constructed_for_gap_only_arm": True,
        },
        "gradient_clipping": {
            "backbone_max_norm": CLIP_NORM,
            "auxiliary_heads_max_norm": CLIP_NORM,
            "separate_parameter_groups": True,
        },
        "checkpoint": {
            "inference_export": "model-state-only-original-gap-head",
            "last_checkpoint_includes": [
                "model",
                "objective_heads",
                "optimizer",
                "scheduler",
                "global_rng_state",
                "augmentation_counter_state",
            ],
            "atomic": True,
            "resume_replay_requires_config_fingerprint": True,
        },
    }
    _assert_exact_recipe_overrides(recipe, config)
    return config


def objective_fingerprint(recipe: object) -> str:
    """Return the canonical fingerprint of :func:`objective_config` exactly."""
    config = objective_config(recipe)
    return _canonical_fingerprint(config)


def verify_frozen_train_targets(values, source_idx, transform):
    """Bind labels bitwise, but never replace frozen constants with a CPU reduction."""
    import torch

    values = values.detach().cpu().view(-1).float().contiguous()
    source_idx = source_idx.detach().cpu().view(-1).long()
    if values.numel() != 100_000 or not torch.equal(source_idx, torch.arange(100_000)):
        raise RuntimeError("Frozen target training membership changed")
    digest = hashlib.sha256(values.numpy().tobytes(order="C")).hexdigest()
    if digest != transform["target_sha256"] or not torch.isfinite(values).all().item():
        raise RuntimeError("Frozen training target bytes changed")
    return {
        "train_target_sha256": digest, "train_rows": values.numel(),
        "train_source_indices_verified": True,
        "computed_train_mean_eV": float(values.mean()),
        "computed_train_sample_std_eV": float(values.std()),
        "computed_statistics_usage": "diagnostic_only_not_transform",
        "applied_mean_eV": float(transform["mean"]),
        "applied_sample_std_eV": float(transform["std"]),
        "transform_source": "immutable_asset_exact_values",
    }


@dataclass(frozen=True)
class CorruptionResult:
    """A single deterministic corruption draw for one batched node table."""

    original_x: Any
    corrupted_x: Any
    selected_mask: Any
    selected_count: int
    epoch: int
    step: int
    generator_seed: int


def _counter_seed(seed: int, epoch: int, step: int) -> int:
    if seed < 0 or epoch < 0 or step < 0:
        raise ValueError("augmentation seed counters must be nonnegative")
    payload = f"{_COUNTER_DOMAIN}:{int(seed)}:{int(epoch)}:{int(step)}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & _MASK


def augmentation_seed(seed: int, epoch: int, step: int) -> int:
    """Expose the stable counter-to-CPU-generator seed derivation."""
    return _counter_seed(int(seed), int(epoch), int(step))


def validate_atom_feature_bounds(x: Any) -> dict[str, Any]:
    """Validate the official nine-column integer category bounds."""
    import torch

    if not torch.is_tensor(x):
        raise TypeError("atom features must be a torch.Tensor")
    if x.ndim != 2 or int(x.shape[1]) != ATOM_FEATURE_COUNT:
        raise ValueError(
            f"atom features must have shape [rows, {ATOM_FEATURE_COUNT}]"
        )
    if x.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("atom features must use an integer dtype")
    minimums: list[int | None] = []
    maximums: list[int | None] = []
    for field, categories in enumerate(ATOM_FEATURE_DIMS):
        values = x[:, field]
        if values.numel() == 0:
            minimums.append(None)
            maximums.append(None)
            continue
        minimum = int(values.min().item())
        maximum = int(values.max().item())
        if minimum < 0 or maximum >= categories:
            raise ValueError(
                f"atom feature column {field} is outside [0, {categories})"
            )
        minimums.append(minimum)
        maximums.append(maximum)
    return {
        "rows": int(x.shape[0]),
        "fields": ATOM_FEATURE_COUNT,
        "category_dims": list(ATOM_FEATURE_DIMS),
        "minimums": minimums,
        "maximums": maximums,
    }


def corrupt_atom_features(
    x: Any,
    *,
    epoch: int,
    step: int,
    seed: int = SEED,
    rate: float = CORRUPTION_RATE,
) -> CorruptionResult:
    """Corrupt atom rows with a private CPU generator and no global RNG draws."""
    import torch

    validate_atom_feature_bounds(x)
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError("atom corruption probability must be in [0, 1]")
    generator_seed = _counter_seed(int(seed), int(epoch), int(step))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(generator_seed)
    original_cpu = x.detach().to(device="cpu", dtype=torch.long)
    corrupted_cpu = original_cpu.clone()
    selected_cpu = torch.rand(
        (int(original_cpu.shape[0]),), generator=generator, device="cpu"
    ) < float(rate)
    selected_indices = torch.nonzero(selected_cpu, as_tuple=False).view(-1)
    for field, categories in enumerate(ATOM_FEATURE_DIMS):
        if selected_indices.numel() == 0:
            continue
        original = original_cpu[selected_indices, field]
        draw = torch.randint(
            categories - 1,
            (int(selected_indices.numel()),),
            generator=generator,
            device="cpu",
        )
        replacement = draw + (draw >= original).to(dtype=torch.long)
        corrupted_cpu[selected_indices, field] = replacement
    return CorruptionResult(
        original_x=x.detach().clone(),
        corrupted_x=corrupted_cpu.to(device=x.device, dtype=x.dtype),
        selected_mask=selected_cpu.to(device=x.device),
        selected_count=int(selected_indices.numel()),
        epoch=int(epoch),
        step=int(step),
        generator_seed=generator_seed,
    )


def mean_atom_reconstruction_ce(
    logits: Sequence[Any],
    original_x: Any,
    selected_mask: Any,
) -> Any:
    """Return the mean of the nine original-feature CEs on selected rows."""
    import torch
    import torch.nn.functional as functional

    if len(logits) != ATOM_FEATURE_COUNT:
        raise ValueError(f"expected {ATOM_FEATURE_COUNT} atom reconstruction heads")
    validate_atom_feature_bounds(original_x)
    if selected_mask.ndim != 1 or int(selected_mask.shape[0]) != int(original_x.shape[0]):
        raise ValueError("selected atom mask must align with atom feature rows")
    if selected_mask.dtype != torch.bool:
        selected_mask = selected_mask.to(dtype=torch.bool)
    selected = selected_mask.to(device=logits[0].device)
    if int(selected.sum().item()) == 0:
        # Keep a graph to every head so zero selection is a differentiable zero.
        zero = logits[0].sum() * 0.0
        for output in logits[1:]:
            zero = zero + output.sum() * 0.0
        return zero
    losses = []
    labels = original_x.to(device=logits[0].device)
    for field, (output, categories) in enumerate(zip(logits, ATOM_FEATURE_DIMS)):
        if output.ndim != 2 or int(output.shape[0]) != int(labels.shape[0]):
            raise ValueError("atom reconstruction logits do not align with nodes")
        if int(output.shape[1]) != categories:
            raise ValueError(
                f"atom reconstruction head {field} has wrong category count"
            )
        losses.append(
            functional.cross_entropy(output[selected], labels[selected, field].long())
        )
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class JointLossResult:
    gap_loss: Any
    auxiliary_loss: Any
    total_loss: Any
    corruption: CorruptionResult


@dataclass
class AugmentationCounter:
    """Checkpointable epoch/step counter for deterministic corruption draws."""

    seed: int = SEED
    epochs: int = EPOCHS
    steps_per_epoch: int = STEPS_PER_EPOCH
    next_epoch: int = 0
    next_step: int = 0
    optimizer_steps: int = 0
    last_generator_seed: int | None = None

    def seed_for(self, epoch: int, step: int) -> int:
        if not 0 <= int(epoch) < self.epochs:
            raise ValueError(f"augmentation epoch outside frozen range: {epoch}")
        if not 0 <= int(step) < self.steps_per_epoch:
            raise ValueError(f"augmentation step outside frozen range: {step}")
        return _counter_seed(self.seed, int(epoch), int(step))

    def advance(self, epoch: int, step: int) -> None:
        if (int(epoch), int(step)) != (self.next_epoch, self.next_step):
            raise RuntimeError(
                "augmentation counter advanced out of order: "
                f"expected {(self.next_epoch, self.next_step)}, got {(epoch, step)}"
            )
        self.last_generator_seed = self.seed_for(epoch, step)
        self.optimizer_steps += 1
        if self.next_step + 1 == self.steps_per_epoch:
            self.next_epoch += 1
            self.next_step = 0
        else:
            self.next_step += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "molgap-k1-augmentation-counter-v1",
            "seed": self.seed,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "next_epoch": self.next_epoch,
            "next_step": self.next_step,
            "optimizer_steps": self.optimizer_steps,
            "last_generator_seed": self.last_generator_seed,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "format": "molgap-k1-augmentation-counter-v1",
            "seed": self.seed,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"augmentation counter identity mismatch: {key}")
        values = {
            "next_epoch": int(state.get("next_epoch", -1)),
            "next_step": int(state.get("next_step", -1)),
            "optimizer_steps": int(state.get("optimizer_steps", -1)),
        }
        if not 0 <= values["next_epoch"] <= self.epochs:
            raise RuntimeError("augmentation counter epoch is outside frozen range")
        if not 0 <= values["next_step"] < self.steps_per_epoch:
            if not (
                values["next_epoch"] == self.epochs
                and values["next_step"] == 0
            ):
                raise RuntimeError("augmentation counter step is outside frozen range")
        if values["optimizer_steps"] != (
            values["next_epoch"] * self.steps_per_epoch + values["next_step"]
        ):
            raise RuntimeError("augmentation counter optimizer-step count changed")
        self.next_epoch = values["next_epoch"]
        self.next_step = values["next_step"]
        self.optimizer_steps = values["optimizer_steps"]
        last = state.get("last_generator_seed")
        self.last_generator_seed = None if last is None else int(last)


def _state_sha256(module: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not torch.is_tensor(value):
            continue
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _find_final_node_state_module(model: Any) -> Any:
    """Resolve the existing layer-9 K1 mixer used as the stable node hook."""
    base = getattr(model, "base", model)
    mixers = getattr(base, "neural_atom_mixers", None)
    if mixers is None or "9" not in mixers:
        raise ValueError(
            "joint K1 objective requires the existing neural_atom_k1_v4 layer-9 mixer"
        )
    return mixers["9"]


def make_joint_objective(model: Any, recipe: object) -> Any:
    """Attach the lazy objective to an already-created exact K1 model.

    The class is defined inside this factory so importing this module does not
    import torch or PyG before a remote runtime has completed its pinning.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    config = objective_config(recipe)
    if config["model_mode"] != MODEL_MODE:
        raise ValueError("joint objective config has an unexpected model mode")
    model_device = next(model.parameters()).device

    def rng_snapshot() -> tuple[Any, tuple[Any, ...]]:
        """Capture every RNG stream that a pinned worker could expose."""
        cpu = torch.random.get_rng_state().clone()
        cuda = (
            tuple(state.clone() for state in torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )
        return cpu, cuda

    rng_before_head_init = rng_snapshot()

    class AtomReconstructionHeads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = nn.ModuleList(
                nn.Linear(HIDDEN_CHANNELS, categories)
                for categories in ATOM_FEATURE_DIMS
            )

        def forward(self, hidden: Any) -> tuple[Any, ...]:
            if hidden.ndim != 2 or int(hidden.shape[1]) != HIDDEN_CHANNELS:
                raise ValueError("final K1 node states must have width 192")
            return tuple(head(hidden) for head in self.heads)

    class JointGapAtomObjective(nn.Module):
        """Training-only heads and corruption bookkeeping around the K1 model."""

        def __init__(self) -> None:
            super().__init__()
            # Construct on CPU under a forked RNG, then move without consuming
            # the encoder/dropout RNG stream on either arm or accelerator.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(SEED)
                heads = AtomReconstructionHeads()
            self.objective_heads = heads.to(model_device)
            rng_after_head_init = rng_snapshot()
            self.head_initialization_rng_unchanged = bool(
                torch.equal(rng_before_head_init[0], rng_after_head_init[0])
                and len(rng_before_head_init[1]) == len(rng_after_head_init[1])
                and all(
                    torch.equal(before, after)
                    for before, after in zip(
                        rng_before_head_init[1], rng_after_head_init[1]
                    )
                )
            )
            if not self.head_initialization_rng_unchanged:
                raise RuntimeError(
                    "auxiliary-head initialization changed CPU or CUDA RNG state"
                )
            self.counter = AugmentationCounter()
            self.config = config
            self.fingerprint = _canonical_fingerprint(config)
            self.head_initial_state_sha256 = _state_sha256(self.objective_heads)
            self._node_states: Any = None
            self._hook_handle = None

        def attach(self) -> "JointGapAtomObjective":
            if self._hook_handle is not None:
                return self

            def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                if not torch.is_tensor(output) or output.ndim != 2:
                    raise RuntimeError("K1 layer-9 hook did not expose node states")
                self._node_states = output

            self._hook_handle = _find_final_node_state_module(model).register_forward_hook(
                capture
            )
            return self

        def close(self) -> None:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None
            self._node_states = None

        def clear_node_states(self) -> None:
            self._node_states = None

        def prepare_batch(self, batch: Any, *, epoch: int, step: int) -> tuple[Any, CorruptionResult]:
            corruption = corrupt_atom_features(
                batch.x,
                epoch=epoch,
                step=step,
                seed=self.counter.seed,
                rate=self.config["corruption_rate"],
            )
            corrupted_batch = batch.clone()
            corrupted_batch.x = corruption.corrupted_x
            return corrupted_batch, corruption

        def loss_from_prediction(
            self,
            prediction: Any,
            batch: Any,
            corruption: CorruptionResult,
            mean: Any,
            std: Any,
        ) -> JointLossResult:
            if self._node_states is None:
                raise RuntimeError("joint objective did not receive final K1 node states")
            target = (batch.y.view(-1) - mean) / std
            gap_loss = functional.l1_loss(prediction.view(-1), target)
            logits = self.objective_heads(self._node_states)
            auxiliary_loss = mean_atom_reconstruction_ce(
                logits, corruption.original_x, corruption.selected_mask
            )
            total_loss = gap_loss + float(
                self.config["auxiliary_loss"]["weight"]
            ) * auxiliary_loss
            return JointLossResult(
                gap_loss=gap_loss,
                auxiliary_loss=auxiliary_loss,
                total_loss=total_loss,
                corruption=corruption,
            )

        def checkpoint_state(self) -> dict[str, Any]:
            return {
                "format": CHECKPOINT_FORMAT,
                "objective_config": self.config,
                "objective_fingerprint": self.fingerprint,
                "augmentation_counter_state": self.counter.state_dict(),
                "head_initial_state_sha256": self.head_initial_state_sha256,
                "head_initialization_rng_unchanged": self.head_initialization_rng_unchanged,
            }

        def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
            if state.get("format") != CHECKPOINT_FORMAT:
                raise RuntimeError("joint objective checkpoint format changed")
            if state.get("objective_fingerprint") != self.fingerprint:
                raise RuntimeError("joint objective fingerprint changed on resume")
            if state.get("objective_config") != self.config:
                raise RuntimeError("joint objective config changed on resume")
            if state.get("head_initial_state_sha256") != self.head_initial_state_sha256:
                raise RuntimeError("auxiliary head initialization changed on resume")
            if state.get("head_initialization_rng_unchanged") is not True:
                raise RuntimeError("auxiliary head RNG-isolation evidence is missing")
            self.counter.load_state_dict(state["augmentation_counter_state"])

    return JointGapAtomObjective().attach()


def objective_parameter_groups(model: Any, objective: Any) -> tuple[list[Any], list[Any]]:
    """Return unchanged-backbone and training-only-head parameter lists."""
    return list(model.parameters()), list(objective.objective_heads.parameters())


def clip_joint_gradients(
    model: Any,
    objective: Any,
    *,
    max_norm: float = CLIP_NORM,
) -> tuple[Any, Any]:
    """Clip backbone and auxiliary heads independently at the same bound."""
    import torch

    backbone_parameters, head_parameters = objective_parameter_groups(model, objective)
    backbone_norm = torch.nn.utils.clip_grad_norm_(backbone_parameters, max_norm)
    head_norm = torch.nn.utils.clip_grad_norm_(head_parameters, max_norm)
    if not bool(torch.isfinite(backbone_norm)) or not bool(torch.isfinite(head_norm)):
        raise RuntimeError("joint objective gradient norm is non-finite")
    return backbone_norm, head_norm


def _finite_optimizer_state(optimizer: Any) -> bool:
    import torch

    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                return False
    return True


def run_objective_preflight(
    model: Any,
    objective: Any,
    optimizer: Any,
    forward: Callable[[Any], Any],
    batch: Any,
    mean: Any,
    std: Any,
    *,
    initial_encoder_state_sha256: str,
    expected_encoder_state_sha256: str | None = None,
) -> dict[str, Any]:
    """Run objective checks on the already-created model, restoring all state."""
    import copy
    import torch

    from .training_reproducibility import capture_rng_state, restore_rng_state

    bounds = validate_atom_feature_bounds(batch.x)
    if int(getattr(batch, "num_graphs", 0)) != PHYSICAL_BATCH_SIZE:
        raise RuntimeError("joint objective preflight requires physical batch 128")
    original_x = batch.x.detach().clone()
    before_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    before_heads = copy.deepcopy(objective.objective_heads.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_rng = capture_rng_state()
    model_training = model.training
    objective_training = objective.training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result: dict[str, Any] = {}
    try:
        first = corrupt_atom_features(batch.x, epoch=0, step=0)
        second = corrupt_atom_features(batch.x, epoch=0, step=0)
        replayable = bool(
            torch.equal(first.corrupted_x, second.corrupted_x)
            and torch.equal(first.selected_mask, second.selected_mask)
            and first.generator_seed == second.generator_seed
        )
        selected_other = True
        if first.selected_count:
            for field in range(ATOM_FEATURE_COUNT):
                selected_other = selected_other and bool(
                    torch.all(
                        first.corrupted_x[first.selected_mask, field]
                        != batch.x[first.selected_mask, field]
                    )
                )
        corrupted_batch = batch.clone()
        corrupted_batch.x = first.corrupted_x
        model.train()
        objective.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(corrupted_batch)
        losses = objective.loss_from_prediction(
            prediction, batch, first, mean, std
        )
        finite_losses = all(
            bool(torch.isfinite(value).item())
            for value in (losses.gap_loss, losses.auxiliary_loss, losses.total_loss)
        )
        losses.total_loss.backward()
        auxiliary_head_gradient_nonzero = any(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and float(parameter.grad.abs().sum()) > 0.0
            for parameter in objective.objective_heads.parameters()
        )
        backbone_norm, head_norm = clip_joint_gradients(model, objective)
        gradients_finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all())
            for parameter in (
                list(model.parameters()) + list(objective.objective_heads.parameters())
            )
        )
        optimizer.step()
        parameters_finite = all(
            bool(torch.isfinite(parameter).all())
            for parameter in (
                list(model.parameters()) + list(objective.objective_heads.parameters())
            )
        )
        optimizer_finite = _finite_optimizer_state(optimizer)
        objective.clear_node_states()
        model.eval()
        objective.eval()
        with torch.no_grad():
            clean_prediction = forward(batch)
        clean_eval = bool(torch.isfinite(clean_prediction).all())
        clean_x_unchanged = bool(torch.equal(batch.x, original_x))
        resume_state = objective.checkpoint_state()
        replay_counter = AugmentationCounter()
        replay_counter.load_state_dict(resume_state["augmentation_counter_state"])
        resume_replayable = replay_counter.seed_for(0, 0) == augmentation_seed(SEED, 0, 0)
        same_encoder_initialization = (
            expected_encoder_state_sha256 is None
            or initial_encoder_state_sha256 == expected_encoder_state_sha256
        )
        auxiliary_head_gradient_required = bool(
            objective.config["auxiliary_loss"]["weight"] > 0
            and first.selected_count > 0
        )
        auxiliary_head_gradient_ok = bool(
            not auxiliary_head_gradient_required
            or (
                auxiliary_head_gradient_nonzero
                and float(head_norm.detach().cpu()) > 0.0
            )
        )
        result = {
            "format": "molgap-k1-joint-objective-preflight-v1",
            "accepted": bool(
                replayable
                and selected_other
                and finite_losses
                and gradients_finite
                and parameters_finite
                and optimizer_finite
                and clean_eval
                and clean_x_unchanged
                and resume_replayable
                and bool(getattr(objective, "head_initialization_rng_unchanged", False))
                and same_encoder_initialization
                and auxiliary_head_gradient_ok
            ),
            "objective_fingerprint": objective.fingerprint,
            "objective_config": objective.config,
            "feature_bounds": bounds,
            "corruption_replayable": replayable,
            "selected_rows_are_other_categories": selected_other,
            "preflight_corrupted_atom_rows": first.selected_count,
            "finite_component_losses": finite_losses,
            "finite_gradients": gradients_finite,
            "backbone_clip_norm": float(backbone_norm.detach().cpu()),
            "auxiliary_head_clip_norm": float(head_norm.detach().cpu()),
            "auxiliary_head_gradient_nonzero": auxiliary_head_gradient_nonzero,
            "auxiliary_head_gradient_required": auxiliary_head_gradient_required,
            "auxiliary_head_gradient_ok": auxiliary_head_gradient_ok,
            "finite_optimizer": bool(parameters_finite and optimizer_finite),
            "clean_eval_finite": clean_eval,
            "clean_eval_no_corruption": clean_x_unchanged,
            "resume_counter_replayable": resume_replayable,
            "head_initialization_rng_unchanged": bool(
                getattr(objective, "head_initialization_rng_unchanged", False)
            ),
            "initial_encoder_state_sha256": initial_encoder_state_sha256,
            "same_encoder_initialization": same_encoder_initialization,
            "same_encoder_initialization_checked_against": expected_encoder_state_sha256,
            "physical_batch_size": PHYSICAL_BATCH_SIZE,
            "batch_atom_rows": int(batch.x.shape[0]),
            "auxiliary_logits_elements": int(
                batch.x.shape[0] * sum(ATOM_FEATURE_DIMS)
            ),
            "inference_model_id": INFERENCE_MODEL_ID,
            "inference_export_is_model_only": True,
        }
        if torch.cuda.is_available():
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    finally:
        model.load_state_dict(before_model, strict=True)
        objective.objective_heads.load_state_dict(before_heads, strict=True)
        optimizer.load_state_dict(before_optimizer)
        objective.clear_node_states()
        model.train(model_training)
        objective.train(objective_training)
        restore_rng_state(before_rng)
    if not result.get("accepted", False):
        raise RuntimeError(f"joint objective preflight failed: {result}")
    return result


__all__ = [
    "ATOM_FEATURE_DIMS",
    "CHECKPOINT_FORMAT",
    "MODEL_MODE",
    "SUPPORTED_RECIPES",
    "TARGET_TRANSFORM_ASSET_ID",
    "TARGET_TRANSFORM_ASSET_SHA256",
    "TARGET_TRANSFORM_FILE_SHA256",
    "AugmentationCounter",
    "CorruptionResult",
    "augmentation_seed",
    "clip_joint_gradients",
    "corrupt_atom_features",
    "make_joint_objective",
    "mean_atom_reconstruction_ce",
    "objective_config",
    "objective_fingerprint",
    "objective_parameter_groups",
    "run_objective_preflight",
    "validate_atom_feature_bounds",
]
