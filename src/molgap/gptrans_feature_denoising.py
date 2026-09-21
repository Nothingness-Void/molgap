"""Feature-complete denoising regularization for the frozen GPTrans-T core."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from .gptrans import OGBGPTransTiny
from .gptrans_variants import apply_variant
from .noisy_nodes import make_noisy_nodes_pair_norm_model
from .pcqm_gptrans_v4 import (
    BATCHES_PER_EPOCH,
    DEVELOPMENT_ROWS,
    EPOCHS,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MANIFEST_SHA256,
    MIN_LEARNING_RATE,
    PHYSICAL_BATCH,
    SAMPLE_PRESENTATIONS,
    SEED,
    TRAIN_ROWS,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    ExponentialMovingAverage,
    FrozenEpochScheduler,
    _evaluate,
    _load_datasets,
    _target_stats,
    _training_loader,
    validate_fixed_assets,
    validate_source_archive,
)
from .training_reproducibility import (
    assert_finite_state_dict,
    atomic_json,
    atomic_torch_save,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)
from .v4_runtime import make_adamw_compat, model_state_sha256, torch_load_compat


ARMS = ("full_atom", "full_atom_bond")
ATOM_FEATURE_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2)
BOND_FEATURE_DIMS = (5, 6, 2)
BASE_PARAMETERS = 5_246_817
EXPECTED_PARAMETERS = {
    "full_atom": BASE_PARAMETERS + sum((256 + 1) * value for value in ATOM_FEATURE_DIMS),
    "full_atom_bond": BASE_PARAMETERS
    + sum((256 + 1) * value for value in ATOM_FEATURE_DIMS)
    + sum((32 + 1) * value for value in BOND_FEATURE_DIMS),
}
NOISE_STD = 0.15
AUXILIARY_WEIGHT = 0.10
CHECKPOINT_FORMAT = "molgap-gptrans-feature-denoising-checkpoint-v1"
RUN_FORMAT = "molgap-gptrans-feature-denoising-result-v1"
REFERENCE_MODEL_SHA256 = (
    "c841cdee799daa7a874e0f112ce6dea2932fe0f640f434812bac15b83684b092"
)
REFERENCE_DEVELOPMENT_MAE_EV = 0.14724504947662354


def _scientific_fields(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    scope = "all-nine-atom-categories"
    if arm == "full_atom_bond":
        scope += "-plus-all-three-bond-categories"
    return {
        "benchmark_id": "ogb-lsc-pcqm4mv2-gap-internal-100k-v4",
        "model_id": f"gptrans_noisy_pair_norm_{arm}",
        "manifest_sha256": MANIFEST_SHA256,
        "seed": SEED,
        "precision": "fp32",
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "physical_batch_per_device": PHYSICAL_BATCH,
        "device_count": 1,
        "gradient_accumulation_steps": 1,
        "tail_batch_policy": "drop_last",
        "epochs": EPOCHS,
        "optimizer_steps": EPOCHS * BATCHES_PER_EPOCH,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "loss_fingerprint": (
            "normalized-gap-l1-plus-feature-denoising-ce-alpha0.1-" + scope
        ),
        "noise_std": NOISE_STD,
        "auxiliary_weight": AUXILIARY_WEIGHT,
        "pair_update_norm": True,
    }


class GPTransFeatureDenoising(OGBGPTransTiny):
    """GPTrans-T with full categorical node and optional bond denoising."""

    def __init__(self, *, denoise_bonds: bool) -> None:
        super().__init__(
            node_channels=256,
            pair_channels=32,
            num_layers=12,
            num_heads=8,
            shortest_path_cap=20,
            dropout=0.1,
            drop_path=0.1,
            layer_scale=1.0,
            n_targets=1,
        )
        self.noise_std = NOISE_STD
        self.loss_weight = AUXILIARY_WEIGHT
        self.denoise_bonds = bool(denoise_bonds)
        self.atom_heads = nn.ModuleList(
            nn.Linear(self.node_channels, classes) for classes in ATOM_FEATURE_DIMS
        )
        self.bond_heads = nn.ModuleList(
            nn.Linear(self.pair_channels, classes) for classes in BOND_FEATURE_DIMS
        ) if self.denoise_bonds else nn.ModuleList()
        for head in (*self.atom_heads, *self.bond_heads):
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def _dense_inputs_with_noise(self, x, edge_index, edge_attr, batch):
        dense_x, node_mask = to_dense_batch(x.long(), batch)
        batch_size, max_nodes = dense_x.shape[:2]
        embedded_nodes, embedded_mask = to_dense_batch(self.atom_encoder(x.long()), batch)
        if not torch.equal(node_mask, embedded_mask):
            raise RuntimeError("categorical and embedded node masks differ")
        if self.training and self.noise_std > 0.0:
            node_noise = torch.randn_like(embedded_nodes) * self.noise_std
            embedded_nodes = embedded_nodes + node_noise.masked_fill(
                ~node_mask.unsqueeze(-1), 0.0
            )

        edge_batch, edge_src, edge_dst = self._local_edges(
            edge_index, batch, int(x.shape[0])
        )
        adjacency = torch.zeros(
            (batch_size, max_nodes, max_nodes), dtype=torch.bool, device=x.device
        )
        adjacency[edge_batch, edge_src, edge_dst] = True
        degree = adjacency.sum(-1).clamp_max(511)
        embedded_nodes = (
            embedded_nodes
            + self.in_degree_encoder(degree)
            + self.out_degree_encoder(degree)
        )
        token = self.graph_token.expand(batch_size, -1, -1)
        node = self.input_dropout(
            self.node_norm(torch.cat((token, embedded_nodes), dim=1))
        )

        pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        spatial = self._shortest_path(adjacency, pair_mask)
        pair = self.spatial_encoder(spatial).permute(0, 3, 1, 2)
        bond = self.bond_encoder(edge_attr.long())
        if self.training and self.denoise_bonds and self.noise_std > 0.0:
            bond = bond + torch.randn_like(bond) * self.noise_std
        pair[edge_batch, :, edge_src, edge_dst] += bond
        full_pair = pair.new_zeros(
            batch_size, self.pair_channels, max_nodes + 1, max_nodes + 1
        )
        full_pair[:, :, 1:, 1:] = pair
        full_pair[:, :, 0:1, :] += self.virtual_pair
        full_pair[:, :, 1:, 0:1] += self.virtual_pair
        full_node_mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=x.device),
                node_mask,
            ),
            dim=1,
        )
        key_padding_mask = ~full_node_mask[:, None, None, :]
        return (
            node,
            full_pair,
            key_padding_mask,
            node_mask,
            edge_batch,
            edge_src,
            edge_dst,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        random_walk_pe=None,
        return_aux_loss: bool = False,
    ):
        del random_walk_pe
        (
            node,
            pair,
            key_padding_mask,
            node_mask,
            edge_batch,
            edge_src,
            edge_dst,
        ) = self._dense_inputs_with_noise(x, edge_index, edge_attr, batch)
        for block in self.blocks:
            node, pair = block(node, pair, key_padding_mask)
        graph_state = torch.cat((node[:, 0], pair[:, :, 0, 0]), dim=-1)
        prediction = self.readout(graph_state)
        if not return_aux_loss or not self.training or self.loss_weight <= 0.0:
            return prediction

        dense_targets, target_mask = to_dense_batch(x.long(), batch)
        if not torch.equal(node_mask, target_mask):
            raise RuntimeError("node target mask changed")
        atom_state = node[:, 1:][node_mask]
        atom_target = dense_targets[node_mask]
        atom_loss = torch.stack(
            [
                F.cross_entropy(head(atom_state), atom_target[:, column])
                for column, head in enumerate(self.atom_heads)
            ]
        ).mean()
        if not self.denoise_bonds:
            return prediction, atom_loss

        bond_state = pair[edge_batch, :, edge_src + 1, edge_dst + 1]
        bond_loss = torch.stack(
            [
                F.cross_entropy(head(bond_state), edge_attr[:, column].long())
                for column, head in enumerate(self.bond_heads)
            ]
        ).mean()
        return prediction, 0.5 * (atom_loss + bond_loss)


def make_model(arm: str) -> GPTransFeatureDenoising:
    if arm not in ARMS:
        raise ValueError(arm)
    model = GPTransFeatureDenoising(denoise_bonds=arm == "full_atom_bond")
    apply_variant(model, "pair_update_norm")
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != EXPECTED_PARAMETERS[arm]:
        raise RuntimeError(
            f"Parameter count changed for {arm}: {observed} != {EXPECTED_PARAMETERS[arm]}"
        )
    return model


def _shared_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("denoise_head.", "atom_heads.", "bond_heads.")):
            continue
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_preflight(
    *,
    arm: str,
    dataset_root: Path,
    manifest_path: Path,
    source_archive: Path,
    source_archive_sha256: str,
    source_commit: str,
    output: Path,
) -> dict:
    configure_fp32_determinism(SEED)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Feature-denoising preflight requires one visible CUDA device")
    validate_source_archive(source_archive, source_archive_sha256, source_commit)
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    # Fixed PCQM shards are PyG InMemoryDataset ``(data, slices)`` payloads,
    # not lists of Data objects.  Reuse the same unpacking path as training so
    # preflight exercises the real serialized graph format.
    _, packed_shards = _load_datasets(assets.train_paths)
    batch = next(
        iter(
            DataLoader(
                packed_shards[0],
                batch_size=PHYSICAL_BATCH,
                shuffle=False,
                num_workers=0,
            )
        )
    )
    batch = batch.to("cuda")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    reference = make_noisy_nodes_pair_norm_model().to("cuda").eval()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    candidate = make_model(arm).to("cuda").eval()
    reference_shared = _shared_state_sha256(reference)
    candidate_shared = _shared_state_sha256(candidate)
    if reference_shared != candidate_shared:
        raise RuntimeError("Candidate changed the frozen GPTrans shared initialization")
    with torch.no_grad():
        reference_output = reference(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
        candidate_output = candidate(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )
    evaluation_equivalent = bool(torch.equal(reference_output, candidate_output))
    if not evaluation_equivalent:
        raise RuntimeError("Auxiliary heads changed evaluation-mode predictions")

    candidate.train()
    prediction, auxiliary_loss = candidate(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        return_aux_loss=True,
    )
    total = prediction.float().square().mean() + candidate.loss_weight * auxiliary_loss
    total.backward()
    auxiliary_gradients = [head.weight.grad for head in candidate.atom_heads]
    auxiliary_gradients.extend(head.weight.grad for head in candidate.bond_heads)
    finite_backward = bool(torch.isfinite(total)) and all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and float(gradient.abs().sum()) > 0.0
        for gradient in auxiliary_gradients
    )
    if not finite_backward:
        raise RuntimeError("Auxiliary feature-denoising backward preflight failed")

    result = {
        "format": "molgap-gptrans-feature-denoising-preflight-v1",
        "accepted": True,
        "arm": arm,
        "parameters": EXPECTED_PARAMETERS[arm],
        "accelerator": torch.cuda.get_device_name(0),
        "visible_device_count": torch.cuda.device_count(),
        "manifest_sha256": MANIFEST_SHA256,
        "source_archive_sha256": source_archive_sha256,
        "source_commit": source_commit,
        "reference_shared_state_sha256": reference_shared,
        "candidate_shared_state_sha256": candidate_shared,
        "evaluation_output_bitwise_equal": evaluation_equivalent,
        "finite_auxiliary_backward": finite_backward,
        "real_graph_shard_deserialized": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "preflight.json", result)
    return result


def _optimizer_step(model, optimizer, ema, batch, mean, std, *, check_finite):
    optimizer.zero_grad(set_to_none=True)
    prediction, auxiliary_loss = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        return_aux_loss=True,
    )
    target = (batch.y.view(-1).float() - mean) / std
    gap_loss = F.l1_loss(prediction.view(-1), target)
    total_loss = gap_loss + model.loss_weight * auxiliary_loss
    if check_finite and not bool(torch.isfinite(total_loss)):
        raise RuntimeError("Training loss became non-finite")
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True
    )
    if check_finite and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Gradient norm became non-finite")
    optimizer.step()
    ema.update(model)
    return float(gap_loss.detach()), float(auxiliary_loss.detach())


def _save_checkpoint(
    path,
    *,
    arm,
    epoch,
    model,
    optimizer,
    scheduler,
    ema,
    trace,
    best,
    best_epoch,
    target_stats,
    source_archive_sha256,
):
    atomic_torch_save(
        path,
        {
            "format": CHECKPOINT_FORMAT,
            "arm": arm,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "trace": trace,
            "best_development_mae_eV": best,
            "best_epoch": best_epoch,
            "target_stats": target_stats,
            "source_archive_sha256": source_archive_sha256,
            "rng_state": capture_rng_state(),
            "scientific_fields": _scientific_fields(arm),
        },
    )


def run_training(
    *,
    arm: str,
    dataset_root: Path,
    manifest_path: Path,
    source_archive: Path,
    source_archive_sha256: str,
    source_commit: str,
    output: Path,
    platform_id: str,
) -> dict:
    configure_fp32_determinism(SEED)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Feature-denoising training requires one visible CUDA device")
    output.mkdir(parents=True, exist_ok=True)
    completion_path = output / "completion_manifest.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("complete") is True and completion.get("arm") == arm:
            return completion
        raise RuntimeError("Existing completion manifest is incompatible")
    validate_source_archive(source_archive, source_archive_sha256, source_commit)
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    if (
        preflight.get("accepted") is not True
        or preflight.get("arm") != arm
        or preflight.get("source_archive_sha256") != source_archive_sha256
    ):
        raise RuntimeError("Candidate preflight identity changed")

    train_graphs, train_shards = _load_datasets(assets.train_paths)
    development_graphs, _ = _load_datasets(assets.development_paths)
    if len(train_graphs) != TRAIN_ROWS or len(development_graphs) != DEVELOPMENT_ROWS:
        raise RuntimeError("Loaded role count changed")
    mean_value, std_value = _target_stats(train_shards)
    target_stats = {"mean_eV": mean_value, "sample_std_eV": std_value}
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = make_model(arm).to("cuda")
    initial_state_sha256 = model_state_sha256(model)
    optimizer = make_adamw_compat(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=False,
        foreach=False,
    )
    scheduler = FrozenEpochScheduler(optimizer)
    ema = ExponentialMovingAverage(model)

    checkpoint_path = output / "last_checkpoint.pt"
    start_epoch = 0
    trace = []
    best = float("inf")
    best_epoch = -1
    if checkpoint_path.is_file():
        checkpoint = torch_load_compat(
            checkpoint_path, map_location="cuda", weights_only=False
        )
        if (
            checkpoint.get("format") != CHECKPOINT_FORMAT
            or checkpoint.get("arm") != arm
            or checkpoint.get("scientific_fields") != _scientific_fields(arm)
            or checkpoint.get("source_archive_sha256") != source_archive_sha256
        ):
            raise RuntimeError("Checkpoint identity changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ema.load_state_dict(checkpoint["ema"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        trace = list(checkpoint["trace"])
        best = float(checkpoint["best_development_mae_eV"])
        best_epoch = int(checkpoint["best_epoch"])

    for epoch in range(start_epoch, EPOCHS):
        learning_rate = scheduler.step(epoch)
        model.train()
        gap_losses = []
        auxiliary_losses = []
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(_training_loader(train_graphs, epoch)):
            if int(batch.num_graphs) != PHYSICAL_BATCH:
                raise RuntimeError("Optimizer received a non-128 batch")
            batch = batch.to("cuda", non_blocking=True)
            gap_loss, auxiliary_loss = _optimizer_step(
                model,
                optimizer,
                ema,
                batch,
                mean,
                std,
                check_finite=(batch_index + 1) % 50 == 0,
            )
            gap_losses.append(gap_loss)
            auxiliary_losses.append(auxiliary_loss)
        development = _evaluate(model, ema, development_graphs, mean, std)
        improved = float(development["mae_eV"]) < best
        if improved:
            best = float(development["mae_eV"])
            best_epoch = epoch
            best_payload = {
                "format": RUN_FORMAT,
                "arm": arm,
                "model": {
                    name: value.detach().cpu() for name, value in ema.state_dict().items()
                },
                "target_stats": target_stats,
                "epoch": epoch,
                "development_mae_eV": best,
                "parameters": EXPECTED_PARAMETERS[arm],
                "initial_state_sha256": initial_state_sha256,
                "manifest_sha256": MANIFEST_SHA256,
                "source_archive_sha256": source_archive_sha256,
                "source_commit": source_commit,
            }
            assert_finite_state_dict(best_payload["model"], label=f"best {arm} model")
            atomic_torch_save(output / "best_model.pt", best_payload)
            atomic_torch_save(
                output / "development_predictions.pt",
                {
                    "prediction_eV": development["prediction_eV"],
                    "target_eV": development["target_eV"],
                    "source_idx": development["source_idx"],
                    "official_validation_role_read": False,
                    "test_dev_role_read": False,
                    "test_challenge_role_read": False,
                },
            )
        row = {
            "epoch": epoch,
            "train_normalized_gap_mae": float(np.mean(gap_losses)),
            "train_auxiliary_ce": float(np.mean(auxiliary_losses)),
            "development_mae_eV": float(development["mae_eV"]),
            "best_development_mae_eV": best,
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "optimizer_step": (epoch + 1) * BATCHES_PER_EPOCH,
            "sample_presentations": (epoch + 1) * BATCHES_PER_EPOCH * PHYSICAL_BATCH,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        trace.append(row)
        atomic_json(output / "trace.json", {"format": RUN_FORMAT, "rows": trace})
        _save_checkpoint(
            checkpoint_path,
            arm=arm,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            trace=trace,
            best=best,
            best_epoch=best_epoch,
            target_stats=target_stats,
            source_archive_sha256=source_archive_sha256,
        )
        print(
            f"{arm} ep{epoch:02d} gap={row['train_normalized_gap_mae']:.6f} "
            f"aux={row['train_auxiliary_ce']:.5f} dev={row['development_mae_eV']:.6f}eV "
            f"best={best:.6f}@{best_epoch} {row['elapsed_seconds']:.1f}s"
            + (" *" if improved else ""),
            flush=True,
        )

    artifacts = {
        name: sha256_file(output / name)
        for name in (
            "preflight.json",
            "best_model.pt",
            "development_predictions.pt",
            "last_checkpoint.pt",
            "trace.json",
        )
    }
    completion = {
        "format": RUN_FORMAT,
        "complete": True,
        "arm": arm,
        "parameters": EXPECTED_PARAMETERS[arm],
        "epochs": EPOCHS,
        "optimizer_steps": EPOCHS * BATCHES_PER_EPOCH,
        "sample_presentations": SAMPLE_PRESENTATIONS,
        "best_development_mae_eV": best,
        "best_epoch": best_epoch,
        "manifest_sha256": MANIFEST_SHA256,
        "source_archive_sha256": source_archive_sha256,
        "source_commit": source_commit,
        "initial_state_sha256": initial_state_sha256,
        "artifact_sha256": artifacts,
        "platform_id": platform_id,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(completion_path, completion)
    return completion


def run_reference_evaluation(
    *,
    dataset_root: Path,
    manifest_path: Path,
    reference_model_path: Path,
    output: Path,
    source_commit: str,
) -> dict:
    """Materialize row-aligned predictions from the accepted strict comparator."""
    configure_fp32_determinism(SEED)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Reference evaluation requires one visible CUDA device")
    if sha256_file(reference_model_path) != REFERENCE_MODEL_SHA256:
        raise RuntimeError("Accepted reference checkpoint hash changed")
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    train_graphs, train_shards = _load_datasets(assets.train_paths)
    development_graphs, _ = _load_datasets(assets.development_paths)
    if len(train_graphs) != TRAIN_ROWS or len(development_graphs) != DEVELOPMENT_ROWS:
        raise RuntimeError("Reference role count changed")
    mean_value, std_value = _target_stats(train_shards)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")

    model = make_noisy_nodes_pair_norm_model().to("cuda")
    accepted_state = torch_load_compat(
        reference_model_path, map_location="cuda", weights_only=False
    )
    if not isinstance(accepted_state, dict):
        raise RuntimeError("Accepted reference checkpoint is not a state dictionary")
    model.load_state_dict(accepted_state, strict=True)
    ema = ExponentialMovingAverage(model)
    ema.load_state_dict(accepted_state)
    development = _evaluate(model, ema, development_graphs, mean, std)
    observed = float(development["mae_eV"])
    if abs(observed - REFERENCE_DEVELOPMENT_MAE_EV) > 1e-8:
        raise RuntimeError(
            "Accepted reference development MAE changed: "
            f"{observed} != {REFERENCE_DEVELOPMENT_MAE_EV}"
        )

    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "development_predictions.pt"
    atomic_torch_save(
        predictions_path,
        {
            "prediction_eV": development["prediction_eV"],
            "target_eV": development["target_eV"],
            "source_idx": development["source_idx"],
            "reference_evidence_id": "pcqm-gptrans-noisy-pair-norm-100k-s42",
            "reference_model_sha256": REFERENCE_MODEL_SHA256,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        },
    )
    completion = {
        "format": "molgap-gptrans-feature-denoising-reference-eval-v1",
        "complete": True,
        "reference_evidence_id": "pcqm-gptrans-noisy-pair-norm-100k-s42",
        "reference_model_sha256": REFERENCE_MODEL_SHA256,
        "development_predictions_sha256": sha256_file(predictions_path),
        "development_mae_eV": observed,
        "rows": DEVELOPMENT_ROWS,
        "manifest_sha256": MANIFEST_SHA256,
        "source_commit": source_commit,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "completion_manifest.json", completion)
    return completion


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform-id", default="kaggle1-t4x2")
    parser.add_argument("--reference-model", type=Path)
    args = parser.parse_args()
    if args.reference_model is not None:
        result = run_reference_evaluation(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest_path,
            reference_model_path=args.reference_model,
            output=args.output,
            source_commit=args.source_commit,
        )
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.arm is None:
        parser.error("--arm is required unless --reference-model is supplied")
    run_preflight(
        arm=args.arm,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest_path,
        source_archive=args.source_archive,
        source_archive_sha256=args.source_archive_sha256,
        source_commit=args.source_commit,
        output=args.output,
    )
    result = run_training(
        arm=args.arm,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest_path,
        source_archive=args.source_archive,
        source_archive_sha256=args.source_archive_sha256,
        source_commit=args.source_commit,
        output=args.output,
        platform_id=args.platform_id,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
