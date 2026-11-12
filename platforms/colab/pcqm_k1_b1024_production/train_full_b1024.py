# Source: user Drive MolGap/code production-run script (Desktop copy), stored verbatim apart from these two header comments.
# Known deviation from the frozen recipe: batch 1024 / lr x6 (not batch128/lr4e-4) and TF32 on (recipe: tf32 false).
"""High-Throughput A100 Full-Dataset Training Runner (VRAM-Resident Vectorized Batching + TF32).

PRODUCTION VARIANT for the K1 Muon arm: physical batch 1024 and lr x6 (2.4e-3 on BOTH optimizer groups).
Derived from train_full.py with a deliberately small diff (see the CHANGED markers). Chosen from the 6-epoch /
200K-row study (round 3): it reached the control's final held-out MAE ~8.5x sooner in training seconds, but
60-epoch behaviour is NOT proven. Fallback: --batch-size 512 --lr-mult 4.
"""
from __future__ import annotations

import argparse
import dataclasses  # CHANGED: needed for the batch-size bypass below
import gc
import json
import math  # CHANGED
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from types import MethodType
from molgap.family_training_data import DatasetSelection, PassBatchSampler
from molgap.family_training_kernel import FamilyKernel
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_session import FamilyTrainingSession
from molgap.k1_execution_backend import K1ExecutionConfig
from molgap.training_reproducibility import atomic_json, atomic_torch_save

MANIFEST_SHA = "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
# CHANGED: production constants (CLI: --batch-size / --lr-mult)
BATCH_SIZE = 1024
LR_MULT = 6.0
BASE_LR = 4e-4  # the original recipe lr; the effective lr is BASE_LR * LR_MULT = 2.4e-3 for both optimizer groups
GPTRANS_BATCH_SIZE = 128  # the gptrans_pairnorm arm is NOT validated for large batch and stays at the original 128
TRAIN_ROWS = 3378606
STEPS_PER_EPOCH = TRAIN_ROWS // BATCH_SIZE  # 3299 (drop_last, as in _PassIterator)
TOTAL_EPOCHS = 60
TOTAL_STEPS = STEPS_PER_EPOCH * TOTAL_EPOCHS  # 197940
PRODUCTION_RUN_NAME = "k1-muon-gptrans-full-20261108"  # the old run: never written to
LOG_EVERY = 200  # CHANGED: step-progress line interval (was 500; there are 8x fewer steps per epoch now)
WARN_WORSE_EV = 0.02
WARN_AFTER_EPOCH = 3


def configure_a100_fast_mode(seed: int) -> None:
    """Seed RNGs and unlock full A100 TF32 TensorCores + hardware CUDA atomic scatter."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False)


# CHANGED: pure arithmetic helpers (unit-testable without a GPU)
def steps_per_epoch_for(total_rows: int, batch_size: int) -> int:
    """drop_last semantics, identical to _PassIterator.total_batches / LoopPlan.steps_per_pass."""
    if batch_size < 1 or total_rows < batch_size:
        raise ValueError(f"batch size {batch_size} does not fit {total_rows} rows")
    return total_rows // batch_size


def eta_hours(step: int, total_steps: int, it_s: float) -> float:
    return ((total_steps - step) / it_s / 3600.0) if it_s > 0 else 0.0


def guard_run_root(run_root: Path) -> None:
    if PRODUCTION_RUN_NAME in {p.name for p in [run_root, *run_root.parents]}:
        raise SystemExit(f"Refusing to use {run_root}: it is inside the OLD production run '{PRODUCTION_RUN_NAME}'. Pick a new run directory.")


def load_compare_log(path: Path | None) -> dict[int, float]:
    """epoch (completed passes) -> train_mae_eV from the OLD run's train_log.jsonl. Read-only."""
    out: dict[int, float] = {}
    if path is None:
        return out
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[int(rec["epoch"])] = float(rec["train_mae_eV"])
    return out


def compare_message(epoch: int, new_ev: float, old: dict[int, float]) -> tuple[str | None, str | None]:
    """Returns (info_suffix, warning_line). The warning never aborts training."""
    if epoch not in old:
        return None, None
    old_ev = old[epoch]
    info = f"old_run={old_ev:.4f} eV ({new_ev - old_ev:+.4f})"
    warn = None
    if epoch > WARN_AFTER_EPOCH and new_ev - old_ev > WARN_WORSE_EV:
        warn = (f"WARNING: epoch {epoch}: new train MAE {new_ev:.4f} eV is worse than the old run ({old_ev:.4f} eV) by "
                f"{new_ev - old_ev:.4f} eV (> {WARN_WORSE_EV} eV). Per the plan: stop and use the fallback (--batch-size 512 --lr-mult 4).")
    return info, warn


def check_resume_compat(saved: dict, sidecar: dict | None, recipe_batch: int, recipe_lr: float, lr_mult: float) -> None:
    """Refuse a checkpoint produced with a different batch size or lr. Raises RuntimeError with a clear message."""
    problems = []
    if sidecar is None:
        problems.append("run_config.json is missing next to the checkpoint, so its batch size / lr cannot be confirmed")
    else:
        if int(sidecar.get("batch_size", -1)) != recipe_batch:
            problems.append(f"batch size: checkpoint run used {sidecar.get('batch_size')}, CLI asks for {recipe_batch}")
        if not math.isclose(float(sidecar.get("learning_rate", -1.0)), recipe_lr, rel_tol=1e-9):
            problems.append(f"learning rate: checkpoint run used {sidecar.get('learning_rate')}, CLI asks for {recipe_lr} (lr-mult {lr_mult})")
    step, samples = saved.get("optimizer_step"), saved.get("sample_presentations")
    if isinstance(step, int) and isinstance(samples, int) and samples != step * recipe_batch:
        problems.append(f"sample_presentations {samples} != optimizer_step {step} x batch {recipe_batch} (checkpoint was written with another batch size)")
    base_lrs = ((saved.get("scheduler") or {}).get("native") or {}).get("base_lrs")
    if base_lrs and not math.isclose(float(base_lrs[0]), recipe_lr, rel_tol=1e-9):
        problems.append(f"scheduler base lr {base_lrs[0]} != {recipe_lr}")
    if problems:
        raise RuntimeError("Refusing to resume: " + "; ".join(problems) +
                           ". Use the same --batch-size/--lr-mult as the original run, or a new run directory.")


def make_recipe(arm: str, total_rows: int = TRAIN_ROWS, batch_size: int = BATCH_SIZE, lr_mult: float = LR_MULT) -> tuple[FamilyRecipe, str]:
    # CHANGED: the k1_muon recipe is first built exactly as in the original (valid batch 128), then re-parameterised.
    steps_per_epoch = total_rows // 128
    total_steps = steps_per_epoch * TOTAL_EPOCHS
    dataset = DatasetSelection("ogb-train-full/manifest.json", MANIFEST_SHA, False)
    if arm == "k1_muon":
        base = FamilyRecipe(
            family="neural_atom_k1",
            mode="reference",
            seed=42,
            initialization_sha256="8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd",
            dataset=dataset,
            train_rows=total_rows,
            development_rows=0,
            optimizer_steps=total_steps,
            learning_rate=BASE_LR,
            weight_decay=1e-5,
            schedule=ScheduleSpec("cosine_epoch", TOTAL_EPOCHS, minimum_lr=1e-6, warmup_units=0),
            selection="final-live",
            sampler="torch-pass-randperm",
            checkpoint_every=steps_per_epoch,
            target_mean_eV=5.689458847045898,
            target_std_eV=1.1621397733688354,
            ema_decay=0.999,
            resume_chunks=False,
            optimizer="k1-muon-batched-adamw-fp32-v1",
            k1_execution=K1ExecutionConfig(
                aggregation="native",
                pooling="native",
                projection="node",
                evaluation_memory_fill=False,
            ),
        )
        if batch_size == 128 and lr_mult == 1.0:
            return base, "k1_init.pt"
        spp = steps_per_epoch_for(total_rows, batch_size)
        # FamilyRecipe validation pins batch_size == 128 (family_training_recipe.py: "Live family training retains physical batch 128"),
        # so: replace() on the valid batch-128 recipe, then set the physical batch on the frozen dataclass (exactly what ran in round 3).
        recipe = dataclasses.replace(base, learning_rate=BASE_LR * lr_mult, optimizer_steps=TOTAL_EPOCHS * spp, checkpoint_every=spp)
        if batch_size != 128:
            object.__setattr__(recipe, "batch_size", batch_size)
        return recipe, "k1_init.pt"
    if arm == "gptrans_pairnorm":
        # unchanged, batch 128 (not validated for large batch)
        steps_per_epoch = total_rows // GPTRANS_BATCH_SIZE
        total_steps = steps_per_epoch * TOTAL_EPOCHS
        return (
            FamilyRecipe(
                family="gptrans_t",
                mode="degree_bond_local_pair_update_norm_ema999",
                seed=42,
                initialization_sha256="a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b",
                dataset=dataset,
                train_rows=total_rows,
                development_rows=0,
                optimizer_steps=total_steps,
                learning_rate=1e-3,
                weight_decay=0.05,
                schedule=ScheduleSpec("warmup_cosine_epoch", TOTAL_EPOCHS, minimum_lr=1e-6, warmup_units=4),
                selection="final-ema",
                sampler="torch-pass-randperm",
                checkpoint_every=steps_per_epoch,
                target_mean_eV=5.689459009120812,
                target_std_eV=1.1621397733722936,
                ema_decay=0.999,
                resume_chunks=False,
                optimizer="adamw",
            ),
            "gptrans_init.pt",
        )
    raise ValueError(f"Unknown arm: {arm}")


class GPUBatch:
    """Lightweight zero-copy GPU batch container compatible with both K1 and GPTrans."""
    __slots__ = (
        "x",
        "edge_index",
        "edge_attr",
        "batch",
        "random_walk_pe",
        "y",
        "num_graphs",
        "_gpu_prepared",
        "_gpu_dense_layout",
        "execution_layout",
    )

    def __init__(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        random_walk_pe: torch.Tensor | None,
        y: torch.Tensor,
        num_graphs: int,
        gpu_dense_layout: dict | None = None,
        execution_layout: dict | None = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch
        self.random_walk_pe = random_walk_pe
        self.y = y
        self.num_graphs = num_graphs
        self._gpu_prepared = True
        if gpu_dense_layout is not None:
            self._gpu_dense_layout = gpu_dense_layout
        if execution_layout is not None:
            self.execution_layout = execution_layout

    def to(self, *args, **kwargs):
        return self


class _DataStub:
    def __init__(self, train_rows: int, target_mean_eV: float, target_std_eV: float):
        self.train = None
        self.development = None
        self.manifest_sha256 = MANIFEST_SHA
        self.train_rows = train_rows
        self.development_rows = 0
        self.train_source_start = 0
        self.development_source_start = None
        self.target_mean_eV = target_mean_eV
        self.target_std_eV = target_std_eV


class VRAMGraphStore:
    """Stores all 3.38M graphs contiguously in A100 40GB VRAM (~4.3 GB) for <0.1ms zero-sync batching."""

    def __init__(self, data_root: Path, *, need_rwse: bool, device: str = "cuda"):
        t0 = time.perf_counter()
        self.device = torch.device(device)
        self.need_rwse = need_rwse
        manifest_path = data_root / "ogb-train-full/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [r for r in manifest["assets"]["topology"] if r.get("role") == "train"]

        # Pass 1: fast header scan via mmap (~0.08s) to pre-allocate exact contiguous GPU buffers
        shard_paths = [manifest_path.parent / r["file"] for r in records]
        total_graphs = 0
        total_nodes = 0
        total_edges = 0
        for sp in shard_paths:
            _, slices = torch.load(sp, map_location="cpu", weights_only=False, mmap=True)
            total_graphs += int(slices["x"].numel() - 1)
            total_nodes += int(slices["x"][-1].item())
            total_edges += int(slices["edge_index"][-1].item())

        self.num_graphs = total_graphs
        self.num_nodes = total_nodes
        self.num_edges = total_edges

        # Pre-allocate compact contiguous tensors directly on GPU (~4.3 GB on A100 40GB)
        self.all_x = torch.empty((total_nodes, 9), dtype=torch.uint8, device=self.device)
        self.all_edge_index = torch.empty((2, total_edges), dtype=torch.int16, device=self.device)
        self.all_edge_attr = torch.empty((total_edges, 3), dtype=torch.uint8, device=self.device)
        self.all_rwse = (
            torch.empty((total_nodes, 16), dtype=torch.float32, device=self.device) if need_rwse else None
        )
        self.all_y = torch.empty((total_graphs,), dtype=torch.float32, device=self.device)

        # Keep node_ptr and edge_ptr on BOTH CPU (for zero-sync int bounds) and GPU (for index gathering)
        self.node_ptr_cpu = torch.empty((total_graphs + 1,), dtype=torch.int64)
        self.edge_ptr_cpu = torch.empty((total_graphs + 1,), dtype=torch.int64)
        self.node_ptr_cpu[0] = 0
        self.edge_ptr_cpu[0] = 0

        g_off = 0
        n_off = 0
        e_off = 0
        for sp in shard_paths:
            data, slices = torch.load(sp, map_location="cpu", weights_only=False, mmap=True)
            g_cnt = int(slices["x"].numel() - 1)
            n_cnt = int(slices["x"][-1].item())
            e_cnt = int(slices["edge_index"][-1].item())

            self.node_ptr_cpu[g_off + 1 : g_off + g_cnt + 1] = slices["x"][1:] + n_off
            self.edge_ptr_cpu[g_off + 1 : g_off + g_cnt + 1] = slices["edge_index"][1:] + e_off

            self.all_x[n_off : n_off + n_cnt].copy_(data.x.to(torch.uint8), non_blocking=True)
            self.all_edge_index[:, e_off : e_off + e_cnt].copy_(data.edge_index.to(torch.int16), non_blocking=True)
            self.all_edge_attr[e_off : e_off + e_cnt].copy_(data.edge_attr.to(torch.uint8), non_blocking=True)
            if self.all_rwse is not None:
                self.all_rwse[n_off : n_off + n_cnt].copy_(data.random_walk_pe, non_blocking=True)
            self.all_y[g_off : g_off + g_cnt].copy_(data.y.view(-1).to(torch.float32), non_blocking=True)

            g_off += g_cnt
            n_off += n_cnt
            e_off += e_cnt

        self.node_ptr_gpu = self.node_ptr_cpu.to(self.device, non_blocking=True)
        self.edge_ptr_gpu = self.edge_ptr_cpu.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        vram_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3) if self.device.type == "cuda" else 0.0
        print(
            f"[VRAMGraphStore] Loaded {len(shard_paths)} shards ({total_graphs:,} graphs, "
            f"{total_nodes:,} nodes, {total_edges:,} edges) directly into GPU VRAM "
            f"({vram_gb:.2f} GB allocated) in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

    def build_batch(self, batch_ids_cpu: torch.Tensor, batch_ids_gpu: torch.Tensor, b_arange_gpu: torch.Tensor, family: str) -> GPUBatch:
        B = int(batch_ids_cpu.numel())
        # Compute integer bounds on CPU in ~2us -> ZERO GPU stream synchronization!
        n_counts_cpu = self.node_ptr_cpu[batch_ids_cpu + 1] - self.node_ptr_cpu[batch_ids_cpu]
        e_counts_cpu = self.edge_ptr_cpu[batch_ids_cpu + 1] - self.edge_ptr_cpu[batch_ids_cpu]
        total_nodes = int(n_counts_cpu.sum().item())
        max_nodes = int(n_counts_cpu.max().item())
        total_edges = int(e_counts_cpu.sum().item())

        # Pure asynchronous GPU kernel launches
        n_start = self.node_ptr_gpu.index_select(0, batch_ids_gpu)
        n_end = self.node_ptr_gpu.index_select(0, batch_ids_gpu + 1)
        n_counts = n_end - n_start
        ptr = torch.zeros(B + 1, dtype=torch.int64, device=self.device)
        torch.cumsum(n_counts, dim=0, out=ptr[1:])

        batch_vec = torch.repeat_interleave(b_arange_gpu, n_counts, output_size=total_nodes)
        local_node = torch.arange(total_nodes, dtype=torch.int64, device=self.device) - ptr[:-1].index_select(0, batch_vec)
        global_node = n_start.index_select(0, batch_vec) + local_node

        x = self.all_x.index_select(0, global_node).to(torch.int64)
        rwse = self.all_rwse.index_select(0, global_node) if self.all_rwse is not None else None

        e_start = self.edge_ptr_gpu.index_select(0, batch_ids_gpu)
        e_end = self.edge_ptr_gpu.index_select(0, batch_ids_gpu + 1)
        e_counts = e_end - e_start
        e_ptr = torch.zeros(B + 1, dtype=torch.int64, device=self.device)
        torch.cumsum(e_counts, dim=0, out=e_ptr[1:])

        edge_graph = torch.repeat_interleave(b_arange_gpu, e_counts, output_size=total_edges)
        local_edge_pos = torch.arange(total_edges, dtype=torch.int64, device=self.device) - e_ptr[:-1].index_select(0, edge_graph)
        global_edge = e_start.index_select(0, edge_graph) + local_edge_pos

        edge_local = self.all_edge_index.index_select(1, global_edge).to(torch.int64)
        edge_index = edge_local + ptr[:-1].index_select(0, edge_graph).unsqueeze(0)
        edge_attr = self.all_edge_attr.index_select(0, global_edge).to(torch.int64)
        y = self.all_y.index_select(0, batch_ids_gpu)

        flat_index = batch_vec * max_nodes + local_node
        valid = torch.arange(max_nodes, dtype=torch.int64, device=self.device).unsqueeze(0) < n_counts.unsqueeze(1)

        if family == "neural_atom_k1":
            return GPUBatch(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch_vec,
                random_walk_pe=rwse,
                y=y,
                num_graphs=B,
                gpu_dense_layout={
                    "flat_index": flat_index,
                    "valid": valid,
                    "nodes": total_nodes,
                    "graphs": B,
                    "max_nodes": max_nodes,
                },
            )
        return GPUBatch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_vec,
            random_walk_pe=None,
            y=y,
            num_graphs=B,
            execution_layout={
                "batch_size": B,
                "max_nodes": max_nodes,
                "mask": valid,
                "flat_index": flat_index,
                "edges": (edge_graph, edge_local[0], edge_local[1]),
            },
        )


class _PassIterator:
    def __init__(self, store: VRAMGraphStore, recipe: FamilyRecipe, pass_index: int, start_batch: int):
        self.store = store
        self.recipe = recipe
        self.pass_index = pass_index
        self.start_batch = start_batch
        self.batch_size = recipe.batch_size
        self.total_batches = recipe.train_rows // self.batch_size

    def __len__(self) -> int:
        return self.total_batches - self.start_batch

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.recipe.seed + self.pass_index)
        order_cpu = torch.randperm(self.recipe.train_rows, generator=gen, dtype=torch.int64)
        order_gpu = order_cpu.to(self.store.device, non_blocking=True)
        b_arange_gpu = torch.arange(self.batch_size, dtype=torch.int64, device=self.store.device)
        B = self.batch_size
        family = self.recipe.family
        for b in range(self.start_batch, self.total_batches):
            s = b * B
            e = s + B
            yield self.store.build_batch(order_cpu[s:e], order_gpu[s:e], b_arange_gpu, family)


def make_vram_loaders(recipe: FamilyRecipe, store: VRAMGraphStore) -> FamilyLoaders:
    """Create an exact FamilyLoaders instance backed by VRAMGraphStore."""
    loaders = object.__new__(FamilyLoaders)
    loaders.recipe = recipe
    loaders.store = store
    loaders.data = _DataStub(store.num_graphs, recipe.target_mean_eV, recipe.target_std_eV)
    loaders.sampler = PassBatchSampler(
        store.num_graphs,
        recipe.batch_size,
        recipe.seed,
        algorithm=recipe.sampler,
        drop_last=recipe.drop_last,
    )
    loaders.generator = torch.Generator()
    loaders.training = None
    loaders.development = None
    loaders.development_cache = None
    loaders.closed = False

    def _train(self, pass_index: int, start_batch: int):
        if self.closed:
            raise RuntimeError("Training loader pool is closed")
        validated = PassBatchSampler(
            self.data.train_rows,
            self.recipe.batch_size,
            self.recipe.seed,
            algorithm=self.recipe.sampler,
            pass_index=pass_index,
            start_batch=start_batch,
            drop_last=self.recipe.drop_last,
        )
        self.sampler.pass_index = validated.pass_index
        self.sampler.start_batch = validated.start_batch
        self.generator.manual_seed(self.recipe.seed + pass_index)
        return _PassIterator(self.store, self.recipe, pass_index, start_batch)

    loaders.train = MethodType(_train, loaders)
    return loaders


def run_arm(arm: str, *, code_dir: Path, data_root: Path, run_root: Path, max_wall_seconds: float, device: str = "cuda",
            batch_size: int = BATCH_SIZE, lr_mult: float = LR_MULT, use_compile: bool = False,
            compare_log: Path | None = None) -> str:
    out_dir = run_root / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    done_file = out_dir / "training_complete.json"
    if done_file.is_file() and (out_dir / "selected_model.pt").is_file():
        print(f"[{arm}] Already completed ({done_file}). Skipping.", flush=True)
        return "COMPLETE"

    configure_a100_fast_mode(42)
    store = VRAMGraphStore(data_root, need_rwse=(arm == "k1_muon"), device=device)
    # CHANGED: arm-specific batch size; steps/epoch and total steps derived from the store (drop_last)
    arm_batch = batch_size if arm == "k1_muon" else GPTRANS_BATCH_SIZE
    arm_lr_mult = lr_mult if arm == "k1_muon" else 1.0
    recipe, init_name = make_recipe(arm, total_rows=store.num_graphs, batch_size=arm_batch, lr_mult=arm_lr_mult)
    steps_per_epoch = steps_per_epoch_for(store.num_graphs, recipe.batch_size)
    total_steps = steps_per_epoch * TOTAL_EPOCHS
    assert recipe.optimizer_steps == total_steps, (recipe.optimizer_steps, total_steps)
    init_path = code_dir / init_name
    print(f"[{arm}] batch={recipe.batch_size} lr={recipe.learning_rate:.3e} (x{arm_lr_mult:g} on both groups) "
          f"steps/epoch={steps_per_epoch:,} total_steps={total_steps:,} compile={use_compile}", flush=True)

    kernel = FamilyKernel(recipe, init_path, device=device)
    if use_compile:
        if arm != "k1_muon":
            raise SystemExit("--compile is only offered for the k1_muon arm")
        print(f"[{arm}] torch.compile(dynamic=True) ENABLED - accuracy NOT validated (round-3 compile arm did not finish).", flush=True)
        kernel.model.compile(dynamic=True)
    loaders = make_vram_loaders(recipe, store)

    ckpt_path = out_dir / "last_checkpoint.pt"
    log_path = out_dir / "train_log.jsonl"
    config_path = out_dir / "run_config.json"
    this_config = {"arm": arm, "batch_size": recipe.batch_size, "lr_mult": arm_lr_mult, "learning_rate": recipe.learning_rate,
                   "total_steps": total_steps, "steps_per_epoch": steps_per_epoch, "compile": bool(use_compile)}
    old_log = load_compare_log(compare_log) if arm == "k1_muon" else {}
    if old_log:
        print(f"[{arm}] comparing per-epoch train MAE with the OLD run ({len(old_log)} epochs read from {compare_log})", flush=True)
    t_start = time.perf_counter()
    step_t0 = [time.perf_counter()]
    step_n0 = [0]

    def on_selected(progress, weights, state, evaluation):
        atomic_torch_save(
            out_dir / "selected_model.pt",
            {
                "arm": arm,
                "model": state,
                "weights": weights,
                "selection": recipe.selection,
                "optimizer_step": progress.optimizer_step,
                "epoch": progress.pass_index,
            },
        )
        if kernel.ema is not None:
            atomic_torch_save(out_dir / "ema_model.pt", {"arm": arm, "model": kernel.ema.state_dict(), "optimizer_step": progress.optimizer_step})
        atomic_torch_save(out_dir / "live_model.pt", {"arm": arm, "model": kernel.model.state_dict(), "optimizer_step": progress.optimizer_step})

    def on_checkpoint(progress, state):
        t_c = time.perf_counter()
        atomic_torch_save(ckpt_path, state)
        if hasattr(os, "sync"):
            os.sync()
        print(
            f"[{arm}] Checkpoint saved at epoch={progress.pass_index}/{TOTAL_EPOCHS} "
            f"step={progress.optimizer_step}/{total_steps} status={progress.status} ({time.perf_counter() - t_c:.1f}s)",
            flush=True,
        )

    def on_pass_finished(progress, metrics):
        norm_mae = float(metrics["normalized_train_gap_mae"])
        ev_mae = norm_mae * recipe.target_std_eV
        lr = metrics["learning_rates_used"][0] if metrics.get("learning_rates_used") else recipe.learning_rate
        elapsed = time.perf_counter() - t_start
        record = {
            "arm": arm,
            "epoch": progress.pass_index,
            "optimizer_step": progress.optimizer_step,
            "sample_presentations": progress.sample_presentations,
            "train_mae_norm": round(norm_mae, 6),
            "train_mae_eV": round(ev_mae, 6),
            "lr": lr,
            "elapsed_sec": round(elapsed, 1),
            "batch_size": recipe.batch_size,
        }
        info, warn = compare_message(progress.pass_index, ev_mae, old_log)  # CHANGED: compare with the old run
        if info:
            record["old_run_train_mae_eV"] = old_log[progress.pass_index]
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
        print(
            f"=== [{arm}] Epoch {progress.pass_index:02d}/{TOTAL_EPOCHS} DONE | "
            f"step {progress.optimizer_step:,}/{total_steps:,} | "
            f"train_MAE={ev_mae:.4f} eV (norm {norm_mae:.4f}) | lr={lr:.2e} | elapsed={elapsed/60:.1f}m"
            + (f" | {info}" if info else "") + " ===",
            flush=True,
        )
        if warn:
            print(warn, flush=True)

    def on_batch_finished(progress, *, learning_rates):
        step = progress.optimizer_step
        if step == 1 or step % LOG_EVERY == 0:
            now = time.perf_counter()
            dt = max(1e-6, now - step_t0[0])
            dn = step - step_n0[0]
            it_s = dn / dt if dn > 0 else 0.0
            step_t0[0], step_n0[0] = now, step
            epoch_idx = (step - 1) // steps_per_epoch + 1
            batch_in_epoch = (step - 1) % steps_per_epoch + 1
            eta_h = eta_hours(step, total_steps, it_s)
            print(
                f"[{arm}] Ep {epoch_idx:02d}/{TOTAL_EPOCHS} | "
                f"Batch {batch_in_epoch:5d}/{steps_per_epoch} ({step*100/total_steps:5.1f}%) | "
                f"LR {learning_rates[0]:.2e} | {it_s:.1f} step/s | ETA {eta_h:.2f}h",
                flush=True,
            )

    session = FamilyTrainingSession(
        kernel,
        loaders,
        checkpoint=on_checkpoint,
        selected=on_selected,
        pass_finished=on_pass_finished,
        batch_finished=on_batch_finished,
    )

    if ckpt_path.is_file():
        print(f"[{arm}] Resuming from checkpoint: {ckpt_path}", flush=True)
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sidecar = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
        check_resume_compat(saved, sidecar, recipe.batch_size, recipe.learning_rate, arm_lr_mult)  # CHANGED: refuse mismatches
        if isinstance(saved.get("cursor"), dict):
            saved["cursor"]["recipe_identity"] = recipe.identity
        session.load_state_dict(saved)
        step_n0[0] = session.progress.optimizer_step
        print(
            f"[{arm}] Restored at epoch={session.progress.pass_index} "
            f"step={session.progress.optimizer_step:,}/{total_steps:,}",
            flush=True,
        )
    else:
        atomic_json(config_path, this_config)  # CHANGED: remembered for the resume check
        print(f"[{arm}] Starting fresh A100 VRAM-resident training: 60 epochs ({total_steps:,} steps)", flush=True)

    stop_requested = [False]
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.__setitem__(0, True))

    def should_pause():
        if stop_requested[0]:
            return True
        if max_wall_seconds > 0 and (time.perf_counter() - t_start) >= max_wall_seconds:
            return True
        return False

    try:
        final_progress = session.run(should_pause=should_pause)
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)
        session.close()
        del store, loaders, kernel, session
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if final_progress.status == "complete":
        atomic_json(
            done_file,
            {
                "arm": arm,
                "status": "COMPLETE",
                "optimizer_steps": final_progress.optimizer_step,
                "sample_presentations": final_progress.sample_presentations,
                "completed_passes": final_progress.pass_index,
                "selection": recipe.selection,
                "batch_size": recipe.batch_size,
                "learning_rate": recipe.learning_rate,
            },
        )
        if hasattr(os, "sync"):
            os.sync()
        print(f"[{arm}] Training 100% COMPLETE! Models saved to {out_dir}", flush=True)
        return "COMPLETE"

    atomic_json(
        out_dir / "pause_receipt.json",
        {
            "arm": arm,
            "status": "PAUSED",
            "optimizer_steps": final_progress.optimizer_step,
            "sample_presentations": final_progress.sample_presentations,
            "completed_passes": final_progress.pass_index,
        },
    )
    if hasattr(os, "sync"):
        os.sync()
    print(f"[{arm}] Cleanly PAUSED at step {final_progress.optimizer_step:,}/{total_steps:,}. Re-run notebook to resume.", flush=True)
    return "PAUSED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("k1_muon", "gptrans_pairnorm", "both"), default="k1_muon")  # CHANGED default
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=9.5 * 3600)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    # CHANGED: production knobs
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="physical batch for the k1_muon arm (default 1024)")
    parser.add_argument("--lr-mult", type=float, default=LR_MULT, help="multiplier on the base lr 4e-4 for BOTH optimizer groups (default 6 -> 2.4e-3)")
    parser.add_argument("--compile", action="store_true", help="torch.compile(dynamic=True) on the K1 model - accuracy NOT validated")
    parser.add_argument("--compare-log", type=Path, default=None, help="OLD run train_log.jsonl (read-only) for the per-epoch comparison")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.lr_mult <= 0:
        parser.error("--batch-size must be >= 1 and --lr-mult must be > 0")
    guard_run_root(args.run_root)

    arms = ("k1_muon", "gptrans_pairnorm") if args.arm == "both" else (args.arm,)
    session_t0 = time.perf_counter()
    for arm in arms:
        remaining = args.max_wall_seconds - (time.perf_counter() - session_t0)
        if remaining < 300:
            print(f"Remaining session budget ({remaining:.0f}s) < 300s; stopping before {arm}.", flush=True)
            break
        status = run_arm(
            arm,
            code_dir=args.code_dir,
            data_root=args.data_root,
            run_root=args.run_root,
            max_wall_seconds=remaining,
            device=args.device,
            batch_size=args.batch_size,
            lr_mult=args.lr_mult,
            use_compile=args.compile,
            compare_log=args.compare_log,
        )
        if status == "PAUSED":
            break


if __name__ == "__main__":
    main()
