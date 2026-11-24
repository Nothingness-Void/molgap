"""Wave-1 Kaggle T4x2 runner: native speed matrix, FP16 parity, 200-epoch scaling, regularized DeMol-CIN, and Fast ETKDG 3D."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from fast_gptrans import (
    EXPECTED_POS_MANIFEST_SHA256,
    FROZEN_GPTRANS_INIT_SHA256,
    PROTECTED_ROW_START,
    TRAIN_100K_MEAN_EV,
    TRAIN_100K_STD_EV,
    FastBondLocalPairNormGPTrans,
    GPUResidentGraphStore,
    load_pos_tables,
    sha256_file,
)
from fast_demol_cin import FastDeMolCINReg, GPUResidentCellStore


def find_input_assets(input_root: Path) -> dict[str, Path]:
    assets: dict[str, Path] = {}
    for p in sorted(input_root.rglob("manifest.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ident = meta.get("identity", {})
        if ident.get("name") == "ogb-train-100k":
            assets["manifest_100k"] = p
            break
    for p in sorted(input_root.rglob("pos_manifest.json")):
        if sha256_file(p) == EXPECTED_POS_MANIFEST_SHA256:
            assets["pos_manifest_100k"] = p
            break
    for p in sorted(input_root.rglob("gptrans.pt")):
        if sha256_file(p) == FROZEN_GPTRANS_INIT_SHA256:
            assets["gptrans_init"] = p
            break
    return assets


def load_100k_shards(manifest_path: Path) -> tuple[list[Path], list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    train_shards, dev_shards = [], []
    for item in manifest["geometry_shards"]:
        role = item["role"]
        fpath = (root / item["file"]).resolve()
        if int(item["source_idx_max"]) >= PROTECTED_ROW_START:
            raise ValueError("Protected row boundary crossed in shard manifest")
        if role == "train":
            train_shards.append(fpath)
        elif role == "development":
            dev_shards.append(fpath)
    assert len(train_shards) == 2 and len(dev_shards) == 1
    return train_shards, dev_shards


class FastBoundEMA:
    """In-place FP32 tensor EMA across model parameters + buffers."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.buf_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name, p in model.named_parameters():
            t = p.detach().clone().float()
            self.shadow[name] = t
            self.pairs.append((t, p.detach()))
        for name, b in model.named_buffers():
            t = b.detach().clone()
            self.shadow[name] = t
            self.buf_pairs.append((t, b.detach()))
        self._targets = [a for a, _ in self.pairs]
        self._sources = [b for _, b in self.pairs]

    @torch.no_grad()
    def update(self) -> None:
        if self._targets:
            torch._foreach_mul_(self._targets, self.decay)
            torch._foreach_add_(self._targets, self._sources, alpha=1.0 - self.decay)
        for t, b in self.buf_pairs:
            t.copy_(b)

    @torch.no_grad()
    def copy_to_model(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {}
        for name, p in model.named_parameters():
            backup[name] = p.detach().clone()
            p.detach().copy_(self.shadow[name].to(p.dtype))
        return backup

    @torch.no_grad()
    def restore_model(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            p.detach().copy_(backup[name])

    def cpu_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}


def lr_for_epoch(epoch_idx: int, total_epochs: int, peak_lr: float, warmup_epochs: int = 4, min_lr: float = 1e-6) -> float:
    if epoch_idx < warmup_epochs:
        return peak_lr * float(epoch_idx + 1) / float(warmup_epochs)
    prog = float(epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs - 1))
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_gptrans(m: torch.nn.Module, batch: dict[str, torch.Tensor | None]) -> torch.Tensor:
    return m(
        batch["dense_x"],
        batch["degree"],
        batch["atom_mask"],
        batch["spd"],
        batch["edge_attr"],
        batch["bond_pair_idx"],
        batch["edge_src_flat"],
        batch["edge_dst_flat"],
        batch["edge_pair_flat"],
        batch["edge_target_scatter"],
        batch["pos"],
        batch["geom_valid"],
    )


def try_compile_gptrans(
    model: torch.nn.Module,
    store: GPUResidentGraphStore,
    *,
    mode: str,
    bs: int,
    use_amp_fp16: bool,
) -> tuple[torch.nn.Module, str, str | None]:
    if mode == "none" or not hasattr(torch, "compile"):
        return model, "none", None
    try:
        opt_model = torch.compile(model, mode=mode, dynamic=False)
        idx = torch.arange(bs, device="cpu")
        batch = store.get_batch(idx, pad_to_n=20, e_bucket=256)
        opt_model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp_fp16)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp_fp16):
                pred = _forward_gptrans(opt_model, batch)
                loss = F.l1_loss(pred.float(), (batch["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        return opt_model, mode, None
    except Exception as exc:
        return model, "none", f"{type(exc).__name__}: {str(exc)[:300]}"


def probe_best_compile_mode(
    gptrans_init: Path,
    train_store: GPUResidentGraphStore,
    *,
    bs: int,
    use_3d: bool,
    candidates: tuple[str, ...] = ("none", "default", "reduce-overhead"),
) -> tuple[str, dict]:
    """Quick ~20-40s probe on the current GPU to choose the fastest stable mode."""
    results = {}
    best_mode = "none"
    best_ms = float("inf")
    batches = [train_store.get_batch(torch.arange(k * bs, (k + 1) * bs), pad_to_n=20, e_bucket=256) for k in range(4)]

    for mode in candidates:
        torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        m = FastBondLocalPairNormGPTrans(use_3d=use_3d).cuda()
        m.load_from_gptrans_checkpoint(gptrans_init)
        m = m.cuda()
        compiled, used_mode, err = try_compile_gptrans(m, train_store, mode=mode, bs=bs, use_amp_fp16=True)
        if err is not None:
            results[mode] = {"ok": False, "error": err}
            continue
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05, fused=True)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        compiled.train()
        for k in range(3):
            b = batches[k % 4]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                pred = _forward_gptrans(compiled, b)
                loss = F.l1_loss(pred.float(), (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 24
        for k in range(iters):
            b = batches[k % 4]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                pred = _forward_gptrans(compiled, b)
                loss = F.l1_loss(pred.float(), (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / iters
        results[mode] = {"ok": True, "ms_per_step": round(ms, 3), "mol_per_sec": round(bs / (ms / 1000.0), 1)}
        if ms < best_ms:
            best_ms = ms
            best_mode = used_mode
        del compiled, m, opt, scaler
        torch.cuda.empty_cache()

    return best_mode, results


@torch.no_grad()
def evaluate_gptrans(
    eval_fn,
    dev_store: GPUResidentGraphStore,
    *,
    bs: int = 256,
    use_amp_fp16: bool = True,
    pad_to_n: int = 20,
) -> tuple[float, torch.Tensor]:
    eval_fn.eval()
    n = len(dev_store)
    preds = []
    for start in range(0, n, bs):
        stop = min(n, start + bs)
        idx = torch.arange(start, stop, dtype=torch.long)
        b = dev_store.get_batch(idx, pad_to_n=pad_to_n, e_bucket=256)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp_fp16):
            p = _forward_gptrans(eval_fn, b).float() * TRAIN_100K_STD_EV + TRAIN_100K_MEAN_EV
        preds.append(p.cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = dev_store.y.cpu()
    mae = float((pred_all - target_all).abs().mean().item())
    return mae, pred_all


def train_gptrans_run(
    *,
    run_name: str,
    out_dir: Path,
    gptrans_init: Path,
    train_store: GPUResidentGraphStore,
    dev_store: GPUResidentGraphStore,
    epochs: int,
    bs: int,
    peak_lr: float,
    ema_decay: float,
    aux_ema_decay: float | None,
    use_amp_fp16: bool,
    compile_mode: str,
    det: bool,
    use_3d: bool,
    seed: int = 42,
    eval_every: int = 1,
    dense_eval_windows: tuple[tuple[int, int], ...] = (),
    snapshot_epochs: tuple[int, ...] = (),
    pad_to_n: int = 20,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    if det:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True

    model = FastBondLocalPairNormGPTrans(use_3d=use_3d).cuda()
    model.load_from_gptrans_checkpoint(gptrans_init)
    model = model.cuda()

    opt_model, actual_compile, compile_err = try_compile_gptrans(
        model, train_store, mode=compile_mode, bs=bs, use_amp_fp16=use_amp_fp16
    )
    if compile_err:
        print(f"[{run_name}] compile fallback ({compile_mode} -> none): {compile_err}", flush=True)

    # Eager evaluation path guarantees EMA weight swapping is clean and exact
    eval_model = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.05, fused=(not det))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_fp16)
    ema = FastBoundEMA(model, decay=ema_decay)
    aux_ema = FastBoundEMA(model, decay=aux_ema_decay) if aux_ema_decay is not None else None

    train_rows = len(train_store)
    steps_per_ep = train_rows // bs
    log_path = out_dir / f"{run_name}_train_log.jsonl"
    best_mae = float("inf")
    best_ep = -1
    best_preds = None
    best_aux_mae = float("inf")
    best_aux_ep = -1
    t_start = time.perf_counter()
    total_steps = 0
    scaler_skips = 0

    def should_eval(ep1: int) -> bool:
        if ep1 == 1 or ep1 == epochs or ep1 in snapshot_epochs or (ep1 % eval_every == 0):
            return True
        return any(lo <= ep1 <= hi for lo, hi in dense_eval_windows)

    for ep0 in range(epochs):
        ep1 = ep0 + 1
        cur_lr = lr_for_epoch(ep0, epochs, peak_lr, warmup_epochs=4, min_lr=1e-6)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        gen = torch.Generator().manual_seed(seed + ep0)
        perm = torch.randperm(train_rows, generator=gen)[: steps_per_ep * bs].view(steps_per_ep, bs)

        opt_model.train()
        ep_loss_sum = 0.0
        t_ep = time.perf_counter()
        for step_idx in range(steps_per_ep):
            b = train_store.get_batch(perm[step_idx], pad_to_n=pad_to_n, e_bucket=256)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp_fp16):
                pred = _forward_gptrans(opt_model, b)
                target_norm = (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV
                loss = F.l1_loss(pred.float(), target_norm)
            if use_amp_fp16:
                prev_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < prev_scale:
                    scaler_skips += 1
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ema.update()
            if aux_ema is not None:
                aux_ema.update()
            ep_loss_sum += float(loss.detach().item()) * TRAIN_100K_STD_EV
            total_steps += 1

        train_mae_ev = ep_loss_sum / float(steps_per_ep)
        rec = {
            "run": run_name,
            "epoch": ep1,
            "step": total_steps,
            "lr": cur_lr,
            "train_mae_eV": train_mae_ev,
            "epoch_s": round(time.perf_counter() - t_ep, 2),
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            "scaler_skips": scaler_skips,
        }

        if should_eval(ep1):
            backup = ema.copy_to_model(eval_model)
            dev_mae, dev_preds = evaluate_gptrans(eval_model, dev_store, bs=max(bs, 256), use_amp_fp16=use_amp_fp16, pad_to_n=pad_to_n)
            ema.restore_model(eval_model, backup)
            rec["dev_mae_eV"] = dev_mae
            if dev_mae < best_mae:
                best_mae = dev_mae
                best_ep = ep1
                best_preds = dev_preds
                torch.save(ema.cpu_state_dict(), out_dir / f"{run_name}_best_model.pt")
                torch.save({"source_idx": dev_store.source_idx.cpu(), "pred_eV": dev_preds, "target_eV": dev_store.y.cpu(), "epoch": ep1}, out_dir / f"{run_name}_best_dev_preds.pt")

            if aux_ema is not None:
                backup2 = aux_ema.copy_to_model(eval_model)
                aux_mae, aux_preds = evaluate_gptrans(eval_model, dev_store, bs=max(bs, 256), use_amp_fp16=use_amp_fp16, pad_to_n=pad_to_n)
                aux_ema.restore_model(eval_model, backup2)
                rec["aux_dev_mae_eV"] = aux_mae
                if aux_mae < best_aux_mae:
                    best_aux_mae = aux_mae
                    best_aux_ep = ep1
                if ep1 in snapshot_epochs:
                    torch.save({"source_idx": dev_store.source_idx.cpu(), "pred_eV": aux_preds, "target_eV": dev_store.y.cpu(), "epoch": ep1}, out_dir / f"{run_name}_aux_ep{ep1:03d}_dev_preds.pt")

            if ep1 in snapshot_epochs:
                torch.save(ema.cpu_state_dict(), out_dir / f"{run_name}_ep{ep1:03d}_model.pt")
                torch.save({"source_idx": dev_store.source_idx.cpu(), "pred_eV": dev_preds, "target_eV": dev_store.y.cpu(), "epoch": ep1}, out_dir / f"{run_name}_ep{ep1:03d}_dev_preds.pt")

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if ep1 == 1 or ep1 % 5 == 0 or ep1 == epochs or ep1 in snapshot_epochs:
            dm = f"{rec['dev_mae_eV']*1000:.3f}meV" if "dev_mae_eV" in rec else "-"
            am = f" aux={rec['aux_dev_mae_eV']*1000:.3f}meV" if "aux_dev_mae_eV" in rec else ""
            print(f"[{run_name}] ep {ep1:3d}/{epochs} | train={train_mae_ev*1000:.3f}meV | dev={dm}{am} | {rec['epoch_s']:.1f}s/ep | total={rec['elapsed_s']:.0f}s", flush=True)

    # Save final checkpoint & predictions explicitly
    backup = ema.copy_to_model(eval_model)
    final_mae, final_preds = evaluate_gptrans(eval_model, dev_store, bs=max(bs, 256), use_amp_fp16=use_amp_fp16, pad_to_n=pad_to_n)
    ema.restore_model(eval_model, backup)
    torch.save(ema.cpu_state_dict(), out_dir / f"{run_name}_final_model.pt")
    torch.save({"source_idx": dev_store.source_idx.cpu(), "pred_eV": final_preds, "target_eV": dev_store.y.cpu(), "epoch": epochs}, out_dir / f"{run_name}_final_dev_preds.pt")

    summary = {
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": bs,
        "peak_lr": peak_lr,
        "ema_decay": ema_decay,
        "aux_ema_decay": aux_ema_decay,
        "use_amp_fp16": use_amp_fp16,
        "compile_mode": actual_compile,
        "compile_error": compile_err,
        "det": det,
        "use_3d": use_3d,
        "best_ep": best_ep,
        "best_dev_mae_eV": best_mae,
        "final_dev_mae_eV": final_mae,
        "best_aux_ep": best_aux_ep,
        "best_aux_dev_mae_eV": best_aux_mae if aux_ema is not None else None,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "scaler_skips": scaler_skips,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    (out_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def evaluate_demol(
    model: FastDeMolCINReg,
    dev_store: GPUResidentCellStore,
    *,
    bs: int = 256,
    use_amp_fp16: bool = True,
) -> tuple[float, torch.Tensor]:
    model.eval()
    n = len(dev_store)
    preds = []
    for start in range(0, n, bs):
        stop = min(n, start + bs)
        idx = torch.arange(start, stop, dtype=torch.long)
        b = dev_store.get_batch(idx)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp_fp16):
            p = model(b).float() * TRAIN_100K_STD_EV + TRAIN_100K_MEAN_EV
        preds.append(p.cpu())
    pred_all = torch.cat(preds, dim=0)
    mae = float((pred_all - dev_store.y.cpu()).abs().mean().item())
    return mae, pred_all


def train_demol_run(
    *,
    run_name: str,
    out_dir: Path,
    train_store: GPUResidentCellStore,
    dev_store: GPUResidentCellStore,
    epochs: int = 60,
    bs: int = 128,
    peak_lr: float = 1e-3,
    ema_decay: float = 0.999,
    dropout: float = 0.1,
    drop_path: float = 0.1,
    use_amp_fp16: bool = True,
    use_3d: bool = False,
    seed: int = 42,
    eval_every: int = 2,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True

    model = FastDeMolCINReg(dropout=dropout, drop_path=drop_path, use_3d=use_3d).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.05, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_fp16)
    ema = FastBoundEMA(model, decay=ema_decay)

    train_rows = len(train_store)
    steps_per_ep = train_rows // bs
    log_path = out_dir / f"{run_name}_train_log.jsonl"
    best_mae = float("inf")
    best_ep = -1
    t_start = time.perf_counter()
    total_steps = 0
    scaler_skips = 0

    for ep0 in range(epochs):
        ep1 = ep0 + 1
        cur_lr = lr_for_epoch(ep0, epochs, peak_lr, warmup_epochs=4, min_lr=1e-6)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        gen = torch.Generator().manual_seed(seed + ep0)
        perm = torch.randperm(train_rows, generator=gen)[: steps_per_ep * bs].view(steps_per_ep, bs)

        model.train()
        ep_loss_sum = 0.0
        t_ep = time.perf_counter()
        for step_idx in range(steps_per_ep):
            b = train_store.get_batch(perm[step_idx])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp_fp16):
                pred = model(b)
                target_norm = (b.y - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV
                loss = F.l1_loss(pred.float(), target_norm)
            if use_amp_fp16:
                prev_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < prev_scale:
                    scaler_skips += 1
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            ema.update()
            ep_loss_sum += float(loss.detach().item()) * TRAIN_100K_STD_EV
            total_steps += 1

        train_mae_ev = ep_loss_sum / float(steps_per_ep)
        rec = {
            "run": run_name,
            "epoch": ep1,
            "step": total_steps,
            "lr": cur_lr,
            "train_mae_eV": train_mae_ev,
            "epoch_s": round(time.perf_counter() - t_ep, 2),
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            "scaler_skips": scaler_skips,
        }
        if ep1 == 1 or ep1 % eval_every == 0 or ep1 >= epochs - 5:
            backup = ema.copy_to_model(model)
            dev_mae, dev_preds = evaluate_demol(model, dev_store, bs=max(bs, 256), use_amp_fp16=use_amp_fp16)
            ema.restore_model(model, backup)
            rec["dev_mae_eV"] = dev_mae
            if dev_mae < best_mae:
                best_mae = dev_mae
                best_ep = ep1
                torch.save(ema.cpu_state_dict(), out_dir / f"{run_name}_best_model.pt")
                torch.save({"source_idx": dev_store.source_idx.cpu(), "pred_eV": dev_preds, "target_eV": dev_store.y.cpu(), "epoch": ep1}, out_dir / f"{run_name}_best_dev_preds.pt")

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if ep1 == 1 or ep1 % 5 == 0 or ep1 == epochs:
            dm = f"{rec['dev_mae_eV']*1000:.3f}meV" if "dev_mae_eV" in rec else "-"
            print(f"[{run_name}] ep {ep1:3d}/{epochs} | train={train_mae_ev*1000:.3f}meV | dev={dm} | {rec['epoch_s']:.1f}s/ep | total={rec['elapsed_s']:.0f}s", flush=True)

    backup = ema.copy_to_model(model)
    final_mae, final_preds = evaluate_demol(model, dev_store, bs=max(bs, 256), use_amp_fp16=use_amp_fp16)
    ema.restore_model(model, backup)
    torch.save(ema.cpu_state_dict(), out_dir / f"{run_name}_final_model.pt")
    torch.save({"source_idx": dev_store.source_idx.cpu(), "pred_eV": final_preds, "target_eV": dev_store.y.cpu(), "epoch": epochs}, out_dir / f"{run_name}_final_dev_preds.pt")

    summary = {
        "run_name": run_name,
        "epochs": epochs,
        "batch_size": bs,
        "dropout": dropout,
        "drop_path": drop_path,
        "use_amp_fp16": use_amp_fp16,
        "use_3d": use_3d,
        "best_ep": best_ep,
        "best_dev_mae_eV": best_mae,
        "final_dev_mae_eV": final_mae,
        "giveback_meV": (final_mae - best_mae) * 1000.0,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "scaler_skips": scaler_skips,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }
    (out_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def paired_bootstrap_meV(cand_ev: torch.Tensor, ref_ev: torch.Tensor, target_ev: torch.Tensor, draws: int = 2000, seed: int = 20261124) -> dict:
    ec = (cand_ev - target_ev).abs().numpy()
    er = (ref_ev - target_ev).abs().numpy()
    diff_mev = (ec - er) * 1000.0
    n = len(diff_mev)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        idx = rng.integers(0, n, size=n)
        means[d] = diff_mev[idx].mean()
    # Residual correlation
    rc = cand_ev.numpy() - target_ev.numpy()
    rr = ref_ev.numpy() - target_ev.numpy()
    r_all = float(np.corrcoef(rc, rr)[0, 1])
    ens_mae_mev = float(np.abs(0.5 * (cand_ev.numpy() + ref_ev.numpy()) - target_ev.numpy()).mean() * 1000.0)
    return {
        "cand_mae_meV": float(ec.mean() * 1000.0),
        "ref_mae_meV": float(er.mean() * 1000.0),
        "delta_meV": float(diff_mev.mean()),
        "ci95_meV": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "residual_corr_all": r_all,
        "ensemble_50_50_mae_meV": ens_mae_mev,
    }


def _ddp_bench_worker(rank: int, world_size: int, train_shards_str: list[str], gptrans_init_str: str, per_gpu_bs: int, compile_mode: str, ret_dict) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29515"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    store = GPUResidentGraphStore([Path(p) for p in train_shards_str], device=f"cuda:{rank}", expected_source_start=0, expected_source_stop=100_000)
    m = FastBondLocalPairNormGPTrans(use_3d=False).to(rank)
    m.load_from_gptrans_checkpoint(Path(gptrans_init_str))
    m = m.to(rank)
    if compile_mode != "none" and hasattr(torch, "compile"):
        try:
            m = torch.compile(m, mode=compile_mode, dynamic=False)
        except Exception:
            pass
    ddp_m = DDP(m, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(ddp_m.parameters(), lr=1e-3, weight_decay=0.05, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    batches = [store.get_batch(torch.arange((rank * 16 + k) * per_gpu_bs, (rank * 16 + k + 1) * per_gpu_bs), pad_to_n=20, e_bucket=256) for k in range(8)]
    for k in range(4):
        b = batches[k % 4]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            pred = _forward_gptrans(ddp_m, b)
            loss = F.l1_loss(pred.float(), (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ddp_m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    dist.barrier()
    torch.cuda.synchronize(rank)
    t0 = time.perf_counter()
    iters = 32
    for k in range(iters):
        b = batches[k % 4]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            pred = _forward_gptrans(ddp_m, b)
            loss = F.l1_loss(pred.float(), (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ddp_m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize(rank)
    dist.barrier()
    ms = (time.perf_counter() - t0) * 1000.0 / iters
    if rank == 0:
        global_bs = per_gpu_bs * world_size
        ret_dict[f"ddp_bs{global_bs}"] = {"ms_per_step": round(ms, 3), "mol_per_sec": round(global_bs / (ms / 1000.0), 1)}
    dist.destroy_process_group()


def run_phase_a_speed_matrix(train_shards: list[Path], gptrans_init: Path, out_dir: Path) -> dict:
    """Measure native Kaggle T4 speed across all optimization levers + 2xT4 DDP."""
    print("=== PHASE A: NATIVE T4 SPEED MATRIX BENCHMARK ===", flush=True)
    device = torch.device("cuda:0")
    store_g = GPUResidentGraphStore(train_shards, device=device, expected_source_start=0, expected_source_stop=100_000)

    # Also build synthetic 3D coordinates directly inside a temporary copy for 3D overhead timing
    from fast_gptrans import FastBondLocalPairNormGPTrans
    rows = []

    def bench_gp(name: str, *, bs: int, amp_fp16: bool, compile_mode: str, det: bool, use_3d: bool = False):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if det:
            torch.use_deterministic_algorithms(True)
        else:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.benchmark = True
        m = FastBondLocalPairNormGPTrans(use_3d=use_3d).to(device)
        m.load_from_gptrans_checkpoint(gptrans_init)
        m = m.to(device)
        compiled, used_mode, err = try_compile_gptrans(m, store_g, mode=compile_mode, bs=bs, use_amp_fp16=amp_fp16)
        if err is not None:
            row = {"name": name, "ok": False, "error": err}
            rows.append(row)
            print(" ", row, flush=True)
            return None
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05, fused=(not det))
        scaler = torch.amp.GradScaler("cuda", enabled=amp_fp16)
        batches = [store_g.get_batch(torch.arange(k * bs, (k + 1) * bs), pad_to_n=20, e_bucket=256) for k in range(8)]
        if use_3d:
            for b in batches:
                b["pos"] = torch.randn((bs, 20, 3), device=device)
                b["geom_valid"] = torch.ones((bs,), dtype=torch.bool, device=device)
        compiled.train()
        for k in range(4):
            b = batches[k % 4]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_fp16):
                pred = _forward_gptrans(compiled, b)
                loss = F.l1_loss(pred.float(), (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            if amp_fp16:
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        iters = 32
        for k in range(iters):
            b = batches[k % 4]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_fp16):
                pred = _forward_gptrans(compiled, b)
                loss = F.l1_loss(pred.float(), (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            if amp_fp16:
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        torch.cuda.synchronize(device)
        ms = (time.perf_counter() - t0) * 1000.0 / iters
        row = {
            "name": name,
            "ok": True,
            "bs": bs,
            "amp_fp16": amp_fp16,
            "compile_mode": used_mode,
            "ms_per_step": round(ms, 3),
            "mol_per_sec": round(bs / (ms / 1000.0), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
        }
        rows.append(row)
        print(f"  {row['name']:44s}: {row['ms_per_step']:6.2f} ms | {row['mol_per_sec']:7.1f} mol/s | {row['peak_vram_gb']:.2f}GB", flush=True)
        del compiled, m, opt, scaler
        torch.cuda.empty_cache()
        return row

    r_base = bench_gp("0. GPTrans NHWC FP32 det=True BS128", bs=128, amp_fp16=False, compile_mode="none", det=True)
    r_fp32_nodet = bench_gp("1. GPTrans NHWC FP32 det=False BS128", bs=128, amp_fp16=False, compile_mode="none", det=False)
    bench_gp("2. GPTrans AMP-FP16 eager BS128", bs=128, amp_fp16=True, compile_mode="none", det=False)
    r_def128 = bench_gp("3. GPTrans AMP-FP16 compile(default) BS128", bs=128, amp_fp16=True, compile_mode="default", det=False)
    r_ro128 = bench_gp("4. GPTrans AMP-FP16 compile(reduce-oh) BS128", bs=128, amp_fp16=True, compile_mode="reduce-overhead", det=False)
    bench_gp("5. GPTrans AMP-FP16 eager BS256", bs=256, amp_fp16=True, compile_mode="none", det=False)
    r_def256 = bench_gp("6. GPTrans AMP-FP16 compile(default) BS256", bs=256, amp_fp16=True, compile_mode="default", det=False)
    bench_gp("7. GPTrans AMP-FP16 compile(default) BS512", bs=512, amp_fp16=True, compile_mode="default", det=False)

    # Pick fastest valid compile mode for BS128
    candidates = [r for r in (rows[2], r_def128, r_ro128) if r and r.get("ok")]
    best_c = min(candidates, key=lambda r: r["ms_per_step"])["compile_mode"] if candidates else "none"
    bench_gp(f"8. GPTrans+Fast3D FP16 ({best_c}) BS128", bs=128, amp_fp16=True, compile_mode=best_c, det=False, use_3d=True)
    bench_gp(f"9. GPTrans+Fast3D FP16 ({best_c}) BS256", bs=256, amp_fp16=True, compile_mode=best_c, det=False, use_3d=True)

    # DeMol-CIN quick timing on first 4096 graphs
    from fast_demol_cin import extract_cellular_parts, LEVEL_SPEC, GPUResidentCellStore
    d0, s0 = torch.load(train_shards[0], map_location="cpu", weights_only=False)
    # Build minimal 2048-mol CellStore by writing temp subset? No, measure DeMol directly on first 2048 graphs using GPUResidentCellStore? Wait: GPUResidentCellStore reads shards; build 2048 graphs in ~1s!
    from types import SimpleNamespace
    import fast_demol_cin as fdc
    chunks = {lvl: {f: [] for f in fields} for lvl, fields in LEVEL_SPEC.items()}
    counts = {lvl: [] for lvl in LEVEL_SPEC}
    ys = []
    for i in range(1024):
        xs, xe = int(s0["x"][i]), int(s0["x"][i + 1])
        es, ee = int(s0["edge_index"][i]), int(s0["edge_index"][i + 1])
        parts = extract_cellular_parts(d0.x[xs:xe], d0.edge_index[:, es:ee], d0.edge_attr[es:ee])
        ys.append(d0.y[i:i+1].float())
        for lvl, fields in LEVEL_SPEC.items():
            for f in fields:
                chunks[lvl][f].append(parts[f])
            counts[lvl].append(int(parts[next(iter(fields))].shape[0]))
    mini_store = object.__new__(GPUResidentCellStore)
    mini_store.device = device
    mini_store.n_graphs = 1024
    mini_store.tensors = {lvl: {f: torch.cat(chunks[lvl][f], dim=0).to(device) for f in fields} for lvl, fields in LEVEL_SPEC.items()}
    mini_store.ptr, mini_store.cnt = {}, {}
    for lvl in LEVEL_SPEC:
        ct = torch.tensor(counts[lvl], dtype=torch.long)
        pt = torch.zeros(1025, dtype=torch.long); pt[1:] = torch.cumsum(ct, 0)
        mini_store.cnt[lvl] = ct.to(device); mini_store.ptr[lvl] = pt.to(device)
    mini_store.y = torch.cat(ys, dim=0).to(device)
    mini_store.has_pos = False; mini_store.pos = None; mini_store.geom_valid = None

    for label, bs_d, amp_d, det_d in [
        ("10. DeMolCINReg FP32 det=True BS128", 128, False, True),
        ("11. DeMolCINReg FP32 det=False BS128", 128, False, False),
        ("12. DeMolCINReg AMP-FP16 BS128", 128, True, False),
        ("13. DeMolCINReg AMP-FP16 BS256", 256, True, False),
    ]:
        torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(det_d)
        dm = FastDeMolCINReg(dropout=0.1, drop_path=0.1).to(device)
        dopt = torch.optim.AdamW(dm.parameters(), lr=1e-3, weight_decay=0.05, fused=(not det_d))
        dscal = torch.amp.GradScaler("cuda", enabled=amp_d)
        dbatches = [mini_store.get_batch(torch.arange(k * bs_d, (k + 1) * bs_d)) for k in range(1024 // bs_d)]
        dm.train()
        for k in range(4):
            b = dbatches[k % len(dbatches)]
            dopt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_d):
                pred = dm(b); loss = F.l1_loss(pred.float(), (b.y - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            if amp_d:
                dscal.scale(loss).backward(); dscal.unscale_(dopt); torch.nn.utils.clip_grad_norm_(dm.parameters(), 1.0); dscal.step(dopt); dscal.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(dm.parameters(), 1.0); dopt.step()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        iters = 24
        for k in range(iters):
            b = dbatches[k % len(dbatches)]
            dopt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_d):
                pred = dm(b); loss = F.l1_loss(pred.float(), (b.y - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            if amp_d:
                dscal.scale(loss).backward(); dscal.unscale_(dopt); torch.nn.utils.clip_grad_norm_(dm.parameters(), 1.0); dscal.step(dopt); dscal.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(dm.parameters(), 1.0); dopt.step()
        torch.cuda.synchronize(device)
        ms = (time.perf_counter() - t0) * 1000.0 / iters
        row = {"name": label, "ok": True, "bs": bs_d, "amp_fp16": amp_d, "ms_per_step": round(ms, 3), "mol_per_sec": round(bs_d / (ms / 1000.0), 1)}
        rows.append(row)
        print(f"  {label:44s}: {ms:6.2f} ms | {row['mol_per_sec']:7.1f} mol/s", flush=True)
        del dm, dopt, dscal

    del store_g, mini_store
    torch.cuda.empty_cache()

    # 2x T4 DDP test via mp.spawn if 2 GPUs exist
    ddp_out = {}
    if torch.cuda.device_count() >= 2:
        try:
            mgr = mp.Manager()
            ret = mgr.dict()
            for per_bs in (64, 128):
                mp.spawn(
                    _ddp_bench_worker,
                    args=(2, [str(p) for p in train_shards], str(gptrans_init), per_bs, best_c, ret),
                    nprocs=2,
                    join=True,
                )
            ddp_out = dict(ret)
            print("  DDP 2xT4 results:", ddp_out, flush=True)
        except Exception as exc:
            ddp_out = {"error": f"{type(exc).__name__}: {exc}"}
            print("  DDP 2xT4 skipped/err:", ddp_out, flush=True)

    summary = {
        "device_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "recommended_bs128_compile_mode": best_c,
        "rows": rows,
        "ddp_2x_t4": ddp_out,
    }
    (out_dir / "speed_matrix.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def worker_entrypoint(args_ns) -> None:
    """Dedicated single-GPU worker (invoked via subprocess with CUDA_VISIBLE_DEVICES pinned)."""
    assets = find_input_assets(Path(args_ns.input_root))
    train_shards, dev_shards = load_100k_shards(assets["manifest_100k"])
    out_dir = Path(args_ns.output_dir)

    if args_ns.task == "phase_a_matrix":
        run_phase_a_speed_matrix(train_shards, assets["gptrans_init"], out_dir)
        return

    pos_tables = load_pos_tables(assets["pos_manifest_100k"]) if args_ns.load_pos else None

    if args_ns.task == "gptrans_run":
        train_store = GPUResidentGraphStore(train_shards, device="cuda", expected_source_start=0, expected_source_stop=100_000, pos_tables=pos_tables)
        dev_store = GPUResidentGraphStore(dev_shards, device="cuda", expected_source_start=100_000, expected_source_stop=150_000, pos_tables=pos_tables)
        compile_mode = args_ns.compile_mode
        if compile_mode == "auto":
            compile_mode, probe_res = probe_best_compile_mode(assets["gptrans_init"], train_store, bs=args_ns.bs, use_3d=args_ns.use_3d)
            (out_dir / f"{args_ns.run_name}_compile_probe.json").write_text(json.dumps({"selected": compile_mode, "probe": probe_res}, indent=2), encoding="utf-8")
            print(f"[{args_ns.run_name}] auto-selected compile={compile_mode}: {probe_res}", flush=True)
        snap = tuple(int(x) for x in args_ns.snapshots.split(",") if x.strip())
        dense_wins = ((55, 60), (115, 120), (195, 200)) if args_ns.epochs >= 200 else ((50, 60),)
        train_gptrans_run(
            run_name=args_ns.run_name,
            out_dir=out_dir,
            gptrans_init=assets["gptrans_init"],
            train_store=train_store,
            dev_store=dev_store,
            epochs=args_ns.epochs,
            bs=args_ns.bs,
            peak_lr=args_ns.lr,
            ema_decay=args_ns.ema,
            aux_ema_decay=args_ns.aux_ema,
            use_amp_fp16=bool(args_ns.amp_fp16),
            compile_mode=compile_mode,
            det=bool(args_ns.det),
            use_3d=bool(args_ns.use_3d),
            seed=args_ns.seed,
            eval_every=args_ns.eval_every,
            dense_eval_windows=dense_wins,
            snapshot_epochs=snap,
        )
    elif args_ns.task == "demol_run":
        train_store = GPUResidentCellStore(train_shards, device="cuda", expected_source_start=0, expected_source_stop=100_000, pos_tables=pos_tables)
        dev_store = GPUResidentCellStore(dev_shards, device="cuda", expected_source_start=100_000, expected_source_stop=150_000, pos_tables=pos_tables)
        train_demol_run(
            run_name=args_ns.run_name,
            out_dir=out_dir,
            train_store=train_store,
            dev_store=dev_store,
            epochs=args_ns.epochs,
            bs=args_ns.bs,
            peak_lr=args_ns.lr,
            ema_decay=args_ns.ema,
            dropout=args_ns.dropout,
            drop_path=args_ns.drop_path,
            use_amp_fp16=bool(args_ns.amp_fp16),
            use_3d=bool(args_ns.use_3d),
            seed=args_ns.seed,
            eval_every=args_ns.eval_every,
        )


def _spawn_gpu_cmd(gpu_id: int, script_path: Path, extra_args: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [sys.executable, str(script_path), "--worker", *extra_args]
    return subprocess.Popen(cmd, env=env)


def run_wave1_mode(mode: str, input_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    assets = find_input_assets(input_root)
    print(f"Resolved assets: { {k: str(v) for k, v in assets.items()} }", flush=True)
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}", flush=True)

    if mode == "wave1_kaggle1":
        # Phase A: Native T4 Speed Matrix (single subproc so CUDA contexts are clean for Phase B)
        subprocess.run([sys.executable, str(script_path), "--worker", "--task", "phase_a_matrix", "--input-root", str(input_root), "--output-dir", str(out_dir)], check=True)
        matrix = json.loads((out_dir / "speed_matrix.json").read_text(encoding="utf-8"))
        best_compile = matrix.get("recommended_bs128_compile_mode", "default")

        # Phase B: Parallel GPU0 FP32 det=False vs GPU1 Fast-FP16 + compile
        p0 = _spawn_gpu_cmd(0, script_path, ["--task", "gptrans_run", "--run-name", "k1_fp32_nodet_60ep", "--input-root", str(input_root), "--output-dir", str(out_dir), "--epochs", "60", "--bs", "128", "--lr", "0.001", "--ema", "0.999", "--aux-ema", "0.9995", "--amp-fp16", "0", "--compile-mode", "none", "--det", "0", "--eval-every", "2", "--snapshots", "60"])
        g1 = 1 if n_gpus >= 2 else 0
        p1 = _spawn_gpu_cmd(g1, script_path, ["--task", "gptrans_run", "--run-name", "k1_fast_fp16_60ep", "--input-root", str(input_root), "--output-dir", str(out_dir), "--epochs", "60", "--bs", "128", "--lr", "0.001", "--ema", "0.999", "--aux-ema", "0.9995", "--amp-fp16", "1", "--compile-mode", best_compile, "--det", "0", "--eval-every", "2", "--snapshots", "60"])
        rc0 = p0.wait()
        rc1 = p1.wait()
        if rc0 != 0 or rc1 != 0:
            raise RuntimeError(f"Wave1 Kaggle1 workers failed: rc0={rc0}, rc1={rc1}")

        f_fp32 = torch.load(out_dir / "k1_fp32_nodet_60ep_final_dev_preds.pt", map_location="cpu", weights_only=False)
        f_fp16 = torch.load(out_dir / "k1_fast_fp16_60ep_final_dev_preds.pt", map_location="cpu", weights_only=False)
        b_fp32 = torch.load(out_dir / "k1_fp32_nodet_60ep_best_dev_preds.pt", map_location="cpu", weights_only=False)
        b_fp16 = torch.load(out_dir / "k1_fast_fp16_60ep_best_dev_preds.pt", map_location="cpu", weights_only=False)
        s_fp32 = json.loads((out_dir / "k1_fp32_nodet_60ep_summary.json").read_text())
        s_fp16 = json.loads((out_dir / "k1_fast_fp16_60ep_summary.json").read_text())
        parity = {
            "fp32_summary": s_fp32,
            "fp16_summary": s_fp16,
            "wall_speedup_fp16_vs_fp32": round(s_fp32["wall_seconds"] / max(s_fp16["wall_seconds"], 1e-6), 3),
            "best_paired_fp16_minus_fp32": paired_bootstrap_meV(b_fp16["pred_eV"], b_fp32["pred_eV"], b_fp32["target_eV"]),
            "final_paired_fp16_minus_fp32": paired_bootstrap_meV(f_fp16["pred_eV"], f_fp32["pred_eV"], f_fp32["target_eV"]),
        }
        (out_dir / "parity_summary.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
        print("=== PARITY SUMMARY ===\n" + json.dumps(parity, indent=2), flush=True)

    elif mode == "wave1_kaggle2":
        # GPU0: 200 epochs BS128 | GPU1: Run 2A DeMolCINReg 60ep, then Run 2B 200 epochs BS256
        p0 = _spawn_gpu_cmd(0, script_path, ["--task", "gptrans_run", "--run-name", "k2_gptrans_200ep_bs128", "--input-root", str(input_root), "--output-dir", str(out_dir), "--epochs", "200", "--bs", "128", "--lr", "0.001", "--ema", "0.999", "--aux-ema", "0.9995", "--amp-fp16", "1", "--compile-mode", "auto", "--det", "0", "--eval-every", "5", "--snapshots", "60,100,120,160,200"])
        g1 = 1 if n_gpus >= 2 else 0
        if n_gpus >= 2:
            p1a = _spawn_gpu_cmd(g1, script_path, ["--task", "demol_run", "--run-name", "k2_demol_reg_60ep", "--input-root", str(input_root), "--output-dir", str(out_dir), "--epochs", "60", "--bs", "128", "--lr", "0.001", "--ema", "0.999", "--dropout", "0.1", "--drop-path", "0.1", "--amp-fp16", "1", "--eval-every", "2"])
            rc1a = p1a.wait()
            if rc1a != 0:
                raise RuntimeError(f"k2_demol_reg_60ep failed rc={rc1a}")
            p1b = _spawn_gpu_cmd(g1, script_path, ["--task", "gptrans_run", "--run-name", "k2_gptrans_200ep_bs256", "--input-root", str(input_root), "--output-dir", str(out_dir), "--epochs", "200", "--bs", "256", "--lr", "0.0014142136", "--ema", "0.998001", "--aux-ema", "0.999", "--amp-fp16", "1", "--compile-mode", "auto", "--det", "0", "--eval-every", "5", "--snapshots", "60,100,120,160,200"])
            rc1b = p1b.wait()
            if rc1b != 0:
                raise RuntimeError(f"k2_gptrans_200ep_bs256 failed rc={rc1b}")
        rc0 = p0.wait()
        if rc0 != 0:
            raise RuntimeError(f"k2_gptrans_200ep_bs128 failed rc={rc0}")

        # Cross-run analysis: 200ep BS128 vs 60ep BS128 vs 200ep BS256 + cross-family ensemble with regularized DeMol
        s200 = torch.load(out_dir / "k2_gptrans_200ep_bs128_final_dev_preds.pt", map_location="cpu", weights_only=False)
        s060 = torch.load(out_dir / "k2_gptrans_200ep_bs128_ep060_dev_preds.pt", map_location="cpu", weights_only=False)
        sb256 = torch.load(out_dir / "k2_gptrans_200ep_bs256_final_dev_preds.pt", map_location="cpu", weights_only=False) if (out_dir / "k2_gptrans_200ep_bs256_final_dev_preds.pt").is_file() else None
        sdem = torch.load(out_dir / "k2_demol_reg_60ep_best_dev_preds.pt", map_location="cpu", weights_only=False) if (out_dir / "k2_demol_reg_60ep_best_dev_preds.pt").is_file() else None
        out = {"bs128_200ep_minus_60ep": paired_bootstrap_meV(s200["pred_eV"], s060["pred_eV"], s060["target_eV"])}
        if sb256 is not None:
            out["bs256_200ep_minus_bs128_200ep"] = paired_bootstrap_meV(sb256["pred_eV"], s200["pred_eV"], s200["target_eV"])
        if sdem is not None:
            out["demol_reg_best_vs_bs128_ep60"] = paired_bootstrap_meV(sdem["pred_eV"], s060["pred_eV"], s060["target_eV"])
            out["cross_200ep_gptrans_plus_demol_reg"] = paired_bootstrap_meV(sdem["pred_eV"], s200["pred_eV"], s200["target_eV"])
        (out_dir / "wave1_kaggle2_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("=== KAGGLE2 200EP+DEMOL ANALYSIS ===\n" + json.dumps(out, indent=2), flush=True)

    elif mode == "wave1_kaggle3":
        # GPU0: FastGPTrans + ETKDG-3D (60ep BS128) | GPU1: Seq 3A FastGPTrans 2D control, Seq 3B FastDeMolCINReg + ETKDG-3D
        p0 = _spawn_gpu_cmd(0, script_path, ["--task", "gptrans_run", "--run-name", "k3_gptrans_3d_60ep", "--input-root", str(input_root), "--output-dir", str(out_dir), "--epochs", "60", "--bs", "128", "--lr", "0.001", "--ema", "0.999", "--aux-ema", "0.9995", "--amp-fp16", "1", "--compile-mode", "auto", "--det", "0", "--load-pos", "1", "--use-3d", "1", "--eval-every", "2", "--snapshots", "60"])
        g1 = 1 if n_gpus >= 2 else 0
        if n_gpus >= 2:
            p1a = _spawn_gpu_cmd(g1, script_path, ["--task", "gptrans_run", "--run-name", "k3_gptrans_2d_ctrl_60ep", "--input-root", str(input_root), "--output-dir", str(out_dir), "--epochs", "60", "--bs", "128", "--lr", "0.001", "--ema", "0.999", "--aux-ema", "0.9995", "--amp-fp16", "1", "--compile-mode", "auto", "--det", "0", "--load-pos", "0", "--use-3d", "0", "--eval-every", "2", "--snapshots", "60"])
            if p1a.wait() != 0:
                raise RuntimeError("k3_gptrans_2d_ctrl_60ep failed")
            p1b = _spawn_gpu_cmd(g1, script_path, ["--task", "demol_run", "--run-name", "k3_demol_3d_60ep", "--input-root", str(input_root), "--output-dir", str(out_dir), "--epochs", "60", "--bs", "128", "--lr", "0.001", "--ema", "0.999", "--dropout", "0.1", "--drop-path", "0.1", "--amp-fp16", "1", "--load-pos", "1", "--use-3d", "1", "--eval-every", "2"])
            if p1b.wait() != 0:
                raise RuntimeError("k3_demol_3d_60ep failed")
        if p0.wait() != 0:
            raise RuntimeError("k3_gptrans_3d_60ep failed")

        g3d = torch.load(out_dir / "k3_gptrans_3d_60ep_best_dev_preds.pt", map_location="cpu", weights_only=False)
        g2d = torch.load(out_dir / "k3_gptrans_2d_ctrl_60ep_best_dev_preds.pt", map_location="cpu", weights_only=False)
        d3d = torch.load(out_dir / "k3_demol_3d_60ep_best_dev_preds.pt", map_location="cpu", weights_only=False)
        ens_3d_3d = 0.5 * (g3d["pred_eV"] + d3d["pred_eV"])
        summary = {
            "gptrans_3d_minus_2d_best": paired_bootstrap_meV(g3d["pred_eV"], g2d["pred_eV"], g2d["target_eV"]),
            "demol_3d_minus_2d_best": paired_bootstrap_meV(d3d["pred_eV"], g2d["pred_eV"], g2d["target_eV"]),
            "gptrans_3d_plus_demol_3d_ensemble": paired_bootstrap_meV(ens_3d_3d, g2d["pred_eV"], g2d["target_eV"]),
            "cross_3d_pair_corr": paired_bootstrap_meV(g3d["pred_eV"], d3d["pred_eV"], g2d["target_eV"]),
        }
        (out_dir / "wave1_3d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("=== KAGGLE3 3D SCREEN SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="wave1_kaggle1")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-dir", default="/kaggle/working")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task", default="gptrans_run")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--aux-ema", type=float, default=None)
    parser.add_argument("--amp-fp16", type=int, default=1)
    parser.add_argument("--compile-mode", default="auto")
    parser.add_argument("--det", type=int, default=0)
    parser.add_argument("--load-pos", type=int, default=0)
    parser.add_argument("--use-3d", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--snapshots", default="60")
    args = parser.parse_args(argv)
    if args.worker:
        worker_entrypoint(args)
    else:
        run_wave1_mode(args.mode, Path(args.input_root), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
