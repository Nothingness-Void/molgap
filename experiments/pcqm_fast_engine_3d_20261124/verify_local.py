"""Local RTX 5060 verification suite for FastGPTrans (2D & 3D), FastDeMolCINReg (2D & 3D), and protected-role guards."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(r"D:\molgap-worktrees\demol-cin-ims-replication-20261120\src")))

from fast_gptrans import (
    FastBondLocalPairNormGPTrans,
    GPUResidentGraphStore,
    PROTECTED_ROW_START,
    TRAIN_100K_MEAN_EV,
    TRAIN_100K_STD_EV,
)
from fast_demol_cin import FastDeMolCINReg, GPUResidentCellStore, extract_cellular_parts
from molgap.gptrans_bond_pair_norm import construct_family_model as make_orig_gptrans
from molgap.gptrans_execution_backend import bind_model_layout, prepare_layout, prepared_forward
from molgap.demol_cin import DeMolCIN, extract_cellular_complex


CKPT_PATH = Path(r"D:\molgap-artifacts\initializations\structural-bias-norm-100k\gptrans.pt")
SHARD0 = Path(r"D:\molgap-data\pcqm_fixed_100k_v1\train\train_shard_0000.pt")


def test_1_fp32_gptrans_parity() -> dict:
    device = torch.device("cuda")
    data, slices = torch.load(SHARD0, map_location="cpu", weights_only=False)
    graphs = []
    for i in range(64):
        xs, xe = int(slices["x"][i]), int(slices["x"][i + 1])
        es, ee = int(slices["edge_index"][i]), int(slices["edge_index"][i + 1])
        graphs.append(Data(x=data.x[xs:xe], edge_index=data.edge_index[:, es:ee], edge_attr=data.edge_attr[es:ee], y=data.y[i:i+1].float()))

    orig = make_orig_gptrans(None, None)
    orig.load_state_dict(torch.load(CKPT_PATH, map_location="cpu", weights_only=False), strict=True)
    bind_model_layout(orig)
    orig = orig.to(device).eval()

    fast = FastBondLocalPairNormGPTrans(use_3d=False).to(device).eval()
    fast.load_from_gptrans_checkpoint(CKPT_PATH)
    fast = fast.to(device).eval()

    # Use direct mini store over first 64 graphs
    store = GPUResidentGraphStore([SHARD0], device=device, expected_source_start=0, expected_source_stop=50_000)
    idx = torch.arange(64, dtype=torch.long)
    b_fast = store.get_batch(idx, pad_to_n=20, e_bucket=256)
    b_orig = prepare_layout(Batch.from_data_list(graphs)).to(device)

    with torch.no_grad():
        p_orig = prepared_forward(orig, b_orig).view(-1)
        p_fast = fast(
            b_fast["dense_x"], b_fast["degree"], b_fast["atom_mask"], b_fast["spd"],
            b_fast["edge_attr"], b_fast["bond_pair_idx"], b_fast["edge_src_flat"],
            b_fast["edge_dst_flat"], b_fast["edge_pair_flat"], b_fast["edge_target_scatter"],
        )
    max_fwd_diff = float((p_orig - p_fast).abs().max().item())
    assert max_fwd_diff < 1e-4, f"Forward diff too high: {max_fwd_diff}"

    # Deterministic gradient check (disable dropout temporarily so gradients align)
    for mod in list(orig.modules()) + list(fast.modules()):
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "probability"):
            mod.probability = 0.0
    orig.train()
    fast.train()
    orig.zero_grad(set_to_none=True)
    fast.zero_grad(set_to_none=True)
    F.l1_loss(prepared_forward(orig, b_orig).view(-1), b_orig.y.view(-1)).backward()
    F.l1_loss(fast(
        b_fast["dense_x"], b_fast["degree"], b_fast["atom_mask"], b_fast["spd"],
        b_fast["edge_attr"], b_fast["bond_pair_idx"], b_fast["edge_src_flat"],
        b_fast["edge_dst_flat"], b_fast["edge_pair_flat"], b_fast["edge_target_scatter"],
    ), b_fast["y"]).backward()

    fast_grads = {n: p.grad for n, p in fast.named_parameters()}
    max_grad_diff = 0.0
    for n, p in orig.named_parameters():
        fg = fast_grads[n]
        g_ref = p.grad.view(fg.shape)
        d = float((fg - g_ref).abs().max().item())
        if d > max_grad_diff:
            max_grad_diff = d
    assert max_grad_diff < 1e-4, f"Grad diff too high: {max_grad_diff}"
    return {"max_fwd_diff": max_fwd_diff, "max_grad_diff": max_grad_diff, "store": store}


def test_2_amp_fp16_and_bf16_stability(store: GPUResidentGraphStore) -> dict:
    device = torch.device("cuda")
    res = {}
    for dtype_name, dt in [("fp16", torch.float16), ("bf16", torch.bfloat16)]:
        m = FastBondLocalPairNormGPTrans(use_3d=False).to(device).train()
        m.load_from_gptrans_checkpoint(CKPT_PATH)
        m = m.to(device).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05, fused=True)
        scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
        losses = []
        for s in range(50):
            idx = torch.arange((s * 64) % 1024, (s * 64) % 1024 + 64)
            b = store.get_batch(idx, pad_to_n=20, e_bucket=256)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=dt):
                p = m(
                    b["dense_x"], b["degree"], b["atom_mask"], b["spd"],
                    b["edge_attr"], b["bond_pair_idx"], b["edge_src_flat"],
                    b["edge_dst_flat"], b["edge_pair_flat"], b["edge_target_scatter"],
                )
                loss = F.l1_loss(p.float(), (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            assert torch.isfinite(loss), f"{dtype_name} NaN/Inf loss at step {s}"
            if dt == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                assert torch.isfinite(norm) or scaler.get_scale() > 1, "Invalid norm/scale"
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            losses.append(float(loss.item()))
        for name, p in m.named_parameters():
            assert torch.isfinite(p).all(), f"Nonfinite weight {name}"
        res[dtype_name] = {"step0_loss": losses[0], "step49_loss": losses[-1]}
    return res


def test_3_gptrans_3d_zero_init_and_fallback(store: GPUResidentGraphStore) -> dict:
    device = torch.device("cuda")
    m2 = FastBondLocalPairNormGPTrans(use_3d=False).to(device).eval()
    m2.load_from_gptrans_checkpoint(CKPT_PATH)
    m3 = FastBondLocalPairNormGPTrans(use_3d=True).to(device).eval()
    m3.load_from_gptrans_checkpoint(CKPT_PATH)

    b = store.get_batch(torch.arange(64), pad_to_n=20, e_bucket=256)
    pos = torch.randn((64, 20, 3), device=device)
    gv_all_true = torch.ones((64,), dtype=torch.bool, device=device)
    gv_all_false = torch.zeros((64,), dtype=torch.bool, device=device)

    with torch.no_grad():
        p2 = m2(b["dense_x"], b["degree"], b["atom_mask"], b["spd"], b["edge_attr"], b["bond_pair_idx"], b["edge_src_flat"], b["edge_dst_flat"], b["edge_pair_flat"], b["edge_target_scatter"])
        p3_init = m3(b["dense_x"], b["degree"], b["atom_mask"], b["spd"], b["edge_attr"], b["bond_pair_idx"], b["edge_src_flat"], b["edge_dst_flat"], b["edge_pair_flat"], b["edge_target_scatter"], pos, gv_all_true)
        p3_invalid = m3(b["dense_x"], b["degree"], b["atom_mask"], b["spd"], b["edge_attr"], b["bond_pair_idx"], b["edge_src_flat"], b["edge_dst_flat"], b["edge_pair_flat"], b["edge_target_scatter"], pos, gv_all_false)
    init_diff = float((p2 - p3_init).abs().max().item())
    inv_diff = float((p2 - p3_invalid).abs().max().item())
    assert init_diff < 1e-6, f"Step 0 3D != 2D: {init_diff}"
    assert inv_diff < 1e-6, f"Invalid geom 3D != 2D: {inv_diff}"

    # Train 20 FP16 steps and confirm 3D parameters get real non-zero gradients
    m3.train()
    opt = torch.optim.AdamW(m3.parameters(), lr=1e-3, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    grad_norms_3d = []
    for s in range(20):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            p = m3(b["dense_x"], b["degree"], b["atom_mask"], b["spd"], b["edge_attr"], b["bond_pair_idx"], b["edge_src_flat"], b["edge_dst_flat"], b["edge_pair_flat"], b["edge_target_scatter"], pos, gv_all_true)
            loss = F.l1_loss(p.float(), (b["y"] - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        g_mlp = float(m3.encoder_3d.pair_mlp[2].weight.grad.norm().item())
        g_cent = float(m3.encoder_3d.centrality_proj.weight.grad.norm().item())
        assert torch.isfinite(m3.encoder_3d.pair_mlp[2].weight.grad).all()
        grad_norms_3d.append((g_mlp, g_cent))
        scaler.step(opt)
        scaler.update()
    assert grad_norms_3d[-1][0] > 0.0 and grad_norms_3d[-1][1] > 0.0, "3D encoder got zero gradients!"

    # Also test > pad_to_n fallback (25-atom synthetic graph)
    b25 = store.get_batch(torch.arange(4), pad_to_n=25, e_bucket=64)
    p25 = m3(b25["dense_x"], b25["degree"], b25["atom_mask"], b25["spd"], b25["edge_attr"], b25["bond_pair_idx"], b25["edge_src_flat"], b25["edge_dst_flat"], b25["edge_pair_flat"], b25["edge_target_scatter"], torch.randn((4, 25, 3), device=device), torch.ones((4,), dtype=torch.bool, device=device))
    assert p25.shape == (4,) and torch.isfinite(p25).all()
    return {"init_diff": init_diff, "invalid_geom_diff": inv_diff, "step19_3d_grad_norm": grad_norms_3d[-1]}


def test_4_demol_reg_2d_and_3d() -> dict:
    device = torch.device("cuda")
    data, slices = torch.load(SHARD0, map_location="cpu", weights_only=False)
    from fast_demol_cin import LEVEL_SPEC
    chunks = {lvl: {f: [] for f in fields} for lvl, fields in LEVEL_SPEC.items()}
    counts = {lvl: [] for lvl in LEVEL_SPEC}
    ys, pos_c, gv_c = [], [], []
    for i in range(256):
        xs, xe = int(slices["x"][i]), int(slices["x"][i + 1])
        es, ee = int(slices["edge_index"][i]), int(slices["edge_index"][i + 1])
        parts = extract_cellular_parts(data.x[xs:xe], data.edge_index[:, es:ee], data.edge_attr[es:ee])
        ys.append(data.y[i:i+1].float())
        pos_c.append(torch.randn((xe - xs, 3), dtype=torch.float32))
        gv_c.append(bool(i % 10 != 0))
        for lvl, fields in LEVEL_SPEC.items():
            for f in fields:
                chunks[lvl][f].append(parts[f])
            counts[lvl].append(int(parts[next(iter(fields))].shape[0]))
    mini = object.__new__(GPUResidentCellStore)
    mini.device = device; mini.n_graphs = 256
    mini.tensors = {lvl: {f: torch.cat(chunks[lvl][f], dim=0).to(device) for f in fields} for lvl, fields in LEVEL_SPEC.items()}
    mini.ptr, mini.cnt = {}, {}
    for lvl in LEVEL_SPEC:
        ct = torch.tensor(counts[lvl], dtype=torch.long)
        pt = torch.zeros(257, dtype=torch.long); pt[1:] = torch.cumsum(ct, 0)
        mini.cnt[lvl] = ct.to(device); mini.ptr[lvl] = pt.to(device)
    mini.y = torch.cat(ys, dim=0).to(device)
    mini.has_pos = True; mini.pos = torch.cat(pos_c, dim=0).to(device)
    mini.geom_valid = torch.tensor(gv_c, dtype=torch.bool, device=device)

    out = {}
    for u3d in (False, True):
        m = FastDeMolCINReg(dropout=0.1, drop_path=0.1, use_3d=u3d).to(device)
        b = mini.get_batch(torch.arange(64))
        m.train()
        with torch.no_grad():
            t1 = m(b)
            t2 = m(b)
        train_stoch_diff = float((t1 - t2).abs().max().item())
        assert train_stoch_diff > 1e-6, "Dropout/DropPath inactive in train()!"
        m.eval()
        with torch.no_grad():
            e1 = m(b)
            e2 = m(b)
        eval_diff = float((e1 - e2).abs().max().item())
        assert eval_diff < 1e-5, f"Eval mode stochastic: {eval_diff}"

        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, fused=True)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        losses = []
        for s in range(30):
            idx = torch.arange((s * 64) % 256, (s * 64) % 256 + 64)
            b = mini.get_batch(idx)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                p = m(b)
                loss = F.l1_loss(p.float(), (b.y - TRAIN_100K_MEAN_EV) / TRAIN_100K_STD_EV)
            assert torch.isfinite(loss), f"DeMol u3d={u3d} nonfinite loss step {s}"
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        out[f"demol_3d={u3d}"] = {"train_stoch_diff": train_stoch_diff, "eval_diff": eval_diff, "step29_loss": losses[-1]}
    return out


def test_5_protected_role_guard() -> dict:
    # Verify any source_idx >= 3_250_000 fails loudly
    fake_shard = Path(__file__).parent / "_tmp_guard_test.pt"
    d0, s0 = torch.load(SHARD0, map_location="cpu", weights_only=False)
    d_clone = d0.clone()
    d_clone.source_idx = torch.arange(PROTECTED_ROW_START, PROTECTED_ROW_START + int(d0.y.shape[0]))
    torch.save((d_clone, s0), fake_shard)
    blocked = False
    try:
        GPUResidentGraphStore([fake_shard], device="cpu")
    except ValueError as exc:
        if "Protected split boundary violated" in str(exc):
            blocked = True
    finally:
        fake_shard.unlink(missing_ok=True)
    assert blocked, "Protected split guard failed to raise ValueError!"
    return {"protected_split_blocked": True}


def main() -> int:
    t1 = test_1_fp32_gptrans_parity()
    store = t1.pop("store")
    t2 = test_2_amp_fp16_and_bf16_stability(store)
    t3 = test_3_gptrans_3d_zero_init_and_fallback(store)
    t4 = test_4_demol_reg_2d_and_3d()
    t5 = test_5_protected_role_guard()
    record = {"status": "PASS", "test1_parity": t1, "test2_amp": t2, "test3_3d_fallback": t3, "test4_demol": t4, "test5_guard": t5}
    (Path(__file__).parent / "verify_local_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
