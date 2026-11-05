"""CPU qualification of the K1 RnC term on synthetic 128-graph batches (no real shards).

Run: .venv\Scripts\python.exe experiments/k1_rank_n_contrast_100k/qualify_cpu.py
"""
import copy, json, time, tracemalloc
import torch
from torch_geometric.data import Batch, Data
from molgap.family_model_ops import k1_optimizer_step
from molgap.k1_rank_contrast import LAMBDA, TEMPERATURE, rank_contrast_loss
from molgap.qm9_neural_atom import make_encoder


def rnc_naive(z, y, tau):
    z = torch.nn.functional.normalize(z, dim=-1); n = len(y); total = 0.
    for i in range(n):
        for j in range(n):
            if i == j: continue
            members = [k for k in range(n) if k != i and abs(y[i] - y[k]) >= abs(y[i] - y[j])]
            logits = torch.stack([-(z[i] - z[k]).norm() / tau for k in members])
            total = total + torch.logsumexp(logits, 0) + (z[i] - z[j]).norm() / tau
    return total / (n * (n - 1))


def graphs(count, seed):
    g = torch.Generator().manual_seed(seed); data = []
    for i in range(count):
        n = int(torch.randint(8, 25, (1,), generator=g))
        src = torch.arange(n - 1); edge = torch.cat([torch.stack([src, src + 1]), torch.stack([src + 1, src])], 1)
        data.append(Data(x=torch.randint(0, 2, (n, 9), generator=g), edge_index=edge,
            edge_attr=torch.randint(0, 2, (edge.shape[1], 3), generator=g),
            random_walk_pe=torch.rand(n, 16, generator=g), y=torch.randn(1, generator=g) + 5.3,
            source_idx=torch.tensor([seed * 1000 + i])))
    return Batch.from_data_list(data)


def main():
    report = {"lambda": LAMBDA, "tau": TEMPERATURE}
    torch.manual_seed(0)
    # 1. vectorised == naive loop, ties, float64, gradient finite
    z = torch.randn(16, 8, dtype=torch.float64, requires_grad=True); y = torch.randn(16, dtype=torch.float64)
    y[3] = y[5]; y[7] = y[2]
    fast, slow = rank_contrast_loss(z, y), rnc_naive(z, y, TEMPERATURE)
    grad, = torch.autograd.grad(fast, z)
    report["loop_match"] = abs(float(fast) - float(slow)) < 1e-9 and bool(torch.isfinite(grad).all())
    # 2. duplicate embeddings and identical labels stay finite
    z2 = torch.zeros(6, 4, requires_grad=True); z2.data[:3] = 1.; y2 = torch.zeros(6)
    g2, = torch.autograd.grad(rank_contrast_loss(z2, y2), z2)
    report["degenerate_finite"] = bool(torch.isfinite(g2).all())
    # 3. molecule isolation: eval embedding of molecule 0 independent of the other 127
    model = make_encoder("neural_atom_k1").eval()
    a, b = graphs(128, 1), graphs(128, 2)
    mix = Batch.from_data_list([a[0]] + [b[i] for i in range(1, 128)])
    def embed(batch): return model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe)
    with torch.no_grad():
        report["isolation_eval_bit_equal"] = bool(torch.equal(embed(a)[0], embed(mix)[0]))
    # 4. step: reference vs rnc from the same init, same batch, same RNG
    batch = graphs(128, 3); mean, std = 5.3383, 1.2751
    out = {}
    for mode in ("reference", "rnc"):
        m = make_encoder("neural_atom_k1"); m.load_state_dict(copy.deepcopy(model.state_dict())); m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=1e-5)
        torch.manual_seed(7); tracemalloc.start(); t = time.perf_counter()
        for _ in range(3):
            loss, absolute, rows = k1_optimizer_step(m, opt, batch, mean, std, mode, clip_norm=1.0, check_finite=True)
        out[mode] = {"loss": float(loss), "abs": float(absolute), "sec_per_step": (time.perf_counter() - t) / 3,
                     "state": {k: v.clone() for k, v in m.state_dict().items()}, "n_params": sum(p.numel() for p in m.parameters())}
        tracemalloc.stop()
    r, c = out["reference"], out["rnc"]
    report["params_equal"] = r["n_params"] == c["n_params"] == 3658817
    report["state_keys_equal"] = r["state"].keys() == c["state"].keys()
    report["rnc_changes_weights"] = any(not torch.equal(r["state"][k], c["state"][k]) for k in r["state"])
    report["step_time_ratio_cpu"] = c["sec_per_step"] / r["sec_per_step"]
    report["finite"] = all(map(lambda v: v == v, (r["loss"], c["loss"])))
    # 5. step-0 prediction/L1 term identical, RnC adds gradient to the encoder only via embedding
    m = make_encoder("neural_atom_k1"); m.load_state_dict(model.state_dict()); m.train()
    torch.manual_seed(7); z = embed_t = m.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe)
    target = (batch.y.view(-1) - mean) / std
    rn = rank_contrast_loss(z, target); rn.backward()
    report["head_grad_from_rnc"] = max(float(p.grad.abs().max()) if p.grad is not None else 0. for p in m.head.parameters())
    report["encoder_grad_from_rnc"] = float(sum(p.grad.norm() ** 2 for n_, p in m.named_parameters()
        if p.grad is not None and not n_.startswith("head")) ** .5)
    report["rnc_value_init"] = float(rn); report["embedding_norm_mean"] = float(z.detach().norm(dim=-1).mean())
    # 6. loss-only cost at N=128
    z = torch.randn(128, 192, requires_grad=True); y = torch.randn(128)
    rank_contrast_loss(z, y).backward(); t = time.perf_counter()
    tracemalloc.start()
    for _ in range(10): z.grad = None; rank_contrast_loss(z, y).backward()
    report["loss_ms_per_step_cpu"] = (time.perf_counter() - t) * 100
    report["loss_peak_bytes_python_heap"] = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
