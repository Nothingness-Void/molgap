"""Local smoke: (a) path reconstruction vs brute force, (b) zero-start equality,
(c) fwd+bwd speed ratio, (d) two passes over the two local 100K train shards."""
import sys, time
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from molgap.gptrans_bond_pair_norm import construct_family_model as reference
from molgap.gptrans_multihop_edge import construct_family_model as candidate, HOPS
from molgap.gptrans_execution_backend import bind_model_layout, prepare_layout, prepared_forward

ROOT = "D:/molgap-data/pcqm_fixed_100k_v1/train/train_shard_000%d.pt"


def shard(index, count=None):
    data, slices = torch.load(ROOT % index, weights_only=False)
    ds = InMemoryDataset()
    ds._data, ds.slices = data, slices
    return ds if count is None else ds[:count]


def pair(seed=42):
    torch.manual_seed(seed)
    ref = reference(None, None)
    cand = candidate(None, None)
    cand.load_state_dict({**ref.state_dict(), "path_mix": cand.path_mix.data}, strict=True)
    return ref, cand


@torch.no_grad()
def brute(batch, model):
    """Independent per-pair python walk; also checks every walk is a shortest path."""
    graph, src, dst = model._local_edges(batch.edge_index, batch.batch, len(batch.x))
    sizes = torch.bincount(batch.batch).tolist()
    embed = model.bond_encoder.bond_embedding_list
    edge = sum(e(batch.edge_attr[:, i].long()) for i, e in enumerate(embed)) / 3
    adj = [dict() for _ in sizes]
    for g, s, t, e in zip(graph.tolist(), src.tolist(), dst.tolist(), range(len(src))):
        adj[g][(s, t)] = e
    out = torch.zeros(len(sizes), max(sizes), max(sizes), 32)
    for g, n in enumerate(sizes):
        near = {i: sorted(t for (s, t) in adj[g] if s == i) for i in range(n)}
        for j in range(n):
            dist, frontier = {j: 0}, [j]
            while frontier:
                frontier = [m for u in frontier for m in near[u] if m not in dist and not dist.update({m: dist[u] + 1})]
            for i in range(n):
                d = dist.get(i, 99)
                if i == j or d > HOPS:
                    continue
                cur, acc = i, 0
                for k in range(d):
                    nxt = min(m for m in near[cur] if dist.get(m, 99) == dist[cur] - 1)
                    acc = acc + edge[adj[g][(cur, nxt)]] @ model.path_mix[k]
                    cur = nxt
                out[g, i, j] = acc / d
    return out


def main():
    cpu = list(DataLoader(shard(0, 200), batch_size=50))
    ref, cand = pair()
    # (a) random mixing matrices so the comparison is not trivially zero
    cand.path_mix.data.normal_(0, 0.3, generator=torch.Generator().manual_seed(1))
    worst, walks = 0.0, 0
    for batch in cpu:
        _, _, padding = cand._dense_inputs(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        edges = cand._local_edges(batch.edge_index, batch.batch, len(batch.x))
        got = cand._multihop(batch.edge_attr, edges, padding).detach()
        want = brute(batch, cand)
        worst = max(worst, float((got[:, : want.shape[1], : want.shape[2]] - want).abs().max()))
        walks += int((want.abs().sum(-1) > 0).sum())
    print(f"(a) molecules=200 nonzero_pairs={walks} max_abs_diff={worst:.3e}")
    # (b) zero start: candidate output equals reference bit for bit, train and eval
    ref, cand = pair()
    for mode in (True, False):
        ref.train(mode), cand.train(mode)
        same = True
        for batch in cpu:
            args = (batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            torch.manual_seed(7); a = ref(*args)
            torch.manual_seed(7); b = cand(*args)
            same &= torch.equal(a, b)
        print(f"(b) train={mode} bitwise_equal={same}")
    # (c) speed on GPU, same prepared batches
    dev = "cuda"
    gpu = [prepare_layout(b) for b in DataLoader(shard(0, 128 * 12), batch_size=128)]
    timings = {}
    for name, builder in (("reference", reference), ("candidate", candidate)):
        torch.manual_seed(42)
        m = builder(None, None).to(dev).train(); bind_model_layout(m)
        for round_ in range(2):
            for i, b in enumerate(gpu):
                if i == 2:
                    torch.cuda.synchronize(); start = time.perf_counter()
                m.zero_grad(); prepared_forward(m, b.to(dev)).sum().backward()
            torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - start) / (len(gpu) - 2)
        timings[name + "_mem"] = torch.cuda.max_memory_allocated() / 2**30
        del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print(f"(c) step_s ref={timings['reference']:.4f} cand={timings['candidate']:.4f} "
          f"ratio={timings['candidate'] / timings['reference']:.3f} peakGiB ref={timings['reference_mem']:.2f} cand={timings['candidate_mem']:.2f}")
    # (d) two passes over the two local train shards (100K rows), candidate only
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    train = torch.utils.data.ConcatDataset([shard(0, 50000), shard(1, 50000)])
    mean, std = 5.338300266634226, 1.2750902063230574
    torch.manual_seed(42)
    m = candidate(None, None).to(dev).train(); bind_model_layout(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    for epoch in range(2):
        losses, begun = [], time.perf_counter()
        for b in DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True):
            b = prepare_layout(b).to(dev)
            loss = (prepared_forward(m, b) - (b.y.view(-1) - mean) / std).abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            losses.append(loss.detach())
        losses = torch.stack(losses).cpu()
        print(f"(d) pass={epoch + 1} first100={losses[:100].mean():.4f} last100={losses[-100:].mean():.4f} "
              f"finite={bool(torch.isfinite(losses).all())} sec={time.perf_counter() - begun:.0f} "
              f"|path_mix|={float(m.path_mix.abs().mean()):.2e}")


if __name__ == "__main__":
    main()
