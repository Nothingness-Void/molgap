"""Smoke: (a) paths vs brute force, (b) zero-start equality, (c) speed ratio, (d) 2 passes on 2 shards."""
import time
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from molgap.gptrans_bond_pair_norm import construct_family_model as reference
from molgap.gptrans_multihop_edge import construct_family_model as candidate, HOPS
from molgap.gptrans_execution_backend import bind_model_layout, prepare_layout, prepared_forward


def shard(i, n):
    data, slices = torch.load(f"D:/molgap-data/pcqm_fixed_100k_v1/train/train_shard_000{i}.pt", weights_only=False)
    ds = InMemoryDataset()
    ds._data, ds.slices = data, slices
    return ds[:n]


@torch.no_grad()
def brute(b, m):
    graph, src, dst = m._local_edges(b.edge_index, b.batch, len(b.x))
    edge = sum(e(b.edge_attr[:, i].long()) for i, e in enumerate(m.bond_encoder.bond_embedding_list)) / 3
    sizes = torch.bincount(b.batch).tolist()
    ids = {(g, s, t): k for k, (g, s, t) in enumerate(zip(graph.tolist(), src.tolist(), dst.tolist()))}
    out = torch.zeros(len(sizes), max(sizes), max(sizes), 32)
    for g, n in enumerate(sizes):
        near = {i: sorted(t for (h, s, t) in ids if h == g and s == i) for i in range(n)}
        for j in range(n):
            dist, frontier = {j: 0}, [j]
            while frontier:
                frontier = [v for u in frontier for v in near[u] if v not in dist and not dist.update({v: dist[u] + 1})]
            for i in range(n):
                d = dist.get(i, 99)
                if i != j and d <= HOPS:
                    cur, acc = i, 0
                    for k in range(d):
                        nxt = min(v for v in near[cur] if dist.get(v, 99) == dist[cur] - 1)
                        acc = acc + edge[ids[g, cur, nxt]] @ m.path_mix[k]
                        cur = nxt
                    out[g, i, j] = acc / d
    return out


def main():
    torch.manual_seed(42)
    ref, cand = reference(None, None), candidate(None, None)
    cand.load_state_dict({**ref.state_dict(), "path_mix": cand.path_mix.data})
    cpu = list(DataLoader(shard(0, 200), batch_size=50))
    for mode in (True, False):  # (b) before path_mix is randomized
        ref.train(mode), cand.train(mode)
        same = True
        for b in cpu:
            args = (b.x, b.edge_index, b.edge_attr, b.batch)
            torch.manual_seed(7); a = ref(*args)
            torch.manual_seed(7); c = cand(*args)
            same &= torch.equal(a, c)
        print(f"(b) train={mode} bitwise_equal={same}")
    cand.path_mix.data.normal_(0, 0.3, generator=torch.Generator().manual_seed(1))
    worst = 0
    for b in cpu:  # (a)
        pad = cand._dense_inputs(b.x, b.edge_index, b.edge_attr, b.batch)[2]
        got = cand._multihop(b.edge_attr, cand._local_edges(b.edge_index, b.batch, len(b.x)), pad).detach()
        want = brute(b, cand)
        worst = max(worst, float((got[:, :want.shape[1], :want.shape[2]] - want).abs().max()))
    print(f"(a) molecules=200 max_abs_diff={worst:.3e} paths_nonzero={bool(want.abs().sum() > 0)}")
    gpu, step = [prepare_layout(b) for b in DataLoader(shard(0, 128 * 12), batch_size=128)], {}
    for name, build in (("reference", reference), ("candidate", candidate)):  # (c)
        torch.manual_seed(42)
        m = build(None, None).cuda().train(); bind_model_layout(m)
        for i, b in enumerate(gpu * 2):
            if i == 14: torch.cuda.synchronize(); t0 = time.perf_counter()
            m.zero_grad(); prepared_forward(m, b.to("cuda")).sum().backward()
        torch.cuda.synchronize()
        step[name] = (time.perf_counter() - t0) / 10
        del m; torch.cuda.empty_cache()
    print(f"(c) step_s ref={step['reference']:.4f} cand={step['candidate']:.4f} ratio={step['candidate'] / step['reference']:.3f}")
    train = torch.utils.data.ConcatDataset([shard(0, 50000), shard(1, 50000)])  # (d)
    torch.manual_seed(42)
    m = candidate(None, None).cuda().train(); bind_model_layout(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    for epoch in range(2):
        losses, t0 = [], time.perf_counter()
        for b in DataLoader(train, batch_size=128, shuffle=True, drop_last=True):
            b = prepare_layout(b).to("cuda")
            loss = (prepared_forward(m, b) - (b.y.view(-1) - 5.3383) / 1.2751).abs().mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            losses.append(loss.detach())
        losses = torch.stack(losses).cpu()
        print(f"(d) pass={epoch + 1} first100={losses[:100].mean():.4f} last100={losses[-100:].mean():.4f} "
              f"finite={bool(torch.isfinite(losses).all())} sec={time.perf_counter() - t0:.0f} |path_mix|={float(m.path_mix.abs().mean()):.2e}")


if __name__ == "__main__":
    main()
