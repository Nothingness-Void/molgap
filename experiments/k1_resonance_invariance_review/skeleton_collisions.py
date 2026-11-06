"""Count molecules in fixed-100K train shard 0 that share a bond-order-free skeleton, net charge and total H with another molecule.

Such pairs are the only way the benchmark could contain two inputs of one molecule (resonance forms or tautomers).
Run: .venv\Scripts\python.exe -I experiments/k1_resonance_invariance_review/skeleton_collisions.py <shard0.pt> <out.json>
"""
import collections, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from molgap.data.packed_graph import PackedGraphDataset

d = PackedGraphDataset(Path(sys.argv[1]), mmap=True)
groups = collections.defaultdict(list)
for i in range(len(d)):
    g = d[i]; x = g.x.numpy(); n = len(x)
    labels = [str(int(r[0])) for r in x]
    adj = [[] for _ in range(n)]
    for a, b in g.edge_index.t().tolist(): adj[a].append(b)
    for _ in range(4):   # 4 WL rounds on element labels only (no bond order, no charge, no H)
        labels = [hashlib.md5((labels[v] + "|" + ",".join(sorted(labels[u] for u in adj[v]))).encode()).hexdigest()[:8] for v in range(n)]
    key = (hashlib.md5("".join(sorted(labels)).encode()).hexdigest(), int(sum(CH for CH in (r[3] - 5 for r in x))), int(x[:, 4].sum()), n, g.edge_index.shape[1])
    groups[key].append((i, float(g.y), x.copy()))
multi = [v for v in groups.values() if len(v) > 1]
differing = [v for v in multi if any((a[2].shape != b[2].shape) or (a[2] != b[2]).any() for a in v for b in v)]
res = dict(molecules=len(d), skeleton_groups=len(groups), groups_with_multiple=len(multi), molecules_in_multi=sum(len(v) for v in multi),
           groups_with_different_atom_features=len(differing),
           max_abs_gap_diff_eV_in_groups=[round(max(abs(a[1] - b[1]) for a in v for b in v), 4) for v in differing][:20])
Path(sys.argv[2]).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
