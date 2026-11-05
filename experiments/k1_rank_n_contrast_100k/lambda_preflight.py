"""One-shot lambda rule: lambda*||g_RnC|| / ||g_L1|| = 0.25 (median over 8 fixed batches).

Gradients are taken w.r.t. all non-head parameters of the frozen reference initial state
in train mode (dropout on), from the first 8 consecutive 128-row batches of train shard 0
(rows 0..1023, file order). Both terms share one forward; lambda is rounded to 1 s.f.

Run: .venv\Scripts\python.exe -I experiments/k1_rank_n_contrast_100k/lambda_preflight.py <shard0.pt> <out.json>
"""
import json, math, statistics, sys
from pathlib import Path
import torch
import torch.nn.functional as functional
from torch_geometric.data import Batch
from molgap.data.packed_graph import PackedGraphDataset
from molgap.k1_rank_contrast import rank_contrast_loss
from molgap.qm9_neural_atom import make_encoder
from molgap.v4_runtime import read_frozen_state_artifact
from molgap.training_reproducibility import sha256_file

TARGET = 0.25
MEAN, STD = 5.3383002281188965, 1.275090217590332
INITIAL = Path("platforms/_records/kaggle/initializations/k1_reference_s42.pt")
INITIAL_SHA = "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"


def main(shard, out):
    state, _ = read_frozen_state_artifact(INITIAL, expected_state_sha256=INITIAL_SHA)
    model = make_encoder("neural_atom_k1"); model.load_state_dict(state, strict=True); model.train()
    dataset = PackedGraphDataset(Path(shard), mmap=True)
    encoder = [p for n, p in model.named_parameters() if not n.startswith("head")]
    rows = []
    for index in range(8):
        batch = Batch.from_data_list([dataset[i] for i in range(index * 128, (index + 1) * 128)])
        torch.manual_seed(1000 + index)
        z = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe)
        target = (batch.y.view(-1) - MEAN) / STD
        l1 = functional.l1_loss(model.head(z).view(-1), target)
        rnc = rank_contrast_loss(z, target)
        g1 = torch.autograd.grad(l1, encoder, retain_graph=True)
        g2 = torch.autograd.grad(rnc, encoder)
        norm = lambda g: math.sqrt(sum(float(t.pow(2).sum()) for t in g))
        rows.append({"l1": float(l1), "rnc": float(rnc), "z_norm": float(z.detach().norm(dim=-1).mean()),
                     "g_l1": norm(g1), "g_rnc": norm(g2), "ratio": norm(g2) / norm(g1)})
    median = statistics.median(r["ratio"] for r in rows)
    raw = TARGET / median
    lam = float(f"{raw:.1g}")
    report = {"shard_sha256": sha256_file(Path(shard)), "initial_sha256": INITIAL_SHA, "target_ratio": TARGET,
              "median_ratio_unit_lambda": median, "lambda_raw": raw, "lambda": lam,
              "in_review_range": 0.002 <= lam <= 0.2, "batches": rows}
    Path(out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "batches"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
