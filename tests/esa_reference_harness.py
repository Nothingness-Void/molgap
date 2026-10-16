"""Isolated execution of pinned MIT author modules, never the Lightning trainer.

Raw source is hash checked. AST changes remove imports only; device literals in
the mask helpers are changed from cuda:0 to cpu. xformers attention and flash
SwiGLU are replaced by their explicit CPU math, not by MolGap implementations.
Only the pre-LN, FP32, no-padding path is certified by this harness.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace, MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import unbatch_edge_index, to_dense_batch


ROOT = Path(__file__).parent / "vendor/esa_reference"
HASHES = {
    "norm_layers.py": "f37cd17d646fea002293a4f62493755330c85cc2b9fdbfae029fe89bc591b65f",
    "estimator_forward.py.txt": "4dc5ac693d33988c424f15c0f74a30762de391206c0d019c5832d06cae635217",
    "mha.py": "ad3527f1b49a490903f75fecfc1fc92f49bc4bb35150ee7e71db6251dc2a99b4",
    "masked_layers.py": "8c7fcfa2e6bc1686778b45009170dd280a112282bffdeb1ccb49ba80ba351d9c",
    "mlp_utils.py": "f37428e6cbd3f77de3bbe94191ec291ecd80a1a4a571db517f127484aff62f88",
}


def cpu_attention(q, k, v, attn_bias=None, p=0.):
    # Deliberately explicit SDP math independent of the candidate's SDPA call.
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    logits = q @ k.transpose(-1, -2) / q.shape[-1] ** .5
    if attn_bias is not None:
        logits = logits + attn_bias
    weights = F.dropout(logits.softmax(-1), p=p, training=p > 0)
    return (weights @ v).transpose(1, 2)


class CPUDevice(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant("cpu"), node) if node.value == "cuda:0" else node


def reference_namespace():
    def unsupported(*args, **kwargs):
        raise AssertionError("Only author pre-LN reference is authorized")
    ns = dict(torch=torch, nn=nn, F=F, unbatch_edge_index=unbatch_edge_index,
              to_dense_batch=to_dense_batch, xavier_normal_=nn.init.xavier_normal_,
              memory_efficient_attention=cpu_attention,
              swiglu=lambda gate, value: F.silu(gate) * value,
              admin_torch=SimpleNamespace(as_module=unsupported), BatchRenorm1d=unsupported)
    for name in ("norm_layers.py", "mlp_utils.py", "mha.py", "masked_layers.py"):
        raw = (ROOT / name).read_bytes()
        if name in HASHES:
            assert hashlib.sha256(raw).hexdigest() == HASHES[name]
        tree = ast.parse(raw, filename=name)
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        tree = CPUDevice().visit(tree)
        exec(compile(ast.fix_missing_locations(tree), name, "exec"), ns)
    return ns


def author_estimator(config, candidate, edge_count):
    """Author constructor components + AST-identical Estimator.forward slice."""
    ns = reference_namespace()
    core = ns["ESA"](num_outputs=config.seeds, dim_output=config.graph_dim,
        dim_hidden=[config.hidden] * len(config.layer_types),
        num_heads=[config.heads] * len(config.layer_types), layer_types=list(config.layer_types),
        norm_type="LN", pre_or_post="pre", use_mlps=True, mlp_hidden_size=config.mlp_hidden,
        num_mlp_layers=2, mlp_type="gated_mlp", set_max_items=edge_count,
        use_bfloat16=False, xformers_or_torch_attn="xformers")
    for local, upstream in zip(candidate.encoder, core.encoder):
        copy_block(local, upstream, pool=False)
    copy_block(candidate.pool, core.decoder[0], pool=True)
    core.decoder_linear.load_state_dict(candidate.graph_projection.state_dict())
    token = ns["SmallMLP"](361, 128, config.hidden, num_layers=2)
    token.mlp.load_state_dict(candidate.token_mlp.state_dict())
    head = nn.Linear(config.graph_dim, 1)
    head.load_state_dict(candidate.head.state_dict())
    obj = SimpleNamespace(lap_encoder=None, rwse_encoder=None, apply_attention_on="edge",
        edge_dim=13, node_edge_mlp=token, st_fast=core, output_mlp=head)
    tree = ast.parse((ROOT / "estimator_forward.py.txt").read_bytes())
    assert hashlib.sha256((ROOT / "estimator_forward.py.txt").read_bytes()).hexdigest() == HASHES["estimator_forward.py.txt"]
    exec(compile(tree, "author.Estimator.forward", "exec"), ns)
    obj.forward = MethodType(ns["forward"], obj)
    return obj, ns


def copy_block(local, upstream, *, pool):
    upstream.norm.ln.load_state_dict(local.norm.state_dict())
    upstream.norm_mlp.ln.load_state_dict(local.norm_mlp.state_dict())
    upstream.mlp.mlp.load_state_dict(local.mlp.state_dict())
    attention = upstream.pma if pool else upstream.sab
    attention.mab.load_state_dict(local.attention.state_dict())
    if pool:
        with torch.no_grad(): attention.S.copy_(local.seeds)
