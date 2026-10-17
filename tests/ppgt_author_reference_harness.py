"""Hash-checked raw-author CPU oracle; no MolGap model/features or trainer.

Selected AST definitions retain their constructor/forward/math verbatim.
Only registration decorators and imports are omitted. Layout/registry shims
replace unavailable integration dependencies, never the attention equations.
The module is test infrastructure and is not a runtime source dependency.
"""
import ast
from copy import deepcopy
from functools import partial
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Union
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric as pyg
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import coalesce, scatter


ROOT = Path(__file__).parent / "vendor/ppgt_reference"


class Config(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def _unsupported(*args, **kwargs):
    raise AssertionError("Outside the pinned PCQM pre-norm CPU oracle scope")


def _rearrange(value, pattern, **axes):
    """Only the six layout operations used in raw self-attention."""
    if pattern in {"b m (h d) -> b h m d", "b n (h d) -> b h n d"}:
        b, n, width = value.shape
        return value.reshape(b, n, axes["h"], width // axes["h"]).permute(0, 2, 1, 3)
    if pattern == "b h (n m) -> b h n m":
        b, h, width = value.shape
        return value.reshape(b, h, axes["n"], width // axes["n"])
    if pattern == "b (h n m) -> b h n m":
        b, width = value.shape
        return value.reshape(b, axes["h"], axes["n"], width // (axes["h"] * axes["n"]))
    if pattern == "b h n d -> b n (h d)":
        b, h, n, d = value.shape
        return value.permute(0, 2, 1, 3).reshape(b, n, h * d)
    if pattern == "b n m h -> b h n m":
        return value.permute(0, 3, 1, 2)
    raise AssertionError(f"Unreviewed author layout: {pattern}")


def _einsum(*args):
    return torch.einsum(args[-1].replace(" ", ""), *args[:-1])


def _complete_graph(index, values, groups, num_nodes, **kwargs):
    # Independent integration shim: preserve all features, add all missing pairs.
    nodes = torch.arange(num_nodes, device=index.device)
    src, dst = torch.meshgrid(nodes, nodes, indexing="ij")
    within = groups[src] == groups[dst]
    complete = torch.stack((src[within], dst[within]))
    padded = values.new_zeros(num_nodes * num_nodes, values.shape[-1])
    padded.index_add_(0, index[0] * num_nodes + index[1], values)
    return complete, padded[complete[0] * num_nodes + complete[1]]


def reference_namespace():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == "62cafd3abbb8d72e9a8b3146f947a192cf0aacaf"
    for entry in manifest["files"]:
        assert hashlib.sha256((ROOT / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    ns = dict(torch=torch, nn=nn, F=F, np=np, pyg=pyg, warnings=warnings,
              math=math, partial=partial, Any=Any, List=List, Optional=Optional,
              Union=Union, Data=Data,
              rearrange=_rearrange, einsum=_einsum, scatter=scatter,
              coalesce=coalesce, to_complete_graph=_complete_graph,
              copy_and_pop=lambda d, keys: {k: deepcopy(v) for k, v in d.items() if k not in keys},
              trunc_normal_=nn.init.trunc_normal_, lecun_normal_=_unsupported,
              drop_path=_unsupported, calculate_gain=nn.init.calculate_gain)
    selected = [
        ("ada_rms_norm.py.txt", {"AdaRMSNorm"}),
        ("residual.py.txt", {"ResidualLayer"}),
        ("normalization.py.txt", {"NormalizationLayer", "Batch2BatchNormalizationLayer"}),
        ("graph_drop_path.py.txt", {"graph_drop_path", "GraphDropPath"}),
        ("initializers.py.txt", {"default_init_", "trunc_init_", "kaiming_uniform_init_"}),
        ("vit_initializer.py.txt", {"init_weights_vit_timm"}),
        ("mlp.py.txt", {"MLP"}),
        ("node_encoder.py.txt", {"MLPNodeEncoder"}),
        ("sin_edge_encoder.py.txt", {"MLPSinEdgeEncoder"}),
        ("edge_ffn.py.txt", {"FFN"}),
        ("attention_and_block.py.txt", {"URPEL2AttentionFast", "PPGTLayer"}),
        ("linear_head.py.txt", {"LinearHead"}),
        ("rrwp.py.txt", {"add_node_attr", "add_full_rrwp", "dense_to_coo"}),
    ]
    for filename, names in selected:
        tree = ast.parse((ROOT / filename).read_bytes(), filename=filename)
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in tree.body} == names, filename
        for node in tree.body:
            node.decorator_list = [d for d in node.decorator_list if not (
                isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                and d.func.id.startswith("register_"))]
        exec(compile(ast.fix_missing_locations(tree), filename, "exec"), ns)
        if filename == "ada_rms_norm.py.txt":
            ns["act_dict"] = {"none": nn.Identity, "gelu": nn.GELU,
                               "ada_rms_norm": ns["AdaRMSNorm"]}
        if filename == "initializers.py.txt":
            ns["kaiming_uniform_linear_init_"] = partial(ns["kaiming_uniform_init_"], a=1)
    return ns


class AuthorOracle(nn.Module):
    """Raw modules assembled in GraphNet's order, with a MolGap OGB front end."""
    def __init__(self, config):
        super().__init__()
        c = config
        if not c.use_phi or c.epsilon != 1e-6:
            raise ValueError("Raw PCQM oracle requires URPE and the source's norm epsilon")
        ns = self.ns = reference_namespace()
        ns["cfg"] = Config(model=Config(graph_pooling=c.pooling),
                           gnn=Config(act="gelu", trunc_init=True))
        ns["register"] = SimpleNamespace(act_dict=ns["act_dict"],
                            pooling_dict={"sum": global_add_pool, "mean": global_mean_pool})
        # Fixed category sizes are the declared MolGap adaptation, independent
        # of candidate imports and without optional OGB package installation.
        self.atoms = nn.ModuleList(nn.Embedding(d, c.hidden) for d in (119, 5, 12, 12, 10, 6, 6, 2, 2))
        self.bonds = nn.ModuleList(nn.Embedding(d, c.pair_dim) for d in (5, 6, 2))
        self.node_pe = ns["MLPNodeEncoder"](c.rrwp_steps + 1, c.hidden,
                                          pe_name="rrwp", kaiming_uniform_init=True)
        self.node_order = ns["MLPNodeEncoder"](1, c.hidden,
                                             pe_name="log_order", kaiming_uniform_init=True)
        self.pair_encoder = ns["MLPSinEdgeEncoder"](c.rrwp_steps + 1, c.pair_dim,
            pe_name="rrwp", add_invN=True, add_raw_feat=True, sin_L=c.spe_bases,
            hid_dim=str(c.pe_mlp_hidden), to_complete_graph=True, kaiming_uniform_init=True)
        self.stem = ns["FFN"](c.pair_dim, c.pair_dim, pe_name="edge",
            pre_backbone_norm="ada_rms_norm", post_backbone_norm="ada_rms_norm",
            pre_mlp_norm="ada_rms_norm", num_blocks=c.stem_blocks,
            ffn_ratio=c.stem_expansion, act="gelu")
        self.pre_norm = ns["Batch2BatchNormalizationLayer"]("ada_rms_norm", c.hidden)
        cfg = Config(dim_edge=c.pair_dim, macro_norm_type="pre_norm", layer_scale=True,
                     alpha=c.layer_scale, ffn_ratio=c.expansion,
                     attn=Config(qkv_bias=True, urpe=True, l2dist=True, clamp=None))
        self.blocks = nn.ModuleList(ns["PPGTLayer"](c.hidden, c.hidden, c.heads,
            norm_fn="ada_rms_norm", act="gelu", attn_dropout=c.attention_dropout,
            drop_path=c.stochastic_depth, cfg=cfg) for _ in range(c.blocks))
        self.post_norm = ns["Batch2BatchNormalizationLayer"]("ada_rms_norm", c.hidden)
        self.head = ns["LinearHead"](c.hidden, 1, L=3)
        self.config = c
        self.apply(ns["init_weights_vit_timm"])

    def prepare(self, batch):
        prepared = []
        for item in batch.to_data_list():
            graph = Data(x=item.x.clone(), edge_index=item.edge_index.clone(),
                         edge_attr=item.edge_attr.clone(), y=torch.zeros(1))
            self.ns["add_full_rrwp"](graph, walk_length=self.config.rrwp_steps,
                                     sym_norm=self.config.sym_norm, add_identity=True)
            prepared.append(graph)
        return Batch.from_data_list(prepared)

    def forward(self, batch):
        # Feature derivation does not use any candidate transform or helper.
        data = self.prepare(batch)
        data.batch_tensor = torch.ones(len(data.ptr) - 1)
        x, bonds = data.x, data.edge_attr
        data.x = sum(embedding(x[:, i]) for i, embedding in enumerate(self.atoms))
        data.edge_attr = sum(embedding(bonds[:, i]) for i, embedding in enumerate(self.bonds))
        data = self.node_pe(data)
        data = self.node_order(data)
        data = self.pair_encoder(data)
        data = self.stem(data)
        data = self.pre_norm(data)
        for block in self.blocks:
            data = block(data)
        data = self.post_norm(data)
        return self.head(data)[0].reshape(-1)


def parameter_map(oracle):
    """Explicit raw-module -> candidate name mapping; no shape-only matching."""
    mapping = {}

    def add(prefix, module):
        for name, parameter in module.named_parameters():
            mapping[f"{prefix}.{name}"] = parameter

    add("atoms", oracle.atoms)
    add("bonds", oracle.bonds)
    add("node_pe", oracle.node_pe.out_fc)
    add("node_order", oracle.node_order.out_fc)
    add("pe_mlp.0", oracle.pair_encoder.mlp[0])
    add("pe_mlp.2", oracle.pair_encoder.out_fc)
    add("stem_pre_norm", oracle.stem.pre_backbone_norm.norm_layer)
    add("stem_post_norm", oracle.stem.post_backbone_norm.norm_layer)
    stems = [m for m in oracle.stem.mlp.children() if isinstance(m, oracle.ns["MLP"])]
    assert len(stems) == oracle.config.stem_blocks
    for i, block in enumerate(stems):
        add(f"stem.{i}.norm", block.pre_mlp_norm.norm_layer)
        add(f"stem.{i}.mlp", block.mlp)
    add("pre_backbone_norm", oracle.pre_norm.norm_layer.norm_layer)
    add("post_backbone_norm", oracle.post_norm.norm_layer.norm_layer)
    for i, block in enumerate(oracle.blocks):
        add(f"blocks.{i}.norm1_h", block.norm1_h.norm_layer)
        add(f"blocks.{i}.norm2_h", block.norm2_h.norm_layer)
        add(f"blocks.{i}.attention", block.attention)
        add(f"blocks.{i}.FFN", block.FFN)
        mapping[f"blocks.{i}.attention_scale"] = block.res_layer1.alpha
        mapping[f"blocks.{i}.ffn_scale"] = block.res_layer2.alpha
    add("head", oracle.head.mlp)
    assert {id(v) for v in mapping.values()} == {id(v) for v in oracle.parameters()}
    return mapping


def copy_weights(candidate, oracle):
    mapped = parameter_map(oracle)
    parameters = dict(candidate.named_parameters())
    assert parameters.keys() == mapped.keys(), (parameters.keys() ^ mapped.keys())
    with torch.no_grad():
        for name, parameter in parameters.items():
            assert parameter.shape == mapped[name].shape, name
            mapped[name].copy_(parameter)
    return mapped
