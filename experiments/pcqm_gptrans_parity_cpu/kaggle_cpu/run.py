"""Pinned, no-data GPTrans parity checks for Kaggle2's CPU runtime."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace


AUTHOR = "c0be01f938570b6cb58e27f6e290bb4a78d1b5a4"
LOCAL = "90251ac75d5bed4e53f9ad9c897da0d8563f5d9f"
AUTHOR_ROOT = f"https://raw.githubusercontent.com/czczup/GPTrans/{AUTHOR}/"
LOCAL_ROOT = f"https://raw.githubusercontent.com/Nothingness-Void/molgap/{LOCAL}/"
SOURCES = {
    "author_model": (AUTHOR_ROOT + "models/gptrans.py", "2db430c07ce1cabc5c306d2149e1de9eaa4ec7427b9f21b5eca7f851a1b846b3"),
    "author_wrapper": (AUTHOR_ROOT + "dataset/wrapper.py", "5bbca0358c4095757a2d79d90cd618be19eefefeb66cb55ef0710e38a8e09a9c"),
    "author_algos": (AUTHOR_ROOT + "dataset/algos.pyx", "167d7412935110f870df43c85190aa3e8b06f70dc483bb5df5b7594bc2cb20fc"),
    "author_collator": (AUTHOR_ROOT + "dataset/collator.py", "c37e44b460a215ad7c787bcb5aecb2b878e15359337ad151282f4a846a4666e6"),
    "author_optimizer": (AUTHOR_ROOT + "optimizer.py", "8a5d7f0bafcc1e8af51a959ae1807e0e4b930c42da4db7022339ee60c3a43341"),
    "author_train": (AUTHOR_ROOT + "main.py", "a45eeec4b2465e0126ab820c915326d1f868b971255e9bfa5d33036928fa9b50"),
    "author_config": (AUTHOR_ROOT + "config.py", "00233a80e2af151fb7247fc644c21721287ce36b7101098f3cce34f33795dd3b"),
    "author_tiny_config": (AUTHOR_ROOT + "configs/pcqm4mv2/gptrans_tiny_pcqm4mv2.yaml", "cdbbd942f444ab9b6c538785179b5ca7fc4da21c0c009ce06dbe7626d7296f73"),
    "local_model": (LOCAL_ROOT + "src/molgap/gptrans.py", "76020400020a475c53cf3628a820b77bf49dc3b8df7dff9ad7506e594126778e"),
    "local_runner": (LOCAL_ROOT + "src/molgap/pcqm_gptrans_full_runner.py", "7f5cada76fc023c3d6f6458ee2e75ec396fab1587bc1c7858357c983679c0a8a"),
}
TEST_NAMES = (
    "source_identity",
    "static_contract",
    "training_contract_static",
    "synthetic_path_semantics",
    "official_cython_path",
    "local_shortest_path",
    "official_collator_padding",
    "propagation_forward",
    "initialization_scale",
    "optimizer_groups",
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_bytes() -> dict[str, bytes]:
    output = {}
    for name, (url, expected) in SOURCES.items():
        request = urllib.request.Request(url, headers={"User-Agent": "molgap-parity-cpu/1"})
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = response.read(150_000)
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ValueError(f"{name}: pinned SHA-256 mismatch: {actual}")
        output[name] = payload
    return output


def source_text(payloads: dict[str, bytes]) -> dict[str, str]:
    return {name: value.decode("utf-8") for name, value in payloads.items()}


def extract(source: str, names: tuple[str, ...], namespace: dict, label: str) -> dict:
    tree = ast.parse(source, filename=label)
    selected = []
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in names:
            item = copy.deepcopy(item)
            # TorchScript requires a physical source file; the extracted Python
            # function has the same arithmetic but is not JIT-compiled.
            item.decorator_list = []
            selected.append(item)
    missing = set(names) - {item.name for item in selected}
    if missing:
        raise ValueError(f"{label}: missing pinned definitions {sorted(missing)}")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, label, "exec"), namespace)
    return namespace


def static_contract(sources: dict[str, str]) -> dict:
    local = sources["local_model"]
    author = sources["author_model"]
    wrapper = sources["author_wrapper"]
    runner = sources["local_runner"]
    optimizer = sources["author_optimizer"]
    collator = sources["author_collator"]
    checks = {
        "author_path_input": "self.edge_encoder(edge_input).mean(-2)" in author,
        "author_path_construction": "algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())" in wrapper,
        "local_spd_input": "self.spatial_encoder(spatial)" in local,
        "local_direct_bond_only": "pair[edge_batch, :, edge_src, edge_dst] += bond" in local,
        "author_zeroes_collator_bias": "attn_bias = torch.zeros_like(attn_bias)" in author,
        "author_embedding_std_002": "module.weight.data.normal_(mean=0.0, std=0.02)" in author,
        "local_default_degree_embedding": "self.in_degree_encoder = nn.Embedding(512, node_channels)" in local,
        "author_exempts_1d_and_bias_from_decay": "len(param.shape) == 1 or name.endswith(\".bias\")" in optimizer,
        "local_all_parameter_decay": "model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY" in runner,
        "local_normalized_gap_l1": '"loss": "normalized-gap-l1"' in runner,
        "author_collator_spd_plus_one": "def pad_spatial_pos_unsqueeze(x, padlen):\n    x = x + 1" in collator,
        "author_path_zero_sentinel": "if k == 0:" in sources["author_algos"],
    }
    if not all(checks.values()):
        raise AssertionError(f"Static contract failed: {[k for k,v in checks.items() if not v]}")
    return {"checks": checks, "scope": "source-level facts, not MAE effects"}


def training_contract_static(sources: dict[str, str]) -> dict:
    author_train = sources["author_train"]
    author_config = sources["author_config"]
    tiny = sources["author_tiny_config"]
    local = sources["local_runner"]
    checks = {
        "author_l1_despite_mse_label": "if config.TRAIN.CRITERION == 'mse':\n        criterion = torch.nn.L1Loss()" in author_train,
        "author_raw_target": "targets = data['y'].cuda(non_blocking=True)" in author_train,
        "author_lr_world_size_scaling": "config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0" in author_train,
        "author_default_criterion_label": "_C.TRAIN.CRITERION = 'mse'" in author_config,
        "author_tiny_pcqm4mv2": "DATASET: pcqm4mv2" in tiny and "NUM_LAYERS: 12" in tiny and "NODE_DIM: 256" in tiny and "EDGE_DIM: 32" in tiny,
        "author_tiny_ema": "DECAY: 0.9999" in tiny,
        "local_normalized_target": "target = (batch.y.view(-1).float() - mean) / std" in local,
        "local_l1": "loss = functional.l1_loss(prediction, target)" in local,
        "local_final_step_selection": '"selection": "fixed-final-step-ema-no-development-or-official-validation"' in local,
        "local_clip": "GRADIENT_CLIP = 1.0" in local,
    }
    if not all(checks.values()):
        raise AssertionError(f"Training source contract drifted: {[key for key, ok in checks.items() if not ok]}")
    return {
        "checks": checks,
        "author_target_scale": "raw label, with author L1 criterion",
        "local_target_scale": "train-set-standardized label, with local L1 criterion",
        "caveat": "This is a source-contract contrast; no optimizer step, validation access or MAE causal effect is tested.",
    }


def symbolic_path(edges: list[tuple[int, int, str]], pair: tuple[int, int]) -> dict:
    n = 3
    adj = [[0] * n for _ in range(n)]
    bond = [[None] * n for _ in range(n)]
    for a, b, kind in edges:
        adj[a][b] = adj[b][a] = 1
        bond[a][b] = bond[b][a] = kind
    distance = [[0 if i == j else 1 if adj[i][j] else 510 for j in range(n)] for i in range(n)]
    middle = [[0] * n for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
                    middle[i][j] = k
    def inner(i: int, j: int) -> list[int]:
        k = middle[i][j]
        return [] if k == 0 else inner(i, k) + [k] + inner(k, j)
    i, j = pair
    path = [i] + inner(i, j) + [j]
    return {
        "distance": distance[i][j],
        "local_initial_key": [distance[i][j], bond[i][j]],
        "author_like_path": path,
        "author_like_bonds": [bond[a][b] for a, b in zip(path, path[1:])],
    }


def synthetic_path_semantics() -> dict:
    a = symbolic_path([(0, 1, "single"), (1, 2, "single")], (0, 2))
    b = symbolic_path([(0, 1, "double"), (1, 2, "single")], (0, 2))
    via_zero = symbolic_path([(1, 0, "single"), (0, 2, "single")], (1, 2))
    if a["local_initial_key"] != b["local_initial_key"]:
        raise AssertionError("Local initial pair changed for a non-direct bond")
    if a["author_like_bonds"] == b["author_like_bonds"]:
        raise AssertionError("Author-like path sequence did not reflect bond change")
    return {
        "single_single": a, "double_single": b, "via_atom_zero": via_zero,
        "caveat": "stdlib translation of pinned Cython, not its compiled execution",
    }


def author_algos(payload: bytes, numpy):
    import importlib
    import pyximport
    work = tempfile.TemporaryDirectory(prefix="molgap_author_algos_")
    path = Path(work.name)
    (path / "author_algos.pyx").write_bytes(payload)
    pyximport.install(setup_args={"include_dirs": numpy.get_include()},
                      language_level=3, build_dir=str(path / "build"))
    sys.path.insert(0, str(path))
    try:
        module = importlib.import_module("author_algos")
    except Exception:
        sys.path.remove(str(path))
        work.cleanup()
        raise
    return module, work, path


def small_item(torch, edges: list[tuple[int, int, int]], idx: int):
    directed = [(a, b, t) for a, b, t in edges] + [(b, a, t) for a, b, t in edges]
    index = torch.tensor([[a for a, _, _ in directed], [b for _, b, _ in directed]], dtype=torch.long)
    attr = torch.tensor([[t, 0, 0] for _, _, t in directed], dtype=torch.long)
    x = torch.tensor([[5, 0, 2, 5, 1, 0, 2, 0, 0]] * 3, dtype=torch.long)
    return SimpleNamespace(x=x, edge_index=index, edge_attr=attr,
                           idx=idx, y=torch.tensor([0.0]))


def official_path_check(sources: dict[str, str], payloads: dict[str, bytes], torch, np):
    algos, work, path = author_algos(payloads["author_algos"], np)
    try:
        ns = {"torch": torch, "np": np, "algos": algos}
        extract(sources["author_wrapper"], ("convert_to_single_emb", "preprocess_item"), ns,
                "<pinned-author-wrapper>")
        first = ns["preprocess_item"](small_item(torch, [(0, 1, 0), (1, 2, 0)], 0))
        changed = ns["preprocess_item"](small_item(torch, [(0, 1, 1), (1, 2, 0)], 1))
        zero = ns["preprocess_item"](small_item(torch, [(1, 0, 0), (0, 2, 0)], 2))
        a = first.edge_input[0, 2, :, :].tolist()
        b = changed.edge_input[0, 2, :, :].tolist()
        if int(first.spatial_pos[0, 2]) != 2 or int(changed.spatial_pos[0, 2]) != 2:
            raise AssertionError("Expected distance-two chain")
        if a == b:
            raise AssertionError("Actual author Cython path input ignored changed bond")
        zero_path = zero.edge_input[1, 2, 0, :].tolist()
        missing_direct_edge = zero.attn_edge_type[1, 2, :].tolist()
        result = {
            "distance_two_a": a, "distance_two_b": b,
            "via_zero_first_hop": zero_path,
            "via_zero_nonbond_code": missing_direct_edge,
            "via_zero_equals_nonbond_code": zero_path == missing_direct_edge,
            "note": "actual pinned Cython and author preprocess_item; only synthetic graphs",
        }
        return result, (first, changed, zero), algos
    finally:
        sys.path.remove(str(path))
        # Loaded Cython module remains usable after its build directory is removed.
        work.cleanup()


def local_shortest_path_check(sources: dict[str, str], items, algos, torch) -> dict:
    tree = ast.parse(sources["local_model"], filename="<pinned-local-model>")
    model = next(item for item in tree.body if isinstance(item, ast.ClassDef)
                 and item.name == "OGBGPTransTiny")
    method = next(item for item in model.body if isinstance(item, ast.FunctionDef)
                  and item.name == "_shortest_path")
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(method)], type_ignores=[]))
    ns = {"torch": torch}
    exec(compile(module, "<pinned-local-shortest-path>", "exec"), ns)
    adjacency = torch.tensor([[[False, True, False],
                               [True, False, True],
                               [False, True, False]]])
    mask = torch.ones((1, 3, 3), dtype=torch.bool)
    actual = ns["_shortest_path"](SimpleNamespace(shortest_path_cap=20), adjacency, mask)[0]
    expected = items[0].spatial_pos
    if not torch.equal(actual, expected):
        raise AssertionError(f"Connected-graph shortest path mismatch: {actual.tolist()} vs {expected.tolist()}")
    disconnected = adjacency.clone()
    disconnected[0, 1, 2] = disconnected[0, 2, 1] = False
    local_disconnected = ns["_shortest_path"](
        SimpleNamespace(shortest_path_cap=20), disconnected, mask)[0]
    author_disconnected, _ = algos.floyd_warshall(disconnected[0].numpy())
    return {
        "connected_local_equals_author": True,
        "connected_distance_0_to_2": int(actual[0, 2]),
        "disconnected_local_code": int(local_disconnected[0, 2]),
        "disconnected_author_code": int(author_disconnected[0, 2]),
        "scope": "actual pinned local BFS and compiled author Cython on three-node synthetic graphs",
    }


def official_padding_check(sources: dict[str, str], items, torch) -> dict:
    ns = {"torch": torch}
    tree = ast.parse(sources["author_collator"])
    names = tuple(item.name for item in tree.body if isinstance(item, ast.FunctionDef))
    extract(sources["author_collator"], names, ns, "<pinned-author-collator>")
    batch = ns["collator"](list(items[:2]))
    if batch["x"].shape[0] != 2:
        raise AssertionError("Expected two-item author collator batch")
    if int(batch["spatial_pos"][0, 0, 2]) != 3:
        raise AssertionError("Author collator did not shift distance two to three")
    # The collator writes -inf for distant pairs, but GraphEdgeFeature later
    # resets the whole attention-bias tensor. Do not call it a runtime mask.
    model_ns = {"torch": torch, "nn": torch.nn, "math": math}
    extract(sources["author_model"], ("init_params", "GraphEdgeFeature"), model_ns,
            "<pinned-author-model-edge>")
    feature = model_ns["GraphEdgeFeature"](
        num_heads=2, num_edges=1536, num_spatial=512, num_edge_dist=128,
        edge_type="multi_hop", multi_hop_max_dist=20, num_layers=12, edge_dim=4)
    with torch.no_grad():
        baseline = feature(batch)
        modified = dict(batch)
        modified["attn_bias"] = torch.full_like(batch["attn_bias"], float("-inf"))
        replay = feature(modified)
    delta = float((baseline - replay).abs().max())
    if not math.isfinite(delta) or delta != 0.0:
        raise AssertionError(f"Author GraphEdgeFeature retained incoming bias: {delta}")
    return {
        "distance_two_collated_code": int(batch["spatial_pos"][0, 0, 2]),
        "padding_code": 0,
        "incoming_bias_max_effect": delta,
        "pair_shape": list(baseline.shape),
    }


def propagation_forward(sources: dict[str, str], torch) -> dict:
    author = {"torch": torch, "nn": torch.nn}
    local = {"torch": torch, "nn": torch.nn}
    extract(sources["author_model"], ("GraphPropagationAttention",), author,
            "<pinned-author-attention>")
    extract(sources["local_model"], ("GraphPropagationAttention",), local,
            "<pinned-local-attention>")
    torch.manual_seed(42)
    official = author["GraphPropagationAttention"](
        node_dim=8, edge_dim=4, num_heads=2, qkv_bias=True, attn_drop=0.0, proj_drop=0.0)
    adapted = local["GraphPropagationAttention"](
        node_channels=8, pair_channels=4, num_heads=2, dropout=0.0)
    aliases = {
        "qkv": "qkv", "reduce": "pair_to_attention", "expand": "attention_to_pair",
        "fc": "pair_to_node", "proj": "output",
    }
    copy_state = {}
    for key, tensor in official.state_dict().items():
        stem, suffix = key.split(".", 1)
        copy_state[aliases[stem] + "." + suffix] = tensor.clone()
    adapted.load_state_dict(copy_state, strict=True)
    official.eval()
    adapted.eval()
    nodes = torch.randn(2, 4, 8)
    pairs = torch.randn(2, 4, 4, 4)
    mask = torch.zeros(2, 1, 1, 4, dtype=torch.bool)
    mask[1, :, :, 3] = True
    with torch.no_grad():
        x1, e1 = official(nodes, pairs, mask)
        x2, e2 = adapted(nodes, pairs, mask)
    node_delta = float((x1 - x2).abs().max())
    pair_delta = float((e1 - e2).abs().max())
    if node_delta > 1e-6 or pair_delta > 1e-6:
        raise AssertionError(f"Propagation mismatch: node={node_delta}, pair={pair_delta}")
    return {"node_max_abs_diff": node_delta, "pair_max_abs_diff": pair_delta,
            "batch": 2, "nodes_including_padding": 4, "equal_weights": True}


def initialization_scale(sources: dict[str, str], torch) -> dict:
    # OGB's nine feature-table lengths; the accepted local model uses its
    # separate AtomEncoder tables and two default nn.Embedding degree tables.
    dims = (119, 5, 12, 12, 10, 6, 6, 2, 2)
    channels = 256
    atom_variance = sum(2.0 / (dim + channels) for dim in dims)
    local_degree_variance = 2.0
    author_atom_variance = 9 * (0.02 ** 2)
    author_degree_variance = 2 * (0.02 ** 2)
    if "self.atom_encoder = AtomEncoder(node_channels)" not in sources["local_model"]:
        raise AssertionError("Local atom initializer assumption no longer holds")
    if "module.weight.data.normal_(mean=0.0, std=0.02)" not in sources["author_model"]:
        raise AssertionError("Author initializer assumption no longer holds")
    if not (0.967 < local_degree_variance / (atom_variance + local_degree_variance) < 0.969):
        raise AssertionError("Local analytical fraction drifted")
    return {
        "ogb_atom_expected_variance": atom_variance,
        "local_degree_expected_variance": local_degree_variance,
        "local_degree_fraction": local_degree_variance / (atom_variance + local_degree_variance),
        "author_atom_expected_variance": author_atom_variance,
        "author_degree_expected_variance": author_degree_variance,
        "author_degree_fraction": author_degree_variance / (author_atom_variance + author_degree_variance),
        "scope": "independent-zero-mean initialization expectation before LayerNorm, not measured MAE",
        "torch_version": torch.__version__,
    }


def optimizer_groups(sources: dict[str, str], torch) -> dict:
    ns = {"logger": logging.getLogger("author-optimizer")}
    extract(sources["author_optimizer"],
            ("check_keywords_in_name", "check_keywords_in_dict", "set_weight_decay_and_lr"),
            ns, "<pinned-author-optimizer>")
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))
    groups = ns["set_weight_decay_and_lr"](model, 0.05, 0.001, layerwise_lr=False)
    by_id = {id(param): group["weight_decay"] for group in groups for param in group["params"]}
    author = {name: by_id[id(param)] for name, param in model.named_parameters()}
    local = {name: 0.05 for name, _ in model.named_parameters()}
    if not (author["0.weight"] == 0.05 and author["0.bias"] == 0.0
            and author["1.weight"] == 0.0 and author["1.bias"] == 0.0):
        raise AssertionError(f"Unexpected pinned author optimizer groups: {author}")
    return {"author_decay_by_parameter": author, "local_full_runner_decay": local}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output = args.output or (Path("/kaggle/working/molgap_gptrans_parity_cpu")
                             if Path("/kaggle/working").is_dir()
                             else Path(tempfile.mkdtemp(prefix="molgap_gptrans_parity_")))
    start = time.monotonic()
    summary = {"schema": "molgap-gptrans-parity-cpu-v1", "author_commit": AUTHOR,
               "local_frozen_commit": LOCAL, "cpu_only": True,
               "dataset_roles_read": [], "checkpoint_loaded": False,
               "kaggle_kernel_id": "kaseichou/molgap-gptrans-cpu-parity-v1",
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "mode": "static-only" if args.static_only else "cpu-runtime",
               "tests": {}, "output_directory": str(output)}
    atomic_json(output / "summary.json", summary)

    def record(name: str, fn) -> None:
        try:
            detail = fn()
            item = {"status": "PASS", "detail": detail}
        except Exception as exc:
            item = {"status": "FAIL", "error_type": type(exc).__name__,
                    "error": str(exc)[:1200]}
        summary["tests"][name] = item["status"]
        atomic_json(output / f"{name}.json", item)
        atomic_json(output / "summary.json", summary)

    payloads = None
    try:
        payloads = source_bytes()
        record("source_identity", lambda: {
            name: {"url": SOURCES[name][0], "sha256": hashlib.sha256(data).hexdigest(),
                   "bytes": len(data)} for name, data in payloads.items()})
    except Exception as exc:
        summary["tests"]["source_identity"] = "FAIL"
        atomic_json(output / "source_identity.json", {"status": "FAIL",
                    "error_type": type(exc).__name__, "error": str(exc)[:1200]})
        summary["elapsed_seconds"] = time.monotonic() - start
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return 1
    sources = source_text(payloads)
    record("static_contract", lambda: static_contract(sources))
    record("training_contract_static", lambda: training_contract_static(sources))
    record("synthetic_path_semantics", synthetic_path_semantics)
    if not args.static_only:
        try:
            import numpy as np
            import torch
            torch.set_num_threads(1)
            if torch.cuda.is_available():
                raise RuntimeError("CPU-only Kaggle kernel unexpectedly sees CUDA")
            summary["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                                  "numpy": np.__version__}
            atomic_json(output / "summary.json", summary)
        except Exception as exc:
            summary["runtime_error"] = f"{type(exc).__name__}: {exc}"[:1200]
            summary["elapsed_seconds"] = time.monotonic() - start
            atomic_json(output / "summary.json", summary)
            print(json.dumps(summary, sort_keys=True))
            return 1
        # The compiled author Cython check is required, not silently replaced
        # by the stdlib translation if the Kaggle image cannot build it.
        holder = {}
        def path_test():
            detail, items, algos = official_path_check(sources, payloads, torch, np)
            holder["items"] = items
            holder["algos"] = algos
            return detail
        record("official_cython_path", path_test)
        if "items" in holder:
            record("local_shortest_path", lambda: local_shortest_path_check(
                sources, holder["items"], holder["algos"], torch))
            record("official_collator_padding", lambda: official_padding_check(sources, holder["items"], torch))
        else:
            for name in ("local_shortest_path", "official_collator_padding"):
                summary["tests"][name] = "NOT_RUN"
                atomic_json(output / f"{name}.json",
                            {"status": "NOT_RUN", "reason": "author Cython path unavailable"})
            atomic_json(output / "summary.json", summary)
        record("propagation_forward", lambda: propagation_forward(sources, torch))
        record("initialization_scale", lambda: initialization_scale(sources, torch))
        record("optimizer_groups", lambda: optimizer_groups(sources, torch))
    expected = TEST_NAMES[:4] if args.static_only else TEST_NAMES
    summary["all_required_pass"] = all(summary["tests"].get(name) == "PASS" for name in expected)
    summary["elapsed_seconds"] = time.monotonic() - start
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_required_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
