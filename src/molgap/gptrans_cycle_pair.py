"""One zero-start common-cycle pair encoding on the accepted local-bond core."""
from pathlib import Path
import torch
from torch import nn

from .gptrans_capacity import BondLocalGPTrans, construct as construct_local
from .screen_policy import canonical_fingerprint
from .v4_runtime import normalized_source_sha256, state_dict_sha256, read_frozen_state_artifact
from .training_reproducibility import atomic_torch_save, sha256_file

PARENT = "degree_bond_local_ema999"
CYCLE = "degree_local_cycle_common_ema999"
MODES = (PARENT, CYCLE)
PARENT_FILE_SHA256 = "d471d9f94ff2c10436261b7f70f3e2d9111978f1ac8aadfabea2da32a5fe9a1d"
PARENT_STATE_SHA256 = "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b"


def cycle_categories(batch, layout):
    """Pack accepted sparse relations on CPU; never enumerate cycles in training."""
    index, category = batch.cycle_pair_index, batch.cycle_pair_category
    if (index.device.type != "cpu" or category.device.type != "cpu" or index.dtype != torch.int64
            or index.ndim != 2 or index.shape[0] != 2 or category.dtype != torch.int64
            or category.shape != (index.shape[1],) or ((category < 1) | (category > 2)).any()
            or (index < 0).any() or (index >= len(batch.x)).any()):
        raise ValueError("Malformed CPU common-cycle relations")
    source, target = index
    graph = batch.batch[source]
    if not torch.equal(graph, batch.batch[target]) or (source == target).any():
        raise ValueError("Cycle relation crosses graphs or includes self")
    local_source = source - batch.ptr[graph] + 1
    local_target = target - batch.ptr[graph] + 1
    dense = torch.zeros((layout["batch_size"], layout["max_nodes"] + 1,
                         layout["max_nodes"] + 1), dtype=torch.long)
    dense[graph, local_source, local_target] = category
    if not torch.equal(dense, dense.transpose(1, 2)):
        raise ValueError("Cycle relation must be symmetric")
    return dense


class CommonCycleGPTrans(BondLocalGPTrans):
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
        layout = getattr(self, "_execution_layout", None)
        if layout is None or "cycle_categories" not in layout:
            raise ValueError("Common-cycle model requires immutable prepared inputs")
        pair = pair + self.cycle_embedding(layout["cycle_categories"]).permute(0, 3, 1, 2)
        graph, source, target = self._local_edges(edge_index, batch, len(x))
        edges = (graph, source + 1, target + 1)
        for block in self.blocks:
            node, pair = block(node, pair, padding, edges)
        return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))


def construct(mode, state):
    if mode not in MODES:
        raise ValueError("Unknown local/cycle mode")
    # Recover only the inherited G1 core; local-message tensors load strictly later.
    base = {}
    for name, value in state.items():
        if name.startswith("cycle_embedding."):
            continue
        fields = name.split(".")
        if fields[0] == "blocks":
            if fields[2] != "core":
                continue
            fields.pop(2)
        base[".".join(fields)] = value
    model = construct_local(PARENT, base)
    if mode == CYCLE:
        model.__class__ = CommonCycleGPTrans
        with torch.random.fork_rng(devices=[]):
            model.cycle_embedding = nn.Embedding(3, 32, padding_idx=0)
            nn.init.zeros_(model.cycle_embedding.weight)
    model._molgap_author_variant = mode
    return model


def freeze_initial(parent_path, output):
    """Extend accepted tensors only, without constructing a molecular model locally."""
    if sha256_file(parent_path) != PARENT_FILE_SHA256:
        raise ValueError("Parent initialization file identity changed")
    parent, _ = read_frozen_state_artifact(parent_path, expected_state_sha256=PARENT_STATE_SHA256)
    if "cycle_embedding.weight" in parent:
        raise ValueError("Parent already contains cycle parameters")
    state = {**parent, "cycle_embedding.weight": torch.zeros(3, 32)}
    digest = state_dict_sha256(state)
    atomic_torch_save(output, {"format": "molgap-cycle-pair-initial-v1", "model_state": state,
        "state_sha256": digest, "parent_tensor_sha256": PARENT_STATE_SHA256,
        "parameters": 5871297})
    return {"state_sha256": digest, "parent_tensor_sha256": PARENT_STATE_SHA256,
            "file_sha256": sha256_file(output), "parameters": 5871297}


def architecture_identity(mode):
    if mode not in MODES:
        raise ValueError("Unknown local/cycle mode")
    return canonical_fingerprint({"mode": mode, "maximum_chordless_cycle": 18,
        "source": normalized_source_sha256(Path(__file__)),
        "parent_source": normalized_source_sha256(Path(__file__).with_name("gptrans_capacity.py"))})
