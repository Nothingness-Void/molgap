"""Default prepared GPTrans inputs and instrumentation, without job authority.

Historical runtime certificates never qualify a new packaged execution identity.
"""
from contextlib import contextmanager
import time
from types import MethodType

import torch


def prepare_layout(batch):
    """Derive the immutable graph layout on CPU before the ordinary H2D copy."""
    if batch.batch.device.type != "cpu" or batch.ptr.device.type != "cpu":
        raise ValueError("Execution layout must be prepared before H2D")
    ptr, membership = batch.ptr, batch.batch
    size, nodes = int(batch.num_graphs), len(batch.x)
    if size < 1 or len(ptr) != size + 1 or int(ptr[-1]) != nodes:
        raise ValueError("Malformed CPU graph layout")
    counts = ptr[1:] - ptr[:-1]
    if bool((counts <= 0).any()) or not torch.equal(
            membership, torch.repeat_interleave(torch.arange(size), counts)):
        raise ValueError("Graph membership must be contiguous and ordered")
    maximum = int(counts.max())
    local = torch.arange(nodes) - ptr[membership]
    graph = membership[batch.edge_index[0]]
    if not torch.equal(graph, membership[batch.edge_index[1]]):
        raise ValueError("A bond crosses graph boundaries")
    mask = torch.arange(maximum)[None, :] < counts[:, None]
    batch.execution_layout = {
        "batch_size": size, "max_nodes": maximum, "mask": mask,
        "flat_index": membership * maximum + local,
        "edges": (graph, local[batch.edge_index[0]], local[batch.edge_index[1]]),
    }
    return batch


def dense_inputs(model, x, edge_index, edge_attr, batch):
    """Same dense tensors/arithmetic, without the unused categorical dense copy."""
    layout = model._execution_layout
    size, maximum = layout["batch_size"], layout["max_nodes"]
    mask = layout["mask"]
    embedded = model.atom_encoder(x.long())
    dense = embedded.new_zeros((size * maximum, embedded.shape[-1]))
    dense[layout["flat_index"]] = embedded
    embedded = dense.view(size, maximum, -1)
    graph, source, target = layout["edges"]
    adjacency = torch.zeros((size, maximum, maximum), dtype=torch.bool, device=x.device)
    adjacency[graph, source, target] = True
    degree = adjacency.sum(-1).clamp_max(511)
    embedded = embedded + model.in_degree_encoder(degree) + model.out_degree_encoder(degree)
    token = model.graph_token.expand(size, -1, -1)
    node = model.input_dropout(model.node_norm(torch.cat((token, embedded), dim=1)))
    spatial = model._shortest_path(adjacency, mask[:, None, :] & mask[:, :, None])
    pair = model.spatial_encoder(spatial).permute(0, 3, 1, 2)
    bond = model.bond_encoder(edge_attr.long())
    pair[graph, :, source, target] += bond
    full = pair.new_zeros(size, model.pair_channels, maximum + 1, maximum + 1)
    full[:, :, 1:, 1:] = pair
    full[:, :, 0:1, :] += model.virtual_pair
    full[:, :, 1:, 0:1] += model.virtual_pair
    full_mask = torch.cat((torch.ones(size, 1, dtype=torch.bool, device=x.device), mask), 1)
    return node, full, ~full_mask[:, None, None, :]


def cached_edges(model, edge_index, batch, node_count):
    return model._execution_layout["edges"]


def bind_model_layout(model):
    """Attach the shared fast input assembly without changing model tensors."""
    from .gptrans import OGBGPTransTiny
    if not isinstance(model, OGBGPTransTiny):
        raise ValueError("Prepared GPTrans layout requires the reviewed model interface")
    if model._dense_inputs.__func__ is not OGBGPTransTiny._dense_inputs:
        raise ValueError("Input-changing addons must supply an accelerated input adapter")
    if model._local_edges is not OGBGPTransTiny._local_edges:
        raise ValueError("Edge-changing addons must supply an accelerated input adapter")
    model._dense_inputs = MethodType(dense_inputs, model)
    model._local_edges = MethodType(cached_edges, model)
    return model


def prepared_forward(model, batch):
    """Registered training forward; missing preparation cannot select a fallback."""
    if not hasattr(batch, "execution_layout"):
        raise ValueError("Accelerated GPTrans requires a CPU-prepared batch layout")
    if hasattr(model, "_execution_layout"):
        raise ValueError("Nested GPTrans forward is unsupported")
    model._execution_layout = batch.execution_layout
    try:
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                     getattr(batch, "random_walk_pe", None)).view(-1)
    finally:
        del model._execution_layout


class EpochBatches:
    def __init__(self, owner, length):
        self.owner, self.length, self.epoch = owner, length, 0

    def __iter__(self):
        return iter(self.owner.DeterministicEpochBatchSampler(self.length, self.epoch))

    def __len__(self):
        return self.length // self.owner.PHYSICAL_BATCH


class PreparedLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for batch in self.loader:
            yield prepare_layout(batch)

    def __len__(self):
        return len(self.loader)


def assert_static_dataset(graphs):
    """Persistent workers must not hide random transforms or role changes."""
    for dataset in getattr(graphs, "datasets", (graphs,)):
        if getattr(dataset, "transform", None) is not None:
            raise ValueError("Execution study forbids stochastic dataset transforms")


def buffered_evaluate(owner, forward, loader, model, graphs, mean, std,
                      *, device="cuda", source_idx_start=100000):
    """Evaluate the active weight view with one bounded prediction transfer."""
    model.eval()
    predictions, targets, sources = [], [], []
    with torch.no_grad():
        for batch in loader(graphs):
            targets.append(batch.y.view(-1).float())
            index = getattr(batch, "source_idx", getattr(batch, "row_index", None))
            if index is None:
                raise ValueError("Development source identity is missing")
            sources.append(index.view(-1).long())
            batch = batch.to(device, non_blocking=True)
            predictions.append(forward(model, batch) * std + mean)
    prediction = torch.cat(predictions).cpu()
    target, source = torch.cat(targets), torch.cat(sources)
    if prediction.numel() != owner.DEVELOPMENT_ROWS or not torch.equal(
            source, torch.arange(source_idx_start, source_idx_start + owner.DEVELOPMENT_ROWS)):
        raise ValueError("Development row count/order changed")
    return {"mae_eV": float((prediction - target).abs().mean()),
            "prediction_eV": prediction, "target_eV": target, "source_idx": source}


@contextmanager
def backend(owner, timing_root=None):
    """Instrument the accelerated native owner; there is no legacy execution mode."""
    originals = {name: getattr(owner, name) for name in ("_optimizer_step", "_save_checkpoint")}
    times = {"optimizer_call_seconds": 0., "training_batches": 0,
             "evaluation_seconds": 0., "evaluation_passes": 0,
             "checkpoint_publication_seconds": 0.}
    def step(*args, **kwargs):
        began = time.perf_counter()
        value = originals["_optimizer_step"](*args, **kwargs)
        times["optimizer_call_seconds"] += time.perf_counter() - began
        times["training_batches"] += 1
        return value
    def checkpoint(*args, **kwargs):
        torch.cuda.synchronize()
        began = time.perf_counter()
        try:
            return originals["_save_checkpoint"](*args, **kwargs)
        finally:
            times["checkpoint_publication_seconds"] += time.perf_counter() - began
            if timing_root is not None:
                from .training_reproducibility import atomic_json
                atomic_json(timing_root / f"epoch_{kwargs['epoch']:02d}.json", {
                    **times, "epoch": kwargs["epoch"],
                    "optimizer_call_timing_scope": "CPU enqueue time; not synchronized GPU step time"})
    owner._optimizer_step, owner._save_checkpoint = step, checkpoint
    try:
        yield times
    finally:
        for name, value in originals.items():
            setattr(owner, name, value)
