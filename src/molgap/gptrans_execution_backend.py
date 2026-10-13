"""Opt-in execution changes; original scientific model/recipe stay untouched.

Only the local-bond performance study enables this process-local context.
Neither checkpoints nor historical runtime certificates qualify these changes.
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
    if hasattr(batch, "cycle_pair_index") or hasattr(batch, "cycle_pair_category"):
        from .gptrans_cycle_pair import cycle_categories
        batch.execution_layout["cycle_categories"] = cycle_categories(batch, batch.execution_layout)
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


def prepared_shortest_path(model, adjacency, pair_mask):
    """Exact existing bounded reachability, without an unused final expansion."""
    size, maximum = adjacency.shape[:2]
    unreachable = model.shortest_path_cap + 1
    distance = torch.full((size, maximum, maximum), unreachable, dtype=torch.long, device=adjacency.device)
    diagonal = torch.eye(maximum, dtype=torch.bool, device=adjacency.device)
    distance.masked_fill_(diagonal.unsqueeze(0) & pair_mask, 0)
    frontier, adjacency_float = adjacency, adjacency.float()
    for step in range(1, model.shortest_path_cap + 1):
        distance.masked_fill_(frontier & (distance == unreachable) & pair_mask, step)
        if step < model.shortest_path_cap:
            frontier = torch.bmm(frontier.float(), adjacency_float) > 0
    return distance


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


def buffered_evaluate(owner, forward, loader, model, ema, graphs, mean, std,
                      *, weights="ema", device="cuda", source_idx_start=100000):
    if weights not in {"ema", "live"}:
        raise ValueError("Invalid weight view")
    live = {k: v.detach().clone() for k, v in model.state_dict().items()} if weights == "ema" else None
    if live is not None:
        model.load_state_dict(ema.state_dict(), strict=True)
    model.eval()
    predictions, targets, sources = [], [], []
    try:
        with torch.no_grad():
            for batch in loader(graphs):
                # Retain CPU tensors before PyG mutates the batch in .to().
                targets.append(batch.y.view(-1).float())
                index = getattr(batch, "source_idx", getattr(batch, "row_index", None))
                if index is None:
                    raise ValueError("Development source identity is missing")
                sources.append(index.view(-1).long())
                batch = batch.to(device, non_blocking=True)
                predictions.append(forward(model, batch) * std + mean)
        prediction = torch.cat(predictions).cpu()
    finally:
        if live is not None:
            model.load_state_dict(live, strict=True)
    target, source = torch.cat(targets), torch.cat(sources)
    if prediction.numel() != owner.DEVELOPMENT_ROWS or not torch.equal(
            source, torch.arange(source_idx_start, source_idx_start + owner.DEVELOPMENT_ROWS)):
        raise ValueError("Development row count/order changed")
    # Keep the original CPU float32 metric reduction, not a GPU approximation.
    return {"mae_eV": float((prediction - target).abs().mean()),
            "prediction_eV": prediction, "target_eV": target, "source_idx": source}


@contextmanager
def backend(owner, optimized, timing_root=None):
    """Patch only this isolated worker; restore originals even on exceptions."""
    from torch_geometric.loader import DataLoader
    originals = {name: getattr(owner, name) for name in
                 ("_make_model", "_training_loader", "_development_loader", "_forward",
                  "_make_training_state", "_optimizer_step", "_evaluate", "_save_checkpoint",
                  "_scientific_fields")}
    loaders, times = {}, {"optimizer_call_seconds": 0., "training_batches": 0,
                        "evaluation_seconds": 0., "evaluation_passes": 0,
                        "checkpoint_publication_seconds": 0.}
    def forward(model, batch):
        if optimized:
            if not hasattr(batch, "execution_layout"):
                raise ValueError("Prepared execution layout missing")
            model._execution_layout = batch.execution_layout
        try:
            return originals["_forward"](model, batch)
        finally:
            if optimized:
                del model._execution_layout
    def make_model(*args, **kwargs):
        model = originals["_make_model"](*args, **kwargs)
        if optimized:
            if getattr(model, "_molgap_author_variant", None) != "degree_bond_local_ema999":
                raise ValueError("Execution backend is qualified only for the local-bond model")
            model._dense_inputs = MethodType(dense_inputs, model)
            model._local_edges = MethodType(cached_edges, model)
        return model
    def training_loader(graphs, epoch):
        if not optimized:
            return originals["_training_loader"](graphs, epoch)
        key = ("train", id(graphs))
        if key not in loaders:
            assert_static_dataset(graphs)
            sampler = EpochBatches(owner, len(graphs))
            generator = torch.Generator()
            loader = DataLoader(graphs, batch_sampler=sampler, num_workers=owner.LOADER_WORKERS,
                persistent_workers=True, pin_memory=True, prefetch_factor=2, generator=generator)
            loaders[key] = sampler, generator, PreparedLoader(loader)
        sampler, generator, loader = loaders[key]
        sampler.epoch = epoch
        generator.manual_seed(owner.SEED + epoch)
        return loader
    def development_loader(graphs):
        if not optimized:
            return originals["_development_loader"](graphs)
        # The original unseeded development iterator consumes one CPU RNG draw
        # per pass. Preserve it even though persistent workers use a private RNG.
        torch.empty((), dtype=torch.int64).random_()
        key = ("dev", id(graphs))
        if key not in loaders:
            assert_static_dataset(graphs)
            # An explicit generator keeps worker initialization off the model RNG.
            loaders[key] = PreparedLoader(DataLoader(graphs, batch_size=owner.PHYSICAL_BATCH,
                shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True,
                generator=torch.Generator().manual_seed(314159)))
        return loaders[key]
    def state(*args, **kwargs):
        model, optimizer, scheduler, ema = originals["_make_training_state"](*args, **kwargs)
        if optimized:
            # A backend experiment, not an unqualified replacement for a frozen reference.
            for group in optimizer.param_groups:
                group["foreach"] = True
            from .foreach_ema import ForeachExponentialMovingAverage
            fast = ForeachExponentialMovingAverage(model, decay=ema.decay)
            fast.load_state_dict(ema.state_dict())
            ema = fast
        return model, optimizer, scheduler, ema
    def step(*args, **kwargs):
        began = time.perf_counter()
        value = originals["_optimizer_step"](*args, **kwargs)
        times["optimizer_call_seconds"] += time.perf_counter() - began
        times["training_batches"] += 1
        return value
    def evaluate(model, ema, graphs, mean, std, **kwargs):
        torch.cuda.synchronize()
        began = time.perf_counter()
        try:
            return (buffered_evaluate(owner, forward, development_loader, model, ema, graphs,
                                     mean, std, **kwargs) if optimized else
                    originals["_evaluate"](model, ema, graphs, mean, std, **kwargs))
        finally:
            torch.cuda.synchronize()
            times["evaluation_seconds"] += time.perf_counter() - began
            times["evaluation_passes"] += 1
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
                    **times, "epoch": kwargs["epoch"], "optimized": optimized,
                    "optimizer_call_timing_scope": "CPU enqueue time; not synchronized GPU step time"})
    def scientific_fields(variant="reference"):
        fields = originals["_scientific_fields"](variant)
        fields["execution_backend"] = "optimized-v1" if optimized else "original-v1"
        if optimized:
            from .screen_policy import canonical_fingerprint
            fields["optimizer_fingerprint"] = canonical_fingerprint({
                "name": "torch-adamw", "foreach": True, "fused": False,
                "lr": owner.LEARNING_RATE, "weight_decay": owner._weight_decay(variant)})
        return fields
    replacements = dict(_make_model=make_model, _training_loader=training_loader,
        _development_loader=development_loader, _forward=forward, _make_training_state=state,
        _optimizer_step=step, _evaluate=evaluate, _save_checkpoint=checkpoint,
        _scientific_fields=scientific_fields)
    for name, value in replacements.items():
        setattr(owner, name, value)
    try:
        yield times
    finally:
        for name, value in originals.items():
            setattr(owner, name, value)
        for value in loaders.values():
            wrapper = value[-1] if isinstance(value, tuple) else value
            iterator = getattr(wrapper.loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()


def bind_model_layout(model):
    """Attach the shared fast input assembly without changing model tensors."""
    from .gptrans import OGBGPTransTiny
    if not isinstance(model, OGBGPTransTiny):
        raise ValueError("Prepared GPTrans layout requires the reviewed model interface")
    if model._dense_inputs.__func__ is not OGBGPTransTiny._dense_inputs:
        raise ValueError("Input-changing addons must supply an accelerated input adapter")
    if model._local_edges is not OGBGPTransTiny._local_edges:
        raise ValueError("Edge-changing addons must supply an accelerated input adapter")
    if model._shortest_path.__func__ is not OGBGPTransTiny._shortest_path:
        raise ValueError("Distance-changing addons must supply an accelerated input adapter")
    model._dense_inputs = MethodType(dense_inputs, model)
    model._local_edges = MethodType(cached_edges, model)
    model._shortest_path = MethodType(prepared_shortest_path, model)
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
