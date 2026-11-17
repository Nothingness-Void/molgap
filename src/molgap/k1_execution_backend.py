"""Explicit batch-local K1 execution providers; model parameters stay unchanged."""
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, dataclass
from functools import partial


@dataclass(frozen=True)
class K1ExecutionConfig:
    aggregation: str = "cached_scatter"
    pooling: str = "cached_scatter"
    projection: str = "edge"
    evaluation_memory_fill: bool = False

    def __post_init__(self):
        if self.aggregation not in {"native", "cached_scatter", "segmented"}:
            raise ValueError("Unknown K1 aggregation execution provider")
        if self.pooling not in {"native", "cached_scatter", "segmented"}:
            raise ValueError("Unknown K1 pooling execution provider")
        if self.projection not in {"edge", "node"}:
            raise ValueError("Unknown K1 edge projection execution provider")
        if type(self.evaluation_memory_fill) is not bool:
            raise ValueError("Evaluation memory fill must be an exact bool")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("Missing or unknown K1 execution options")
        return cls(**value)


def _bind(stack, obj, name, value):
    missing = object()
    previous = vars(obj).get(name, missing)
    def restore():
        if previous is missing:
            delattr(obj, name)
        else:
            setattr(obj, name, previous)
    setattr(obj, name, value)
    stack.callback(restore)


def _node_projected_update(update, hidden, edge_index, edge_state):
    source, target = edge_index
    context = (edge_state + update.source(hidden).index_select(0, source)
        + update.target(hidden).index_select(0, target))
    return update.output_norm(edge_state + update.update(context))


@contextmanager
def execution_context(model, cpu_batch, device, config):
    """Borrow original edge order, changing only the aggregation view and scope."""
    if config is None:
        yield
        return
    if type(config) is not K1ExecutionConfig:
        raise TypeError("Typed K1 execution configuration is required")
    if hasattr(model, "_k1_execution_active"):
        raise ValueError("Nested K1 execution provider")
    if not hasattr(model, "local_blocks") or not hasattr(model, "edge_updates"):
        raise ValueError("Execution provider requires the owning K1 substrate")
    from .k1_execution_aggregation import AggregationPlan
    with ExitStack() as stack:
        _bind(stack, model, "_k1_execution_active", True)
        if config.aggregation != "native":
            plan = AggregationPlan.from_cpu(cpu_batch.edge_index[1], int(cpu_batch.x.shape[0]), device)
            def aggregate(inputs, index, ptr=None, dim_size=None):
                if dim_size != plan.output_rows or index.shape != (plan.edge_count,):
                    raise ValueError("Aggregation output shape differs from the CPU topology")
                return plan.reduce(inputs, backend=config.aggregation)
            for block in model.local_blocks:
                conv = block.conv
                if conv.aggr != "add" or conv.node_dim not in (0, -2):
                    raise ValueError("Only node-axis additive local aggregation is supported")
                _bind(stack, conv, "aggregate", aggregate)
        if config.pooling != "native":
            if model.pooling != "mean":
                raise ValueError("Execution pooling requires mean readout")
            graphs = int(cpu_batch.ptr.numel()-1)
            pool = AggregationPlan.from_cpu(cpu_batch.batch, graphs, device)
            import torch
            counts = (cpu_batch.ptr[1:] - cpu_batch.ptr[:-1]).clamp_min(1).to(
                device, dtype=torch.float32, non_blocking=True).view(-1, 1)
            def mean(hidden, membership, *, size=None):
                if membership is not cpu_batch.batch or (size is not None and size != graphs):
                    raise ValueError("Readout graph count differs from CPU boundaries")
                total = pool.reduce(hidden, backend=config.pooling)
                return total / counts.to(dtype=hidden.dtype)
            _bind(stack, model, "_pool", mean)
        if config.projection == "node":
            for update in model.edge_updates:
                _bind(stack, update, "forward", partial(_node_projected_update, update))
        yield
