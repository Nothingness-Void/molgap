"""Static-model execution helpers; no trainer, IO or scientific recipe changes."""
from contextlib import contextmanager
from copy import copy
from pathlib import Path
import weakref

import torch


MODEL_FIELDS = ('x', 'edge_index', 'edge_attr', 'batch', 'execution_layout')


class LoadedDatasetCache:
    """Reuse deserialization, never replace the owner's content acceptance gates."""
    def __init__(self, load):
        self.load = load
        self.entries = {}
        self.statistics = dict(loads=0, hits=0)

    @staticmethod
    def _stamps(paths):
        return tuple((s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino)
                     for s in (p.stat() for p in paths))

    def __call__(self, paths):
        paths = tuple(Path(p).resolve(strict=True) for p in paths)
        stamps = self._stamps(paths)
        if paths in self.entries:
            retained_stamps, result = self.entries[paths]
            if stamps != retained_stamps:
                raise RuntimeError("Loaded dataset files changed inside frozen context")
            self.statistics['hits'] += 1
            return result
        if not paths or len(self.entries) >= 2:
            raise RuntimeError("Only one frozen train and one development dataset are supported")
        result = self.load(paths)
        if self._stamps(paths) != stamps:
            raise RuntimeError("Dataset changed during deserialization")
        self.entries[paths] = stamps, result
        self.statistics['loads'] += 1
        return result

    def clear(self):
        self.entries.clear()


def development_cpu_view(batch):
    """Do not spend the bounded dev cache on unused RWSE/geometry payloads."""
    result = copy(batch)
    retained = set(MODEL_FIELDS) | {'y', 'source_idx', 'row_index', 'ptr', 'num_nodes'}
    for key in list(result.keys()):
        if key not in retained:
            del result[key]
    for name in ('_slice_dict', '_inc_dict'):
        values = getattr(result, name, None)
        if values is not None:
            setattr(result, name, {k: v for k, v in values.items() if k in retained})
    return result


class BoundForeachEMA:
    """One cached tensor-list EMA implementation for all live training callers."""
    def __init__(self, model, decay=0.9999):
        if decay not in (0.9999, 0.999):
            raise ValueError("EMA decay must be a reviewed frozen value")
        self.decay = decay
        self.state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        self._binding = None
        # Reports retain this plain counter dict, never the EMA/model tensors.
        self.statistics = dict(plan_builds=0, update_calls=0)

    @property
    def plan_builds(self):
        return self.statistics['plan_builds']

    @property
    def update_calls(self):
        return self.statistics['update_calls']

    def load_state_dict(self, state):
        if state.keys() != self.state.keys():
            raise RuntimeError("EMA state identity changed")
        self.state = {name: value.clone() for name, value in state.items()}
        self._binding = None

    def state_dict(self):
        return self.state

    def _bind(self, model):
        raw = model.state_dict(keep_vars=True)
        if raw.keys() != self.state.keys():
            raise RuntimeError("EMA model keys changed")
        slots, floating_targets, floating_sources, integer_pairs = [], [], [], []
        # Keep module registration guards without reconstructing state_dict per step.
        registries = []
        for module in model.modules():
            for kind in ('_parameters', '_buffers', '_modules'):
                mapping = getattr(module, kind)
                registries.append((module, kind, mapping, tuple(mapping.items())))
        for name, value in raw.items():
            parent, _, leaf = name.rpartition('.')
            module = model.get_submodule(parent) if parent else model
            mapping = module._parameters if leaf in module._parameters else module._buffers
            if mapping.get(leaf) is not value:
                raise RuntimeError("Custom state_dict tensors are unsupported by bound EMA")
            target = self.state[name]
            if (target.shape != value.shape or target.dtype != value.dtype
                    or target.device != value.device):
                raise RuntimeError("EMA tensor schema changed")
            source = value.detach()
            slots.append((name, value, source, target, source.data_ptr(), target.data_ptr()))
            if value.is_floating_point():
                floating_targets.append(target)
                floating_sources.append(source)
            else:
                integer_pairs.append((target, source))
        self._binding = (weakref.ref(model), self.state, registries, slots,
                         floating_targets, floating_sources, integer_pairs)
        self.statistics['plan_builds'] += 1

    @torch.no_grad()
    def update(self, model):
        if self._binding is None:
            self._bind(model)
        bound, state, registries, slots, targets, sources, integer_pairs = self._binding
        if bound() is not model or state is not self.state or len(state) != len(slots):
            raise RuntimeError("EMA binding changed; use load_state_dict to rebind")
        for module, kind, mapping, entries in registries:
            if (getattr(module, kind) is not mapping or len(mapping) != len(entries)
                    or any(mapping.get(name) is not value for name, value in entries)):
                raise RuntimeError("EMA model registration changed")
        for name, value, source, target, source_ptr, target_ptr in slots:
            if (value.data_ptr() != source_ptr or target.data_ptr() != target_ptr
                    or state.get(name) is not target or value.dtype != source.dtype
                    or value.device != source.device or value.shape != source.shape
                    or value.stride() != source.stride() or target.shape != source.shape
                    or target.dtype != source.dtype or target.device != source.device):
                raise RuntimeError("EMA bound tensor storage changed")
        for target, source in integer_pairs:
            target.copy_(source)
        if targets:
            torch._foreach_mul_(targets, self.decay)
            torch._foreach_add_(targets, sources, alpha=1.0 - self.decay)
        self.statistics['update_calls'] += 1


class BatchTransferView:
    """Transfer only fields consumed by registered GPTrans model/step owners."""
    def __init__(self, batch, *, training):
        self.batch_data = batch
        self.training = training

    def __getattr__(self, name):
        return getattr(self.batch_data, name)

    def to(self, device, *, non_blocking=False):
        fields = MODEL_FIELDS
        if self.training:
            fields += ('y',)
        return self.batch_data.to(device, *fields, non_blocking=non_blocking)


class TransferLoader:
    def __init__(self, source, *, training):
        self.source, self.training = source, training

    @property
    def loader(self):
        return self.source.loader

    @property
    def generator(self):
        return self.loader.generator

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        for batch in self.source:
            yield BatchTransferView(batch, training=self.training)


class ReusedWeightView:
    """Retain one live backup, preserving strict native state loading and restore."""
    def __init__(self):
        self.model = None
        self.backup = None
        self.allocations = 0
        self.retained_tensor_bytes = 0
        self.busy = False

    @contextmanager
    def use(self, model, ema, weights):
        if weights not in {'live', 'ema'} or self.busy:
            raise ValueError("Invalid or nested evaluation weight view")
        self.busy = True
        try:
            if weights == 'live':
                yield
                return
            current = model.state_dict()
            if self.backup is None or self.model() is not model:
                self.backup = {k: torch.empty_like(v) for k, v in current.items()}
                self.model = weakref.ref(model)
                self.allocations += 1
                self.retained_tensor_bytes = sum(v.numel() * v.element_size() for v in self.backup.values())
            if current.keys() != self.backup.keys() or any(
                v.shape != self.backup[k].shape or v.dtype != self.backup[k].dtype
                or v.device != self.backup[k].device for k, v in current.items()
            ):
                raise RuntimeError("Evaluation backup schema changed")
            with torch.no_grad():
                if current:
                    torch._foreach_copy_(list(self.backup.values()), list(current.values()))
            try:
                # Retain native load hooks, strict key checks and buffer semantics.
                model.load_state_dict(ema.state_dict(), strict=True)
                yield
            finally:
                model.load_state_dict(self.backup, strict=True)
        finally:
            self.busy = False

    def clear(self):
        self.backup = None
        self.model = None
        self.retained_tensor_bytes = 0
