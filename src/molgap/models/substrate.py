"""Opt-in runtime slots over retained K1/GPTrans model modules.

Spec owns scientific addon identity. These bindings only describe tensor seams;
they neither register a scientific mechanism nor change a frozen family route.
Attachments must be reconstructed before loading a state_dict and sealed before
creating an optimizer. Whole-model pickle and concurrent forwards are unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class SlotName(str, Enum):
    PRE_MESSAGE = "pre_message"
    POST_AGGREGATION = "post_aggregation"
    PRE_READOUT = "pre_readout"


@dataclass(frozen=True)
class SlotBinding:
    family: str
    slot: SlotName
    layer: int | None = None

    def __post_init__(self):
        if self.family not in {"neural_atom_k1", "gptrans_t"}:
            raise ValueError("Unsupported substrate family")
        if type(self.slot) is not SlotName:
            raise TypeError("Expected a typed SlotName")
        if self.slot is SlotName.PRE_READOUT:
            if self.layer is not None:
                raise ValueError("Readout binding has no layer")
        elif type(self.layer) is not int or self.layer < 1:
            raise ValueError("Message bindings require a positive one-based layer")


@dataclass(frozen=True)
class K1MessageInput:
    node: object
    edge_index: object
    edge_attr: object
    graph_index: object


@dataclass(frozen=True)
class K1AggregationOutput:
    aggregated: object
    target_index: object
    graph_index: object


@dataclass(frozen=True)
class GPTransMessageInput:
    node: object
    pair: object
    key_padding_mask: object


@dataclass(frozen=True)
class GPTransAggregationOutput:
    node_update: object
    pair_update: object
    key_padding_mask: object


@dataclass(frozen=True)
class ReadoutInput:
    embedding: object


@dataclass(frozen=True)
class AttachedAddon:
    name: str
    binding: SlotBinding


def _argument(args, kwargs, index, name, *, optional=False):
    if len(args) > index:
        return args[index]
    if name in kwargs:
        return kwargs[name]
    if optional:
        return None
    raise ValueError("Substrate call lacks " + name)


def _replace_argument(args, kwargs, index, name, value):
    if len(args) > index:
        values = list(args)
        values[index] = value
        return tuple(values), kwargs
    return args, {**kwargs, name: value}


def _same_tensor_shape(actual, expected, label):
    import torch
    if (not torch.is_tensor(actual) or not torch.is_tensor(expected)
            or actual.shape != expected.shape or actual.dtype != expected.dtype
            or actual.device != expected.device):
        raise ValueError("Addon changed tensor shape/dtype/device: " + label)


def _same_context(actual, expected, fields):
    if any(getattr(actual, name) is not getattr(expected, name) for name in fields):
        raise ValueError("Addon changed immutable slot context")


class ModelSubstrate:
    """Attach reviewed modules in caller/Spec order without replacing a backbone.

    Each addon accepts and returns the same typed payload. K1 post-aggregation
    means the sparse additive aggregate before the convolution's residual;
    GPTrans post-aggregation means attention's node/pair updates before block
    residuals. Readout receives an already pooled/virtual graph representation.
    """

    namespace = "_molgap_addons"

    def __init__(self, model, family: str):
        import torch
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Expected torch.nn.Module substrate")
        if family not in {"neural_atom_k1", "gptrans_t"}:
            raise ValueError("Unsupported substrate family")
        if hasattr(model, self.namespace) or hasattr(model, "_molgap_substrate"):
            raise ValueError("Model already has an addon owner or reserved namespace")
        self.model, self.family = model, family
        self._entries, self._handles, self._root_handles = {}, {}, ()
        self._sealed, self._active, self._graph_index = False, False, None
        self._calls = []
        object.__setattr__(model, "_molgap_substrate", self)

    @property
    def bindings(self):
        return tuple(AttachedAddon(name, binding) for name, binding in self._entries.items())

    @property
    def sealed(self):
        return self._sealed

    def seal(self):
        """Freeze attachments before optimizer creation; sealing is irreversible."""
        if self._active:
            raise RuntimeError("Cannot seal during a substrate forward")
        self._sealed = True

    def _mutable(self):
        if self._sealed:
            raise RuntimeError("Addon attachments are sealed for optimizer state")
        if self._active:
            raise RuntimeError("Cannot modify attachments during a substrate forward")

    def _target(self, binding):
        if type(binding) is not SlotBinding or binding.family != self.family:
            raise ValueError("Slot binding differs from substrate family")
        if binding.slot is SlotName.PRE_READOUT:
            name = "head" if self.family == "neural_atom_k1" else "readout"
            target = getattr(self.model, name, None)
        else:
            blocks = getattr(self.model, "local_blocks" if self.family == "neural_atom_k1" else "blocks", ())
            if binding.layer > len(blocks):
                raise ValueError("Slot layer exceeds substrate depth")
            block = blocks[binding.layer - 1]
            target = getattr(block, "conv" if self.family == "neural_atom_k1" else "attention", None)
        import torch
        if not isinstance(target, torch.nn.Module):
            raise ValueError("Substrate does not expose the declared slot module")
        if (self.family == "neural_atom_k1" and binding.slot is SlotName.POST_AGGREGATION
                and not callable(getattr(target, "register_aggregate_forward_hook", None))):
            raise ValueError("K1 aggregate slot requires the PyG aggregate hook")
        return target

    def _install_root(self):
        if self._root_handles:
            return
        def begin(module, args, kwargs):
            # always_call also runs when a pre-hook fails: remember whether
            # this invocation owns the frame before checking for reentrancy.
            self._calls.append(False)
            if self._active:
                raise RuntimeError("Substrate does not support concurrent/reentrant forwards")
            graph = _argument(args, kwargs, 3, "batch", optional=self.family != "neural_atom_k1")
            self._graph_index, self._active = graph, True
            self._calls[-1] = True
        def finish(module, args, kwargs, output):
            if self._calls and self._calls.pop():
                self._active, self._graph_index = False, None
        before = self.model.register_forward_pre_hook(begin, with_kwargs=True)
        try:
            after = self.model.register_forward_hook(finish, with_kwargs=True, always_call=True)
        except BaseException:
            before.remove()
            raise
        self._root_handles = before, after

    def _apply(self, name, payload):
        if not self._active:
            raise RuntimeError("Slot must run inside its owning substrate forward")
        result = getattr(self.model, self.namespace)[name](payload)
        if type(result) is not type(payload):
            raise TypeError("Addon must return the same typed slot payload")
        return result

    def _install(self, name, binding, target):
        if binding.slot is SlotName.PRE_READOUT:
            def before(module, args, kwargs):
                value = _argument(args, kwargs, 0, "input")
                result = self._apply(name, ReadoutInput(value))
                _same_tensor_shape(result.embedding, value, "readout embedding")
                return _replace_argument(args, kwargs, 0, "input", result.embedding)
            return target.register_forward_pre_hook(before, with_kwargs=True)
        if self.family == "neural_atom_k1":
            if binding.slot is SlotName.PRE_MESSAGE:
                def before(module, args, kwargs):
                    node = _argument(args, kwargs, 0, "x")
                    edge = _argument(args, kwargs, 1, "edge_index")
                    attr = _argument(args, kwargs, 2, "edge_attr", optional=True)
                    payload = K1MessageInput(node, edge, attr, self._graph_index)
                    result = self._apply(name, payload)
                    _same_context(result, payload, ("edge_index", "graph_index"))
                    _same_tensor_shape(result.node, node, "K1 node")
                    if attr is None:
                        if result.edge_attr is not None:
                            raise ValueError("Addon introduced an undeclared K1 edge attribute")
                    else:
                        _same_tensor_shape(result.edge_attr, attr, "K1 edge attribute")
                    args, kwargs = _replace_argument(args, kwargs, 0, "x", result.node)
                    if attr is not None:
                        args, kwargs = _replace_argument(args, kwargs, 2, "edge_attr", result.edge_attr)
                    return args, kwargs
                return target.register_forward_pre_hook(before, with_kwargs=True)
            def after(module, inputs, output):
                if len(inputs) != 1 or not isinstance(inputs[0], dict) or "index" not in inputs[0]:
                    raise ValueError("Unexpected K1 aggregate hook ABI")
                payload = K1AggregationOutput(output, inputs[0]["index"], self._graph_index)
                result = self._apply(name, payload)
                _same_context(result, payload, ("target_index", "graph_index"))
                _same_tensor_shape(result.aggregated, output, "K1 aggregate")
                return result.aggregated
            return target.register_aggregate_forward_hook(after)
        if binding.slot is SlotName.PRE_MESSAGE:
            def before(module, args, kwargs):
                node = _argument(args, kwargs, 0, "node")
                pair = _argument(args, kwargs, 1, "pair")
                mask = _argument(args, kwargs, 2, "key_padding_mask")
                payload = GPTransMessageInput(node, pair, mask)
                result = self._apply(name, payload)
                _same_context(result, payload, ("key_padding_mask",))
                _same_tensor_shape(result.node, node, "GPTrans node")
                _same_tensor_shape(result.pair, pair, "GPTrans pair")
                args, kwargs = _replace_argument(args, kwargs, 0, "node", result.node)
                return _replace_argument(args, kwargs, 1, "pair", result.pair)
            return target.register_forward_pre_hook(before, with_kwargs=True)
        def after(module, args, kwargs, output):
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError("Unexpected GPTrans attention output ABI")
            mask = _argument(args, kwargs, 2, "key_padding_mask")
            payload = GPTransAggregationOutput(*output, mask)
            result = self._apply(name, payload)
            _same_context(result, payload, ("key_padding_mask",))
            _same_tensor_shape(result.node_update, output[0], "GPTrans node update")
            _same_tensor_shape(result.pair_update, output[1], "GPTrans pair update")
            return result.node_update, result.pair_update
        return target.register_forward_hook(after, with_kwargs=True)

    def attach(self, name: str, addon, binding: SlotBinding):
        """Attach one reviewed module; explicit call order defines composition."""
        self._mutable()
        if type(name) is not str or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError("Addon namespace name must be a Python identifier")
        if name in self._entries:
            raise ValueError("Addon name is already attached")
        import torch
        if not isinstance(addon, torch.nn.Module) or any(addon is item for item in self.model.modules()):
            raise ValueError("Addon must be a distinct torch.nn.Module")
        if hasattr(torch.nn.ModuleDict(), name):
            raise ValueError("Addon name collides with the module namespace")
        target = self._target(binding)
        reference = next(self.model.parameters(), None)
        if reference is not None:
            for tensor in (*addon.parameters(), *addon.buffers()):
                if tensor.is_floating_point() and tensor.dtype != reference.dtype:
                    raise ValueError("Addon floating dtype differs from substrate")
            addon.to(device=reference.device)
        addon.train(self.model.training)
        created = not hasattr(self.model, self.namespace)
        if created:
            self.model.add_module(self.namespace, torch.nn.ModuleDict())
        getattr(self.model, self.namespace)[name] = addon
        handle = None
        try:
            handle = self._install(name, binding, target)
            self._install_root()
        except BaseException:
            if handle is not None:
                handle.remove()
            del getattr(self.model, self.namespace)[name]
            if created:
                delattr(self.model, self.namespace)
            raise
        self._handles[name], self._entries[name] = handle, binding
        return addon

    def detach(self, name: str):
        """Remove both execution hook and serialized addon weights before sealing."""
        self._mutable()
        if name not in self._entries:
            raise ValueError("Addon is not attached")
        self._handles.pop(name).remove()
        self._entries.pop(name)
        modules = getattr(self.model, self.namespace)
        addon = modules[name]
        del modules[name]
        if not self._entries:
            delattr(self.model, self.namespace)
            for handle in self._root_handles:
                handle.remove()
            self._root_handles = ()
        return addon

    def named_addon_parameters(self):
        """Yield state_dict-compatible parameter names for sandbox connectivity."""
        if hasattr(self.model, self.namespace):
            for name, parameter in getattr(self.model, self.namespace).named_parameters():
                yield self.namespace + "." + name, parameter


def bind_substrate(model, family: str, attachments=()):
    """Reconstruct runtime bindings from ordered (name, module, binding) tuples."""
    owner = ModelSubstrate(model, family)
    try:
        for name, addon, binding in attachments:
            owner.attach(name, addon, binding)
    except BaseException:
        for entry in reversed(owner.bindings):
            owner.detach(entry.name)
        delattr(model, "_molgap_substrate")
        raise
    return owner


__all__ = ["SlotName", "SlotBinding", "AttachedAddon", "K1MessageInput",
           "K1AggregationOutput", "GPTransMessageInput", "GPTransAggregationOutput",
           "ReadoutInput", "ModelSubstrate", "bind_substrate"]


# Retained pickles continue to name the historical owner.
for _compat_object in (SlotName, SlotBinding, K1MessageInput, K1AggregationOutput, GPTransMessageInput, GPTransAggregationOutput, ReadoutInput, AttachedAddon, _argument, _replace_argument, _same_tensor_shape, _same_context, ModelSubstrate, bind_substrate,):
    _compat_object.__module__ = "molgap.model_substrate"
del _compat_object
