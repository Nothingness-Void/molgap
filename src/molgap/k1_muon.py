"""Frozen FP32 Muon for K1 hidden matrices, with AdamW for other parameters."""
import math

import torch
from torch import nn
from torch_geometric.nn.dense.linear import Linear as GeometricLinear


MUON_POLICY = {
    "momentum": 0.95,
    "nesterov": True,
    "ns_coefficients": [3.4445, -4.775, 2.0315],
    "ns_steps": 5,
    "eps": 1e-7,
    "adjust_lr_fn": "match_rms_adamw",
    "compute_dtype": "float32",
    "parameter_selection": "k1-hidden-linear-and-attention-v1",
}


def parameter_groups(model, weight_decay):
    hidden = set()
    for name, module in model.named_modules():
        if name == "head" or name.startswith("head."):
            continue
        if isinstance(module, (nn.Linear, GeometricLinear)):
            hidden.add(id(module.weight))
        elif isinstance(module, nn.MultiheadAttention):
            hidden.add(id(module.in_proj_weight))
    groups = [{"params": [], "param_names": [], "algorithm": algorithm,
               "weight_decay": weight_decay} for algorithm in ("muon", "adamw")]
    for name, param in sorted(model.named_parameters()):
        if not param.requires_grad:
            continue
        if param.dtype != torch.float32:
            raise ValueError("K1 Muon requires FP32 parameters")
        group = groups[0 if id(param) in hidden else 1]
        group["params"].append(param)
        group["param_names"].append(name)
    return groups


class MuonAdamW(torch.optim.AdamW):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["algorithm"] != "muon":
                continue
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[param]
                if not state:
                    state["step"] = torch.tensor(0., dtype=torch.float32)
                    state["momentum_buffer"] = torch.zeros_like(param)
                state["step"].add_(1)
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 0.05)
                update = grad.lerp(buffer, 0.95)
                transpose = update.shape[0] > update.shape[1]
                if transpose:
                    update = update.T
                update.div_(update.norm().clamp(min=1e-7))
                for _ in range(5):
                    gram = update @ update.T
                    polynomial = torch.addmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
                    update = torch.addmm(update, polynomial, update, beta=3.4445)
                if transpose:
                    update = update.T
                param.mul_(1 - group["lr"] * group["weight_decay"])
                param.add_(update, alpha=-group["lr"] * 0.2 * math.sqrt(max(param.shape)))
        groups = self.param_groups
        try:
            self.param_groups = [group for group in groups if group["algorithm"] == "adamw"]
            super().step()
        finally:
            self.param_groups = groups
        return loss
