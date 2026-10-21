"""Passive observations for the frozen K1 normalization experiment only.

No training loop, logger, parameter mutation or data-role discovery. The session
activates update observation; qualification/model probes never advance counters.
"""
from copy import deepcopy
import math


class NormDiagnostics:
    def __init__(self, variant):
        self.variant = variant
        self.completed_updates = 0
        self.gradient_windows = []
        self.pending = []
        self.probes = []
        self.history_status = "complete_from_step_zero"

    def observer(self, clip_norm, previous=None):
        from .family_training_core import UpdateObserver
        def after_clip(model, gradient_norm):
            if previous is not None and previous.after_clip is not None:
                previous.after_clip(model, gradient_norm)
            self.completed_updates += 1
            if self.completed_updates <= 1000:
                self.pending.append((self.completed_updates, gradient_norm.detach().clone()))
                if self.completed_updates % 100 == 0:
                    self._flush(clip_norm)
        return UpdateObserver(after_clip=after_clip,
            after_backward=previous.after_backward if previous else None,
            scope=previous.scope if previous else None)

    def _flush(self, clip_norm):
        import torch
        if not self.pending:
            return
        values = torch.stack([value.double().cpu() for _, value in self.pending])
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Nonfinite normalization gradient observation")
        self.gradient_windows.append({"first_step": self.pending[0][0],
            "last_step": self.pending[-1][0], "updates": len(self.pending),
            "preclip_grad_norm_mean": float(values.mean()),
            "preclip_grad_norm_max": float(values.max()),
            "clip_fraction": float((values > clip_norm).double().mean()),
            "clip_norm": clip_norm})
        self.pending = []

    def state_dict(self):
        # Keep partial windows separate so checkpoint cadence does not change bins.
        return deepcopy({"format": "molgap-k1-norm-diagnostics-v1", "variant": self.variant,
            "completed_updates": self.completed_updates, "gradient_windows": self.gradient_windows,
            "pending": [[step, float(value.detach().cpu())] for step, value in self.pending],
            "probes": self.probes, "history_status": self.history_status})

    def restore(self, state, *, optimizer_step):
        import torch
        if state is None:
            self.completed_updates = optimizer_step
            self.gradient_windows, self.pending, self.probes = [], [], []
            self.history_status = "missing_before_resume"
            return
        if (state.get("format") != "molgap-k1-norm-diagnostics-v1"
                or state.get("variant") != self.variant
                or state.get("completed_updates") != optimizer_step):
            raise ValueError("Normalization observation resume identity/step differs")
        restored = deepcopy(state)
        self.completed_updates = optimizer_step
        self.gradient_windows = restored["gradient_windows"]
        self.probes = restored["probes"]
        self.pending = [(step, torch.tensor(value, dtype=torch.float64)) for step,value in restored["pending"]]
        self.history_status = restored["history_status"]

    def probe(self, kernel, loaders, *, completed_pass):
        """Live/eval fixed train prefix; preserve model flags and all RNG state."""
        import torch
        from torch_geometric.loader import DataLoader
        from torch.utils.data import Subset
        from .family_training_core import evaluate_development
        from .training_reproducibility import capture_rng_state, restore_rng_state
        model = kernel.model
        rows = min(4096, loaders.data.train_rows)
        flags = [(module, module.training) for module in model.modules()]
        rng = capture_rng_state()
        handles, summaries, inputs = [], {}, {}
        def add(name, values):
            values = values.detach().double()
            total, count = summaries.get(name, (0.0, 0))
            summaries[name] = (total + float(values.sum().cpu()), count + values.numel())
        def start(index):
            def before(module, args):
                inputs[index] = args[0].detach()
            return before
        def finish(index):
            def after(module, args, output):
                add(f"block{index+1}.input_rms", args[0].square().mean(-1).sqrt())
                add(f"block{index+1}.output_rms", output.square().mean(-1).sqrt())
            return after
        def conv(index):
            def after(module, args, output):
                add(f"block{index+1}.conv_residual_input_ratio",
                    output.norm(dim=-1)/inputs[index].norm(dim=-1).clamp_min(1e-20))
            return after
        def norm3(index, block):
            def before(module, args):
                if block.prenorm:
                    inputs[(index,"ffn")] = args[0].detach()
            return before
        def ffn(index, block):
            def after(module, args, output):
                baseline = inputs[(index,"ffn")] if block.prenorm else args[0]
                add(f"block{index+1}.ffn_residual_input_ratio",
                    output.norm(dim=-1)/baseline.norm(dim=-1).clamp_min(1e-20))
            return after
        def gain(name):
            def before(module, args):
                x = args[0]
                denominator = (x.square().mean(-1)+module.eps).sqrt()
                numerator = ((module.alpha*x+module.beta).square().mean(-1)+module.eps).sqrt()
                add(name+".adaptive_rms", numerator)
                add(name+".adaptive_over_input_rms", numerator/denominator)
            return before
        try:
            model.eval()
            for i,block in enumerate(model.local_blocks):
                handles.extend([block.register_forward_pre_hook(start(i)),
                    block.register_forward_hook(finish(i)),
                    block.conv.register_forward_hook(conv(i)),
                    block.norm3.register_forward_pre_hook(norm3(i,block)),
                    block.mlp.register_forward_hook(ffn(i,block))])
                for name in ("norm1", "norm3"):
                    norm = getattr(block,name)
                    if hasattr(norm,"alpha"):
                        handles.append(norm.register_forward_pre_hook(gain(f"block{i+1}.{name}")))
            batches = DataLoader(Subset(loaders.data.train,range(rows)),
                batch_size=kernel.recipe.batch_size, shuffle=False, num_workers=0,
                generator=torch.Generator().manual_seed(kernel.recipe.seed))
            evaluation = evaluate_development(model,batches,
                forward=lambda model,batch: kernel.forward(batch,model=model), rows=rows,
                source_start=loaders.data.train_source_start,
                mean=kernel.recipe.target_mean_eV,std=kernel.recipe.target_std_eV)
            observed = {key: total/count for key,(total,count) in summaries.items()}
            if not all(math.isfinite(value) for value in observed.values()):
                raise ValueError("Nonfinite normalization probe")
            self.probes.append({"completed_pass": completed_pass,"optimizer_step":self.completed_updates,
                "weights":"live","mode":"eval","role":"train","rows":rows,
                "source_start":loaders.data.train_source_start,"train_eval_mae_eV":evaluation["mae_eV"],
                "layer_means":observed,
                "ada_parameters": {f"block{i+1}.{name}": {
                    field+"_rms": float(getattr(getattr(block,name),field).detach().double().square().mean().sqrt().cpu())
                    for field in ("alpha","beta","weight")}
                    for i,block in enumerate(model.local_blocks) for name in ("norm1","norm3")
                    if hasattr(getattr(block,name),"alpha")}})
        finally:
            for handle in handles:
                handle.remove()
            for module, training in flags:
                module.training = training
            restore_rng_state(rng)
