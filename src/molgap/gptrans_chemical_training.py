"""Chemical-objective bindings for the existing GPTrans V4 trainer."""
from __future__ import annotations

from .gptrans_objective import GPTransObjectiveConfig, export_gap_state_dict


class ChemicalTrainingAddon:
    def __init__(self, config: GPTransObjectiveConfig, cache=None):
        if config.enabled and cache is None:
            raise ValueError("Enabled chemical supervision requires an accepted label cache")
        if not config.enabled and cache is not None:
            raise ValueError("Control arm must not attach an auxiliary cache")
        self.config, self.cache = config, cache
        self.identity = {"objective": config.to_dict(), "objective_sha256": config.identity,
                         "label_cache": None if cache is None else cache.identity}

    def scientific_fields(self, base):
        return {**base, "loss_fingerprint": ("normalized-gap-l1+chemical-aux:" + self.config.identity
                                             if self.config.enabled else base["loss_fingerprint"])}

    def validate_graphs(self, graphs):
        if self.cache is None:
            return
        # V4 graph IDs are the frozen role-local source_idx, not OGB official IDs.
        from .pcqm_gptrans_v4 import TRAIN_ROWS
        if len(graphs) != TRAIN_ROWS or set(self.cache.positions) != set(range(TRAIN_ROWS)):
            raise ValueError("Auxiliary cache does not cover exact V4 training indices")
        if self.cache.manifest["role"]["row_identity_semantics"] != "pcqm-fixed100k-v4-source_idx":
            raise ValueError("Auxiliary cache uses a different source-index coordinate system")
        if self.cache.manifest["role"]["dataset_identity"] != "pcqm-fixed100k-v4":
            raise ValueError("Auxiliary cache is bound to a different dataset")

    def attach(self, batch):
        return batch if self.cache is None else self.cache.attach(batch)

    @staticmethod
    def export_state(state):
        return export_gap_state_dict(state)


def profile_training_overhead(*, addon, dataset_root, manifest_path, initial_state_path,
                              warmup=8, repeats=20):
    """Short alternating same-device profile; never a trained accuracy control."""
    import statistics
    import time
    import torch
    from .pcqm_gptrans_v4 import (
        validate_fixed_assets, _load_datasets, _target_stats, _training_loader,
        _make_training_state, _optimizer_step, _forward, configure_fp32_determinism,
    )
    configure_fp32_determinism(42)
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    graphs, shards = _load_datasets(assets.train_paths)
    addon.validate_graphs(graphs)
    mean, std = _target_stats(shards)
    batch = addon.attach(next(iter(_training_loader(graphs, 0)))).to("cuda")
    states = [_make_training_state(initial_state_path),
              _make_training_state(initial_state_path, objective_config=addon.config)]
    for model, _, _, _ in states:
        model.eval()
    with torch.no_grad():
        torch.testing.assert_close(_forward(states[0][0], batch), _forward(states[1][0], batch), rtol=0, atol=0)
    durations = [[], []]
    for turn in range(warmup + repeats):
        for index in ((0, 1) if turn % 2 == 0 else (1, 0)):
            model, optimizer, scheduler, ema = states[index]
            model.train(); scheduler.step(0)
            torch.cuda.synchronize(); started = time.perf_counter()
            _optimizer_step(model, optimizer, ema, batch, mean, std, check_finite=True,
                            objective=getattr(model, "_training_objective", None))
            torch.cuda.synchronize()
            if turn >= warmup:
                durations[index].append(time.perf_counter() - started)
    medians = [statistics.median(values) for values in durations]
    ratio = medians[1] / medians[0]
    return {"format": "molgap-chemical-objective-profile-v1", "objective_sha256": addon.config.identity,
            "scope": "gpu-resident-optimizer-step-only", "end_to_end_wall_overhead_qualified": False,
            "warmup": warmup, "repeats": repeats, "order": "alternating-same-device",
            "hardware": torch.cuda.get_device_name(0), "initial_gap_equivalence": True,
            "baseline_step_seconds": durations[0], "candidate_step_seconds": durations[1],
            "median_step_ratio": ratio, "maximum_step_ratio": 1.05, "accepted": ratio <= 1.05}
