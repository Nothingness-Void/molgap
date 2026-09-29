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
