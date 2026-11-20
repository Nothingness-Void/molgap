"""Persistent CPU input pools for the common trainer, not platform loaders."""
from __future__ import annotations

from .family_training_data import PassBatchSampler, TrainingData
from .family_training_recipe import FamilyRecipe


def training_loader_options(workers):
    if type(workers) is not int or workers < 0:
        raise ValueError("Nonnegative integer worker count required")
    return {"num_workers": workers, "persistent_workers": workers > 0,
            "pin_memory": True, **({"prefetch_factor": 2} if workers else {})}


class FamilyLoaders:
    """Own one immutable train view and an optional bounded development cache."""
    def __init__(self, recipe: FamilyRecipe, data: TrainingData, tuning, *, device_inputs=None):
        import torch
        from torch_geometric.loader import DataLoader
        from .gptrans_execution_backend import PreparedLoader, assert_static_dataset
        from .gptrans_execution_tuning import ExecutionTuning, PreparedDevelopmentCache

        if type(recipe) is not FamilyRecipe or type(data) is not TrainingData or type(tuning) is not ExecutionTuning:
            raise TypeError("Expected typed recipe, accepted data and execution resources")
        if (data.manifest_sha256 != recipe.dataset.manifest_sha256 or
                data.train_rows != recipe.train_rows or data.development_rows != recipe.development_rows):
            raise ValueError("Loaded data differs from the frozen recipe selection")
        assert_static_dataset(data.train)
        self.recipe, self.data, self.tuning = recipe, data, tuning
        self.sampler = PassBatchSampler(data.train_rows, recipe.batch_size, recipe.seed,
            algorithm=recipe.sampler, drop_last=recipe.drop_last)
        self.generator = torch.Generator()
        self.training = DataLoader(data.train, batch_sampler=self.sampler,
            generator=self.generator, **training_loader_options(tuning.training_workers))
        from .family_training_policy import FamilyModePolicy, mode_policy
        policy = (FamilyModePolicy() if recipe._frozen_read else mode_policy(recipe.family, recipe.mode))
        extra_fields = policy.extra_batch_fields
        self.device_train, self.device_inputs_reason = None, "off" if device_inputs is None else None
        if hasattr(data.train, "train_stream"):  # compact GPU-resident cells (demol_cin_resident)
            data.train.store.upload("cuda")  # raises when it does not fit; never falls back
            self.device_train, self.device_inputs_reason = data.train, None
        elif device_inputs is not None:
            from .family_training_device_inputs import DeviceResidentTrain
            if policy.prepare_batch is not None or extra_fields:
                self.device_inputs_reason = "mode has a prepare_batch hook"
            else:
                store = DeviceResidentTrain(data.train, data.train_rows)
                if DeviceResidentTrain.admit(store.estimate, device_inputs):
                    self.device_train = store.upload(device_inputs)
                else:
                    self.device_inputs_reason = "device memory or CUDA unavailable for %d bytes" % store.estimate
        self.development = None
        self.development_cache = None
        self.closed = False
        if hasattr(data.development, "sequential_stream"):
            data.development.store.upload("cuda")
            self.development_cache = data.development.sequential_stream(recipe.batch_size)
        elif data.development is not None:
            assert_static_dataset(data.development)
            self.development = DataLoader(data.development, batch_size=recipe.batch_size,
                shuffle=False, generator=torch.Generator().manual_seed(recipe.seed),
                **training_loader_options(tuning.development_workers))
            source = PreparedLoader(self.development) if recipe.family == "gptrans_t" else self.development
            self.development_cache = PreparedDevelopmentCache(source, rows=data.development_rows,
                batch_size=recipe.batch_size, max_bytes=tuning.development_cache_bytes,
                source_start=data.development_source_start, retain_model_inputs=True,
                retain_random_walk_pe=recipe.family == "neural_atom_k1", extra_fields=extra_fields)

    def train(self, pass_index, start_batch):
        if self.closed:
            raise RuntimeError("Training loader pool is closed")
        # Use the common validator before mutating the retained sampler cursor.
        validated = PassBatchSampler(self.data.train_rows, self.recipe.batch_size, self.recipe.seed,
            algorithm=self.recipe.sampler, pass_index=pass_index, start_batch=start_batch,
            drop_last=self.recipe.drop_last)
        self.sampler.pass_index, self.sampler.start_batch = validated.pass_index, validated.start_batch
        self.generator.manual_seed(self.recipe.seed + pass_index)
        if hasattr(self.device_train, "train_stream"):
            return self.device_train.train_stream(self.sampler)
        if self.device_train is not None:
            from .family_training_device_inputs import device_training_stream
            return device_training_stream(self.device_train, self.sampler)
        return self.training

    def dev(self):
        if self.closed or self.development_cache is None:
            raise ValueError("No selected development role in this loader pool")
        return self.development_cache

    def calibration(self, policy):
        """Independent ordered train-only stream; never advances training RNG/cursor."""
        import torch
        from torch_geometric.loader import DataLoader
        if self.closed or policy != self.recipe.clean_bn:
            raise ValueError("Calibration requires the frozen selected training policy")
        return DataLoader(self.data.train, batch_size=policy.batch_size, shuffle=False,
            drop_last=True, num_workers=0, generator=torch.Generator().manual_seed(self.recipe.seed))

    def close(self):
        if self.closed:
            return
        self.closed = True
        for loader in (self.training, self.development):
            iterator = getattr(loader, "_iterator", None) if loader is not None else None
            if iterator is not None:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if shutdown is not None:
                    shutdown()
                loader._iterator = None
        if self.development_cache is not None:
            self.development_cache.clear()
        if hasattr(self.device_train, "train_stream"):
            self.device_train.store.release()
