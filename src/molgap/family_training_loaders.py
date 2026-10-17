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
    def __init__(self, recipe: FamilyRecipe, data: TrainingData, tuning):
        import torch
        from torch_geometric.loader import DataLoader
        from .gptrans_execution_backend import PreparedLoader, assert_static_dataset
        from .gptrans_execution_tuning import ExecutionTuning, PreparedDevelopmentCache

        if type(recipe) is not FamilyRecipe or type(data) is not TrainingData or type(tuning) is not ExecutionTuning:
            raise TypeError("Expected typed recipe, accepted data and execution resources")
        if recipe.cpu_synthetic_only and (data.synthetic is not True or
                type(data.train) not in (list, tuple) or
                (data.development is not None and type(data.development) not in (list, tuple))):
            raise ValueError("CPU family cannot load real/packed/protected datasets")
        if data.indexed and not getattr(recipe, "integration_preview", False):
            raise ValueError("Indexed data requires its explicitly versioned integration interface")
        if getattr(recipe, "integration_preview", False):
            if (len(data.train) != data.train_rows or
                    (data.development is not None and len(data.development) != data.development_rows)):
                raise ValueError("Integration input decoded row count differs from its declaration")
            if recipe.dataset.manifest.startswith("synthetic/"):
                if data.synthetic is not True:
                    raise ValueError("Integration fixture cannot substitute real data")
            else:
                from .data.indexed_inputs import AcceptedIndexedInputPermit, IndexedRoleDataset
                selected = (data.train, data.development) if recipe.dataset.development else (data.train,)
                if data.synthetic or not data.indexed or any(
                        type(dataset) is not IndexedRoleDataset or
                        type(dataset.permit) is not AcceptedIndexedInputPermit for dataset in selected):
                    raise ValueError("Real integration input requires accepted indexed role datasets, not an unverified list/cache")
                if (data.train.permit.role != ("train_100k" if recipe.train_rows == 100000 else "train_500k") or
                        (recipe.dataset.development and data.development.permit.role != "common_dev")):
                    raise ValueError("Real integration data role differs from the explicit scale")
        if data.indexed and (data.train_source_start is not None or data.development_source_start is not None or
                data.train_source_indices is None or len(data.train_source_indices) != data.train_rows or
                (data.development is not None and (data.development_source_indices is None or
                    len(data.development_source_indices) != data.development_rows))):
            raise ValueError("Indexed data cannot be represented as a contiguous source range")
        if data.indexed:
            from .data.indexed_inputs import IndexedRoleDataset
            for dataset, indices in ((data.train, data.train_source_indices),
                                     (data.development, data.development_source_indices)):
                if dataset is None:
                    continue
                if (type(indices) is not tuple or any(type(i) is not int or i < 0 for i in indices) or
                        any(a >= b for a, b in zip(indices, indices[1:])) or
                        (type(dataset) is IndexedRoleDataset and indices != dataset.source_indices)):
                    raise ValueError("Indexed input members differ from the verified dataset")
        if (data.manifest_sha256 != recipe.dataset.manifest_sha256 or
                data.train_rows != recipe.train_rows or data.development_rows != recipe.development_rows):
            raise ValueError("Loaded data differs from the frozen recipe selection")
        assert_static_dataset(data.train)
        self.recipe, self.data, self.tuning = recipe, data, tuning
        self.feature_bindings = None
        if getattr(recipe, "integration_preview", False) and data.indexed:
            from .data.indexed_inputs import IndexedRoleDataset
            if type(data.train) is IndexedRoleDataset:
                from .modern2d_feature_binding import build_indexed_feature_binding
                self.feature_bindings = {role: build_indexed_feature_binding(dataset,
                    family=recipe.interface, configuration=recipe.model_config) for role, dataset in
                    (("train", data.train), ("development", data.development)) if dataset is not None}
        self.sampler = PassBatchSampler(data.train_rows, recipe.batch_size, recipe.seed,
            algorithm=recipe.sampler, drop_last=recipe.drop_last)
        self.generator = torch.Generator()
        self.training = DataLoader(data.train, batch_sampler=self.sampler,
            generator=self.generator, **training_loader_options(tuning.training_workers))
        self.development = None
        self.development_cache = None
        self.closed = False
        if data.development is not None:
            assert_static_dataset(data.development)
            self.development = DataLoader(data.development, batch_size=recipe.batch_size,
                shuffle=False, generator=torch.Generator().manual_seed(recipe.seed),
                **training_loader_options(tuning.development_workers))
            source = PreparedLoader(self.development) if recipe.family == "gptrans_t" else self.development
            self.development_cache = None if recipe.cpu_synthetic_only or data.indexed or getattr(recipe, "integration_preview", False) else PreparedDevelopmentCache(source, rows=data.development_rows,
                batch_size=recipe.batch_size, max_bytes=tuning.development_cache_bytes,
                source_start=data.development_source_start, retain_model_inputs=True,
                retain_random_walk_pe=recipe.family == "neural_atom_k1")

    def train(self, pass_index, start_batch):
        if self.closed:
            raise RuntimeError("Training loader pool is closed")
        # Use the common validator before mutating the retained sampler cursor.
        validated = PassBatchSampler(self.data.train_rows, self.recipe.batch_size, self.recipe.seed,
            algorithm=self.recipe.sampler, pass_index=pass_index, start_batch=start_batch,
            drop_last=self.recipe.drop_last)
        self.sampler.pass_index, self.sampler.start_batch = validated.pass_index, validated.start_batch
        self.generator.manual_seed(self.recipe.seed + pass_index)
        return self.training

    def dev(self):
        if not self.closed and (self.recipe.cpu_synthetic_only or self.data.indexed or
                getattr(self.recipe, "integration_preview", False)) and self.development is not None:
            return self.development
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
