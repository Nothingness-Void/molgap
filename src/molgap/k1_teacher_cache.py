"""Pinned train-only K1 teacher predictions, joined by CPU source indices."""
from __future__ import annotations

import json
from pathlib import Path

from .training_reproducibility import sha256_file


FORMAT = "molgap-k1-teacher-cache-v1"
TRAIN_ROWS = 100_000
FIXED_MANIFEST_SHA256 = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
DISTILL_WEIGHTS = {"distill_weak": 0.1, "distill_strong": 1.0}


def validate_teacher_config(config, mode):
    """Keep the two registered strengths and teacher/cache pins exact."""
    if type(config) is not dict or set(config) != {"weight", "teacher_identity", "cache_manifest_sha256"}:
        raise ValueError("Expected exact K1 teacher cache configuration")
    if (mode not in DISTILL_WEIGHTS or type(config["weight"]) not in (int, float)
            or config["weight"] != DISTILL_WEIGHTS[mode]):
        raise ValueError("K1 distillation weight differs from frozen mode")
    from .experiment_spec import _digest
    for field in ("teacher_identity", "cache_manifest_sha256"):
        _digest(config[field], "teacher_cache." + field)
    return dict(config)


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate teacher manifest key")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("Nonfinite teacher manifest number: " + value)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=reject_constant)


class K1TeacherCache:
    """The payload cannot contain development predictions or targets."""

    def __init__(self, input_root, *, manifest_sha256, teacher_identity):
        import torch
        from .experiment_spec import _digest
        _digest(manifest_sha256, "cache_manifest_sha256")
        _digest(teacher_identity, "teacher_identity")
        root = Path(input_root).resolve(strict=True)
        matches = []
        for path in root.rglob("manifest.json"):
            # The independently declared digest selects the exact manifest bytes.
            if sha256_file(path) == manifest_sha256:
                matches.append(path)
        if len(matches) != 1:
            raise ValueError("Expected exactly one pinned K1 teacher cache manifest")
        path = matches[0].resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Teacher manifest escapes input root")
        manifest = _json(path)
        interval = manifest.get("source_idx_range") if type(manifest) is dict else None
        if (type(manifest) is not dict or manifest.get("format") != FORMAT
                or manifest.get("teacher_identity") != teacher_identity
                or manifest.get("role") != "internal_training"
                or type(interval) is not list or any(type(index) is not int for index in interval)
                or interval != [0, TRAIN_ROWS]
                or manifest.get("dataset_manifest_sha256") != FIXED_MANIFEST_SHA256):
            raise ValueError("Teacher cache role/teacher/dataset identity mismatch")
        if manifest.get("path") != "teacher_predictions.pt":
            raise ValueError("Expected fixed teacher prediction payload filename")
        _digest(manifest.get("sha256"), "teacher_payload.sha256")
        payload_path = (path.parent / manifest["path"]).resolve(strict=True)
        if not payload_path.is_relative_to(path.parent):
            raise ValueError("Teacher payload escapes manifest directory")
        if sha256_file(payload_path) != manifest["sha256"]:
            raise ValueError("Teacher prediction payload hash mismatch")
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        if type(payload) is not dict or set(payload) != {"source_idx", "prediction_eV"}:
            raise ValueError("Expected exact teacher prediction payload fields")
        indices, predictions = payload["source_idx"], payload["prediction_eV"]
        if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64
                or indices.shape != (TRAIN_ROWS,)
                or not torch.equal(indices, torch.arange(TRAIN_ROWS, dtype=torch.int64))
                or not isinstance(predictions, torch.Tensor) or predictions.dtype != torch.float32
                or predictions.shape != (TRAIN_ROWS,) or not bool(torch.isfinite(predictions).all())
                or predictions.requires_grad):
            raise ValueError("Teacher payload requires finite float32 predictions for ascending train indices")
        self.prediction_eV = predictions.detach().clone()
        self.manifest = manifest
        self.identity = manifest_sha256
        self.teacher_identity = teacher_identity

    def attach(self, batch):
        import torch
        indices = getattr(batch, "source_idx", None)
        if (not isinstance(indices, torch.Tensor) or indices.device.type != "cpu"
                or indices.dtype != torch.int64):
            raise ValueError("Teacher join requires CPU int64 source indices")
        indices = indices.view(-1)
        if (indices.numel() == 0 or bool((indices < 0).any())
                or bool((indices >= TRAIN_ROWS).any())
                or indices.unique().numel() != indices.numel()):
            raise ValueError("Teacher batch contains duplicate or non-training indices")
        batch.teacher_eV = self.prediction_eV[indices].clone()
        return batch


def load_teacher_cache(input_root, config, *, mode):
    config = validate_teacher_config(config, mode)
    return K1TeacherCache(input_root, manifest_sha256=config["cache_manifest_sha256"],
                          teacher_identity=config["teacher_identity"])
