"""Portable execution primitives for V4 reference screens."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import tarfile
from collections.abc import Mapping
from pathlib import Path
from pathlib import PurePosixPath

from .training_reproducibility import sha256_file


V4_RUNTIME_FORMAT = "molgap-v4-runtime-primitives-v1"


def torch_load_compat(path: Path, *, map_location="cpu", weights_only: bool = False):
    """Load a trusted artifact on both legacy and current PyTorch."""
    import torch

    kwargs = {"map_location": map_location}
    signature_known = True
    try:
        parameters = inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        signature_known = False
        parameters = {}
    if "weights_only" in parameters:
        kwargs["weights_only"] = weights_only
    elif not signature_known:
        try:
            return torch.load(path, **kwargs, weights_only=weights_only)
        except TypeError as error:
            message = str(error)
            if "unexpected keyword argument" not in message or "weights_only" not in message:
                raise
    return torch.load(path, **kwargs)


def sample_std_compat(values, *, correction: int = 1):
    """Return a sample standard deviation without requiring the new API."""
    if correction not in (0, 1):
        raise ValueError("Legacy-compatible V4 statistics support correction 0 or 1")
    return values.std(unbiased=bool(correction))


def validate_adamw_mode(*, fused: bool) -> dict:
    """Check optimizer mode support before loading large graph assets."""
    import torch

    signature_known = True
    try:
        supported = inspect.signature(torch.optim.AdamW).parameters
    except (TypeError, ValueError):
        signature_known = False
        supported = {}
    if fused and (not signature_known or "fused" not in supported):
        raise RuntimeError("This runtime cannot satisfy a fused-AdamW contract")
    return {
        "fused": bool(fused),
        "fused_keyword_supported": "fused" in supported,
        "foreach_keyword_supported": "foreach" in supported,
    }


def make_adamw_compat(parameters, *, fused: bool = False, **kwargs):
    """Construct AdamW without passing unsupported execution keywords."""
    import torch

    support = validate_adamw_mode(fused=fused)
    signature_known = True
    try:
        supported = inspect.signature(torch.optim.AdamW).parameters
    except (TypeError, ValueError):
        signature_known = False
        supported = {}
    options = dict(kwargs)
    if support["foreach_keyword_supported"]:
        options.setdefault("foreach", False)
    if fused:
        options["fused"] = True
    elif support["fused_keyword_supported"]:
        options["fused"] = False
    unsupported = set(options) - set(supported) if signature_known else set()
    if unsupported:
        raise RuntimeError(
            f"AdamW runtime does not support requested options: {sorted(unsupported)}"
        )
    return torch.optim.AdamW(parameters, **options)


def state_dict_sha256(state_dict: Mapping) -> str:
    """Hash tensor values independently of serialization format."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8") + b"\0")
        tensor = value.detach().cpu().contiguous()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model) -> str:
    return state_dict_sha256(model.state_dict())


def normalized_source_sha256(path: Path) -> str:
    """Hash text source semantics independently of checkout line endings."""
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_initial_state(
    model,
    path: Path,
    *,
    expected_file_sha256: str,
    expected_state_sha256: str,
    expected_format: str,
):
    """Load and verify a framework-independent frozen initialization."""
    if sha256_file(path) != expected_file_sha256:
        raise RuntimeError("Frozen initial-state artifact changed")
    payload = torch_load_compat(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != expected_format:
        raise RuntimeError("Frozen initial-state artifact format changed")
    if "model_state" not in payload:
        raise RuntimeError("Frozen initial-state artifact has no model_state")
    model.load_state_dict(payload["model_state"], strict=True)
    observed = model_state_sha256(model)
    if payload.get("state_sha256") != observed or observed != expected_state_sha256:
        raise RuntimeError("Frozen initial-state tensor identity changed")
    return {
        "format": expected_format,
        "file_sha256": expected_file_sha256,
        "state_sha256": observed,
    }


def certify_numerical_repeatability(
    *,
    losses,
    states,
    maximum_loss_delta: float = 1.0e-7,
    maximum_parameter_delta: float = 1.0e-7,
) -> dict:
    """Accept deterministic GPU runs with bounded floating-point reduction noise."""
    import torch

    if len(losses) != 2 or len(states) != 2:
        raise ValueError("V4 repeatability calibration requires exactly two repeats")
    left_names = set(states[0])
    right_names = set(states[1])
    if left_names != right_names:
        raise RuntimeError("Repeat state dictionaries have different keys")
    if any(not math.isfinite(float(value)) for value in losses):
        raise RuntimeError("Repeatability losses contain non-finite values")

    max_name = ""
    max_delta = 0.0
    for name in sorted(left_names):
        left = states[0][name].detach().cpu().contiguous()
        right = states[1][name].detach().cpu().contiguous()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"Repeat tensor metadata changed: {name}")
        if left.is_floating_point():
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                raise RuntimeError(f"Repeat state contains non-finite values: {name}")
            delta = float((left - right).abs().max()) if left.numel() else 0.0
            if delta > max_delta:
                max_name = name
                max_delta = delta
        elif not bool(torch.equal(left, right)):
            raise RuntimeError(f"Non-floating repeat state changed: {name}")

    loss_delta = abs(float(losses[0]) - float(losses[1]))
    accepted = (
        loss_delta <= maximum_loss_delta
        and max_delta <= maximum_parameter_delta
    )
    result = {
        "accepted": accepted,
        "losses": [float(value) for value in losses],
        "loss_delta": loss_delta,
        "maximum_loss_delta": maximum_loss_delta,
        "state_sha256": [state_dict_sha256(state) for state in states],
        "max_parameter_delta": max_delta,
        "maximum_parameter_delta": maximum_parameter_delta,
        "max_parameter_name": max_name,
    }
    if not accepted:
        raise RuntimeError(
            "Seeded optimizer-step calibration exceeds numerical tolerance: "
            + json.dumps(result, sort_keys=True)
        )
    return result


def validate_standard_source_bundle(
    source_archive: Path, source_archive_sha256: str, source_commit: str
) -> str:
    """Validate a standard V4 source bundle and all sidecars."""
    source_archive = source_archive.resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    if sha256_file(source_archive) != source_archive_sha256:
        raise RuntimeError("Source archive identity changed")
    commit_path = source_archive.with_name("SOURCE_COMMIT.txt")
    sha_path = source_archive.with_name("SOURCE_ARCHIVE_SHA256.txt")
    inventory_path = source_archive.with_name("SOURCE_FILES.json")
    for path in (commit_path, sha_path, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if commit_path.read_text(encoding="utf-8").strip() != source_commit:
        raise RuntimeError("Source commit sidecar changed")
    if sha_path.read_text(encoding="utf-8").strip() != source_archive_sha256:
        raise RuntimeError("Source archive SHA sidecar changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("format") not in (None, "molgap-v4-source-inventory-v1"):
        raise RuntimeError("Source inventory format changed")
    if inventory.get("source_commit") not in (None, source_commit):
        raise RuntimeError("Source inventory commit identity changed")
    expected = {item["path"]: item["sha256"] for item in inventory["files"]}
    if len(expected) != len(inventory["files"]):
        raise RuntimeError("Source inventory contains duplicate paths")
    observed: dict[str, str] = {}
    with tarfile.open(source_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"Unsupported source archive entry: {member.name}")
            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Unsafe source archive path: {member.name}")
            if member.name in observed:
                raise RuntimeError(f"Duplicate source archive path: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Cannot read source member: {member.name}")
            normalized = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                normalized.update(block)
            observed[member.name] = normalized.hexdigest()
    if observed != expected:
        raise RuntimeError("Source archive file inventory changed")
    return source_archive_sha256
