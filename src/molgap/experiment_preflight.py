"""Versioned, package-isolated real-data diagnostics, without evidence authority.

This file is also the stdlib-only isolated worker launcher. Family imports are
deliberately local: the worker must not import the caller's installed molgap.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experiment_spec import ExperimentSpec

__all__ = ["run_real_shard_preflight", "validate_real_shard_manifest"]
MANIFEST_FORMAT = "molgap-real-shard-manifest-v1"
REPORT_FORMAT = "molgap-real-shard-preflight-v1"
MANIFEST_NAME = "real_shard_manifest.json"
CHECKS = (
    "package_verified", "shard_verified", "loader_batch_built", "forward_checked",
    "backward_checked", "optimizer_step_checked", "checkpoint_roundtrip_checked",
)
SUCCESS = "REAL_SHARD_PREFLIGHT_PASSED"
STATUSES = frozenset({
    SUCCESS, "PACKAGE_REJECTED", "SHARD_REJECTED", "MISSING_REAL_SHARD",
    "NON_REAL_SHARD", "UNSUPPORTED_FAMILY_PREFLIGHT", "UNSUPPORTED_INITIALIZATION",
    "ENVIRONMENT_BLOCKED", "IMPORT_ORIGIN_REJECTED", "FAILED", "TIMED_OUT",
})
LIMITATIONS = [
    "Bounded diagnostic only; not training, scientific acceptance or platform GPU acceptance.",
    "No runtime certificate, role-history admission, terminal or research-evidence authority.",
    "Source and authorization declarations must be reviewed; hashes are not signatures.",
    "Only the published GPTrans V4 100K/50K packed assets have an executable v1 loader.",
    "Checkpoint check is an ephemeral tensor/optimizer roundtrip, not trainer resume.",
]


class PreflightBlocked(ValueError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(raw: bytes):
    value = json.loads(raw, object_pairs_hook=_unique)
    if _json(value) != raw:
        raise ValueError("Expected strict canonical JSON bytes")
    return value


def _fields(value, fields: str):
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError(f"Expected exactly fields: {fields}")


def _digest(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA256")


def _name(value: str) -> str:
    if (type(value) is not str or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive):
        raise ValueError("Expected portable relative POSIX path")
    for part in value.split("/"):
        if (part in {"", ".", ".."} or part.endswith((".", " "))
                or any(ord(c) < 32 or c in '<>"|?*' for c in part)
                or part.split(".")[0].upper() in {
                    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
                    *(f"LPT{i}" for i in range(10)),
                }):
            raise ValueError(f"Unsafe path: {value!r}")
    return value


def _no_links(path: Path):
    for item in (path, *path.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/junction/reparse paths are forbidden")


def _regular(root: Path, name: str) -> Path:
    path = root / _name(name)
    _no_links(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escaped root")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Expected an unlinked regular file")
    return path


def _atomic(path: Path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validated(spec):
    from .experiment_spec import ExperimentSpec
    if type(spec) is not ExperimentSpec or set(vars(spec)) != {"_canonical_json"}:
        raise TypeError("Expected exact ExperimentSpec snapshot")
    rebuilt = ExperimentSpec.from_json(spec.to_json())
    if rebuilt.to_json() != spec.to_json() or rebuilt.identity != spec.identity:
        raise ValueError("Invalid spec snapshot")
    return rebuilt


def _entry(entry, *, role=False):
    _fields(entry, "path sha256 bytes role" if role else "path sha256 bytes")
    _name(entry["path"])
    _digest(entry["sha256"])
    if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
        raise ValueError("Expected positive integer file size")


def _manifest(raw: bytes, spec) -> dict:
    manifest = _load(raw)
    _fields(manifest, "format spec_identity arms")
    if manifest["format"] != MANIFEST_FORMAT or manifest["spec_identity"] != spec.identity:
        raise ValueError("Manifest format/spec identity mismatch")
    arms = spec.to_dict()["arms"]
    if type(manifest["arms"]) is not list or len(manifest["arms"]) != len(arms):
        raise ValueError("Manifest must bind every arm in spec order")
    from .screen_policy import canonical_fingerprint
    for record, arm in zip(manifest["arms"], arms, strict=True):
        _fields(record, "arm_id arm_identity data provenance files")
        if record["arm_id"] != arm["arm_id"] or record["arm_identity"] != canonical_fingerprint(arm):
            raise ValueError("Manifest arm identity mismatch")
        if _json(record["data"]) != _json(arm["data"]):
            raise ValueError("Dataset/split/feature/target/role identity mismatch")
        roles = {entry["role"] for entry in arm["data"]["roles"]}
        if not roles <= {"train", "development"}:
            raise ValueError("Protected role is forbidden")
        provenance = record["provenance"]
        _fields(provenance, "kind native_manifest")
        if provenance["kind"] not in ("real-packed-shard", "test-only-synthetic"):
            raise ValueError("Unknown shard provenance kind")
        native = provenance["native_manifest"]
        if native is not None:
            _entry(native)
        if type(record["files"]) is not list or not record["files"]:
            raise ValueError("No shard files declared")
        seen = set()
        observed_roles = set()
        for entry in record["files"]:
            _entry(entry, role=True)
            if type(entry["role"]) is not str or entry["role"] not in roles:
                raise ValueError("Protected or unauthorized shard role")
            name = entry["path"].casefold()
            if name in seen or name == MANIFEST_NAME.casefold():
                raise ValueError("Duplicate/reserved shard path")
            seen.add(name)
            observed_roles.add(entry["role"])
        if observed_roles != roles:
            raise ValueError("Missing authorized role")
        if native is not None and native["path"].casefold() in seen | {MANIFEST_NAME.casefold()}:
            raise ValueError("Native manifest overlaps shard paths")
    return manifest


def _verify_files(record, root: Path):
    entries = list(record["files"])
    if record["provenance"]["native_manifest"] is not None:
        entries.append(record["provenance"]["native_manifest"])
    for entry in entries:
        path = _regular(root, entry["path"])
        if path.stat().st_size != entry["bytes"] or _file_sha(path) != entry["sha256"]:
            raise ValueError(f"Shard size/SHA mismatch: {entry['path']}")


def validate_real_shard_manifest(
    spec: ExperimentSpec, shard_root: Path, *, manifest_path: Path | None = None,
) -> dict:
    """Validate canonical declarations and all file bytes, not scientific provenance.

    The default manifest is shard_root/real_shard_manifest.json. An explicit
    manifest must also be inside shard_root. Missing inputs raise FileNotFoundError;
    malformed/unauthorized inputs raise ValueError. No model or pickle is loaded.
    """
    spec = _validated(spec)
    root = Path(shard_root).absolute()
    _no_links(root)
    path = Path(manifest_path).absolute() if manifest_path is not None else root / MANIFEST_NAME
    _no_links(path)
    path = _regular(root, path.relative_to(root).as_posix())
    raw = path.read_bytes()
    manifest = _manifest(raw, spec)
    for record in manifest["arms"]:
        _verify_files(record, root)
    return {"manifest": manifest, "manifest_sha256": _sha(raw)}


def _extract(package: Path, root: Path, manifest: dict) -> dict:
    """Extract regular inventory-bound payloads only, without extractall."""
    inventory = json.loads((package / "SOURCE_FILES.json").read_bytes())
    expected = {entry["path"]: entry for entry in inventory["files"]}
    seen = set()
    folded = set()
    with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
        for member in archive:
            name = _name(member.name)
            if (not member.isfile() or member.linkname or member.issparse()
                    or name in seen or name.casefold() in folded or name not in expected):
                raise ValueError("Unsafe/duplicate archive member")
            entry = expected[name]
            if member.size != entry["bytes"]:
                raise ValueError("Archive size mismatch")
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Archive traversal")
            _no_links(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Missing archive stream")
            with source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
            if _file_sha(path) != entry["sha256"]:
                raise ValueError("Extracted source digest mismatch")
            seen.add(name)
            folded.add(name.casefold())
    if seen != set(manifest["relative_allowlist"]) or seen != set(expected):
        raise ValueError("Extracted inventory mismatch")
    return expected


class _PackageOnly(importlib.abc.MetaPathFinder):
    """Reject a missing package module instead of falling through to site-packages."""
    def __init__(self, root: Path, inventory: dict):
        self.root = root.resolve()
        self.inventory = inventory

    def check(self, origin):
        if not origin:
            raise PreflightBlocked("IMPORT_ORIGIN_REJECTED", "Namespace/unknown molgap origin")
        path = Path(origin).resolve()
        if not path.is_relative_to(self.root):
            raise PreflightBlocked("IMPORT_ORIGIN_REJECTED", "molgap import escaped extracted package")
        relative = path.relative_to(self.root).as_posix()
        entry = self.inventory.get(relative)
        if entry is None or _file_sha(path) != entry["sha256"]:
            raise PreflightBlocked("IMPORT_ORIGIN_REJECTED", "molgap import is not an inventoried source")
        return {"path": relative, "sha256": entry["sha256"]}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "molgap" and not fullname.startswith("molgap."):
            return None
        locations = [str(self.root / "src")] if fullname == "molgap" else path
        found = importlib.machinery.PathFinder.find_spec(fullname, locations)
        if found is None:
            raise PreflightBlocked("UNSUPPORTED_FAMILY_PREFLIGHT", f"Frozen package lacks {fullname}")
        self.check(found.origin)
        return found

    def audit(self):
        origins = {}
        for name, module in tuple(sys.modules.items()):
            if name == "molgap" or name.startswith("molgap."):
                origins[name] = self.check(getattr(module, "__file__", None))
                self.check(getattr(getattr(module, "__spec__", None), "origin", None))
        return origins


def _base(spec, arm, device, package_binding, shard_digest):
    from .screen_policy import canonical_fingerprint
    return {
        "format": REPORT_FORMAT, "spec_identity": spec.identity,
        "arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm),
        "package": package_binding, "shard_manifest_sha256": shard_digest,
        "requested_device": device, "observed_device": None,
        "status": "FAILED", "reason": "Not executed",
        "observed": {key: False for key in CHECKS}, "import_origins": {},
        "batches_checked": 0, "limitations": list(LIMITATIONS), "missing_evidence": [],
    }


def _finish(result):
    result["missing_evidence"] = [key for key, value in result["observed"].items() if not value]
    result["missing_evidence"].extend(["target_platform_acceptance", "scientific_acceptance"])
    return result


def _native_assets(record, root, family):
    native = record["provenance"]["native_manifest"]
    if native is None:
        raise PreflightBlocked("MISSING_REAL_SHARD", "Published fixed native manifest is required")
    path = _regular(root, native["path"])
    if _file_sha(path) != family.MANIFEST_SHA256:
        raise PreflightBlocked("SHARD_REJECTED", "Not the frozen published GPTrans native manifest")
    # Compare the entire native read set before calling the legacy validator.
    # It must not open any undeclared file, even under an authorized root.
    payload = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    native_files = [
        {"path": item["file"], "role": item["role"], "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in payload.get("geometry_shards", [])
    ]
    if _json(native_files) != _json(record["files"]):
        raise PreflightBlocked("SHARD_REJECTED", "Native read set differs from authorized shards")
    return family.validate_fixed_assets(root, path, verify_content=True)


def _equal_state(left, right, torch):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal_state(left[k], right[k], torch) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_equal_state(a, b, torch) for a, b in zip(left, right))
    return left == right


def _gptrans(spec, arm, record, root, work, device, max_batches, result, guard):
    import torch
    from molgap import pcqm_gptrans_v4 as family
    from molgap.gptrans_adapter import build_gptrans_model, gptrans_metadata
    from molgap.v4_runtime import make_adamw_compat, model_state_sha256, torch_load_compat

    for function in (family._load_datasets, family._training_loader, family._development_loader,
                     family._forward, build_gptrans_model, gptrans_metadata):
        guard.check(function.__code__.co_filename)
    result["import_origins"] = guard.audit()
    assets = _native_assets(record, root, family)
    result["observed"]["shard_verified"] = True
    if arm["initialization"]["kind"] != "random":
        raise PreflightBlocked("UNSUPPORTED_INITIALIZATION", "v1 has no external frozen-state artifact input")
    target_device = torch.device(device)
    if target_device.type == "cuda" and (
        not torch.cuda.is_available() or target_device.index >= torch.cuda.device_count()
    ):
        raise PreflightBlocked("ENVIRONMENT_BLOCKED", f"Requested device unavailable: {device}")
    torch.manual_seed(arm["initialization"]["seed"])
    gptrans_metadata(spec, arm["arm_id"])
    model = build_gptrans_model(spec, arm["arm_id"])
    if model_state_sha256(model) != arm["initialization"]["state_sha256"]:
        raise PreflightBlocked("FAILED", "Observed initial model state differs from spec")
    model.to(target_device)
    result["observed_device"] = str(next(model.parameters()).device)
    train, shards = family._load_datasets(assets.train_paths)
    development, development_shards = family._load_datasets(assets.development_paths)
    if len(train) != family.TRAIN_ROWS or len(development) != family.DEVELOPMENT_ROWS:
        raise ValueError("Frozen loader row count mismatch")
    mean, std = family._target_stats(shards)
    # Keep the frozen batch/sampler. Only I/O worker count changes to avoid
    # unguarded spawned imports; this is explicitly a single-process diagnostic.
    family.LOADER_WORKERS = 0
    training = iter(family._training_loader(train, epoch=0))
    dev_batch = next(iter(family._development_loader(development))).to(target_device)
    first_batch = next(training).to(target_device)
    result["observed"]["loader_batch_built"] = True
    optimizer = make_adamw_compat(model.parameters(), lr=family.LEARNING_RATE,
                                  weight_decay=family.WEIGHT_DECAY, fused=False)

    def prediction(batch):
        values = family._forward(model, batch)
        targets = batch.y.view(-1)
        if (values.shape != targets.shape or values.numel() == 0
                or not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(targets).all())):
            raise ValueError("Nonfinite/misaligned predictions or targets")
        return values, targets

    model.eval()
    with torch.no_grad():
        prediction(dev_batch)
    for index in range(max_batches):
        batch = first_batch if index == 0 else next(training).to(target_device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values, targets = prediction(batch)
        result["observed"]["forward_checked"] = True
        loss = torch.nn.functional.l1_loss(values, (targets - mean) / std)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite diagnostic loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("Missing/nonfinite gradients")
        result["observed"]["backward_checked"] = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), family.GRADIENT_CLIP)
        before = model_state_sha256(model)
        optimizer.step()
        family.assert_finite_state_dict(model.state_dict(), context="real-shard preflight")
        if before == model_state_sha256(model):
            raise ValueError("Optimizer step did not change model state")
        result["observed"]["optimizer_step_checked"] = True
        result["batches_checked"] += 1
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "spec_identity": spec.identity, "arm_id": arm["arm_id"]}
    checkpoint = work / "diagnostic.pt"
    torch.save(state, checkpoint)
    restored = torch_load_compat(checkpoint, map_location="cpu", weights_only=True)
    if not _equal_state(state, restored, torch):
        raise ValueError("Checkpoint tensor/optimizer roundtrip mismatch")
    model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    if not _equal_state(model.state_dict(), state["model"], torch):
        raise ValueError("Reloaded model mismatch")
    result["observed"]["checkpoint_roundtrip_checked"] = True
    result["import_origins"] = guard.audit()
    result.update(status=SUCCESS, reason="All bounded checks observed on frozen published real shards")


def _worker(request_path: Path):
    request = _load(request_path.read_bytes())
    result = request["result"]
    root = Path(request["source_root"])
    guard = _PackageOnly(root, request["inventory"])
    try:
        if any(name == "molgap" or name.startswith("molgap.") for name in sys.modules):
            raise PreflightBlocked("IMPORT_ORIGIN_REJECTED", "molgap was imported before isolation")
        # -I -S suppresses PYTHONPATH, sitecustomize, .pth and editable finders.
        # Only explicit dependency directories are added, never a checkout src.
        sys.path.insert(0, str(root / "src"))
        sys.path.extend(request["dependency_paths"])
        sys.meta_path.insert(0, guard)
        from molgap.experiment_spec import ExperimentSpec
        spec = ExperimentSpec.from_json(request["spec_json"])
        if spec.identity != result["spec_identity"]:
            raise ValueError("Worker spec identity mismatch")
        arm = next(item for item in spec.to_dict()["arms"] if item["arm_id"] == result["arm_id"])
        record = request["record"]
        if record["provenance"]["kind"] == "test-only-synthetic":
            raise PreflightBlocked("NON_REAL_SHARD", "Synthetic inputs are never real-shard evidence")
        key = (arm["family"]["name"], arm["family"]["version"])
        if key != ("gptrans_t", "1"):
            raise PreflightBlocked("UNSUPPORTED_FAMILY_PREFLIGHT", "K1 frozen PCQM loader is not migrated in v1")
        _verify_files(record, Path(request["shard_root"]))
        _gptrans(spec, arm, record, Path(request["shard_root"]), request_path.parent,
                 request["device"], request["max_batches"], result, guard)
    except PreflightBlocked as exc:
        result.update(status=exc.status, reason=str(exc))
    except (ImportError, OSError) as exc:
        result.update(status="ENVIRONMENT_BLOCKED", reason=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
    try:
        result["import_origins"] = guard.audit()
    except Exception as exc:
        result.update(status="IMPORT_ORIGIN_REJECTED", reason=str(exc), import_origins={})
    _atomic(request_path.parent / "worker_result.json", _finish(result))


def _check_worker(loaded, expected, max_batches, inventory):
    if type(loaded) is not dict or set(loaded) != set(expected):
        raise ValueError("Worker result schema mismatch")
    for key in ("format", "spec_identity", "arm_id", "arm_identity", "package",
                "shard_manifest_sha256", "requested_device", "limitations"):
        if _json(loaded[key]) != _json(expected[key]):
            raise ValueError(f"Worker result binding mismatch: {key}")
    if type(loaded["status"]) is not str or loaded["status"] not in STATUSES:
        raise ValueError("Unknown worker status")
    if type(loaded["reason"]) is not str or not loaded["reason"]:
        raise ValueError("Missing worker reason")
    if (type(loaded["observed"]) is not dict or set(loaded["observed"]) != set(CHECKS)
            or any(type(value) is not bool for value in loaded["observed"].values())
            or loaded["observed"]["package_verified"] is not True):
        raise ValueError("Invalid observed checks")
    count = loaded["batches_checked"]
    if type(count) is not int or not 0 <= count <= max_batches:
        raise ValueError("Invalid observed batch count")
    if loaded["observed_device"] not in (None, expected["requested_device"]):
        raise ValueError("Observed device mismatch")
    origins = loaded["import_origins"]
    if type(origins) is not dict:
        raise ValueError("Invalid import origins")
    for name, entry in origins.items():
        _fields(entry, "path sha256")
        if not (name == "molgap" or name.startswith("molgap.")):
            raise ValueError("Invalid import name")
        if inventory.get(entry["path"], {}).get("sha256") != entry["sha256"]:
            raise ValueError("Unbound import origin")
    if loaded["status"] == SUCCESS and (
        not all(loaded["observed"].values()) or count != max_batches
        or loaded["observed_device"] != expected["requested_device"]
        or not {"molgap.pcqm_gptrans_v4", "molgap.gptrans_adapter"} <= origins.keys()
    ):
        raise ValueError("Unobserved checks cannot produce success")
    if loaded["missing_evidence"] != _finish(dict(loaded))["missing_evidence"]:
        raise ValueError("Worker missing-evidence mismatch")
    return loaded


def _run_arm(spec, arm, record, shard_root, snapshot, manifest, result, device, max_batches, timeout):
    if record["provenance"]["kind"] == "test-only-synthetic":
        raise PreflightBlocked("NON_REAL_SHARD", "Synthetic inputs are never real-shard evidence")
    if (arm["family"]["name"], arm["family"]["version"]) != ("gptrans_t", "1"):
        raise PreflightBlocked("UNSUPPORTED_FAMILY_PREFLIGHT", "K1 frozen PCQM loader is not migrated in v1")
    with tempfile.TemporaryDirectory(prefix="molgap-preflight-arm-") as temporary:
        work = Path(temporary)
        source = work / "source"
        source.mkdir()
        inventory = _extract(snapshot, source, manifest)
        data = work / "shards"
        data.mkdir()
        entries = list(record["files"])
        if record["provenance"]["native_manifest"] is not None:
            entries.append(record["provenance"]["native_manifest"])
        # Per-arm immutable-by-convention copies close the usual external input
        # mutation gap and exclude every undeclared/protected file from the loader.
        for entry in entries:
            original = _regular(shard_root, entry["path"])
            target = data / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with original.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
        _verify_files(record, data)
        paths = sysconfig.get_paths()
        dependencies = sorted({str(Path(paths[key]).resolve()) for key in ("purelib", "platlib")})
        request = {
            "spec_json": spec.to_json(), "result": result, "record": record,
            "source_root": str(source), "shard_root": str(data), "inventory": inventory,
            "dependency_paths": dependencies, "device": device, "max_batches": max_batches,
        }
        request_path = work / "request.json"
        _atomic(request_path, request)
        environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
        if device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        command = [sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), "--worker", str(request_path)]
        try:
            with (work / "worker.log").open("wb") as log:
                completed = subprocess.run(command, cwd=work, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise PreflightBlocked("TIMED_OUT", "Isolated worker exceeded deadline; checks unobserved") from exc
        if completed.returncode != 0:
            raise PreflightBlocked("ENVIRONMENT_BLOCKED", f"Isolated worker exited {completed.returncode}")
        return _check_worker(_load((work / "worker_result.json").read_bytes()), result, max_batches, inventory)


def _aggregate_status(results):
    statuses = {item["status"] for item in results}
    if statuses == {SUCCESS}:
        return SUCCESS
    if len(statuses) == 1:
        return next(iter(statuses))
    return "PARTIAL_FAILURE" if SUCCESS in statuses else "BLOCKED_OR_FAILED"


def run_real_shard_preflight(
    spec: ExperimentSpec, package_dir: Path, shard_root: Path, output_root: Path,
    *, device: str = "cpu", max_batches: int = 1, manifest_path: Path | None = None,
    timeout_seconds: float = 300.0,
) -> dict:
    """Publish independent arm diagnostics to a new directory, summary last.

    This call executes bounded numerical checks when authorized real inputs are
    available. It must not be invoked merely to validate an implementation.
    """
    from .experiment_package import SIDECARS, verify_experiment_source_package
    spec = _validated(spec)
    if type(device) is not str or not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", device):
        raise ValueError("device must be cpu or cuda:<index>")
    if type(max_batches) is not int or not 1 <= max_batches <= 8:
        raise ValueError("max_batches must be an integer in [1, 8]")
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    arms = spec.to_dict()["arms"]
    for arm in arms:
        _name(arm["arm_id"])
    if len({arm["arm_id"].casefold() for arm in arms}) != len(arms):
        raise ValueError("Case-colliding arm IDs")
    output = Path(output_root).absolute()
    package_dir = Path(package_dir).absolute()
    shard_root = Path(shard_root).absolute()
    for path in (output, package_dir, shard_root):
        _no_links(path)
    if ".." in output.parts or output == Path(output.anchor):
        raise ValueError("Unsafe output root")
    _name(output.name)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Output must be new with an existing parent; never overwrite")
    for source in (package_dir, shard_root):
        if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
            raise ValueError("Output overlaps input")
    output.mkdir(exist_ok=False)
    (output / "arms").mkdir()
    binding = None
    shard_digest = None
    package_error = None
    shard_error = None
    manifest = None
    results = []
    with tempfile.TemporaryDirectory(prefix="molgap-preflight-package-") as temporary:
        snapshot = Path(temporary)
        try:
            original = verify_experiment_source_package(package_dir)
            for name in SIDECARS:
                shutil.copyfile(_regular(package_dir, name), snapshot / name)
            package_manifest = verify_experiment_source_package(snapshot)
            if original != package_manifest or package_manifest["spec_identity"] != spec.identity:
                raise ValueError("Package changed or spec identity mismatch")
            binding = {"package_identity": package_manifest["package_identity"],
                       "manifest_sha256": _file_sha(snapshot / "package_manifest.json"),
                       "archive_sha256": package_manifest["archive_sha256"]}
        except Exception as exc:
            package_error = ("PACKAGE_REJECTED", f"{type(exc).__name__}: {exc}")
        if package_error is None:
            try:
                path = Path(manifest_path).absolute() if manifest_path is not None else shard_root / MANIFEST_NAME
                path = _regular(shard_root, path.relative_to(shard_root).as_posix())
                raw = path.read_bytes()
                shard_digest = _sha(raw)
                manifest = _manifest(raw, spec)
            except FileNotFoundError as exc:
                shard_error = ("MISSING_REAL_SHARD", str(exc))
            except Exception as exc:
                shard_error = ("SHARD_REJECTED", f"{type(exc).__name__}: {exc}")
        for index, arm in enumerate(arms):
            directory = output / "arms" / arm["arm_id"]
            directory.mkdir()
            result = _base(spec, arm, device, binding, shard_digest)
            result["observed"]["package_verified"] = package_error is None
            if package_error or shard_error:
                result["status"], result["reason"] = package_error or shard_error
            else:
                try:
                    record = manifest["arms"][index]
                    _verify_files(record, shard_root)
                    result = _run_arm(spec, arm, record, shard_root, snapshot, package_manifest,
                                      result, device, max_batches, timeout_seconds)
                except PreflightBlocked as exc:
                    result.update(status=exc.status, reason=str(exc))
                except FileNotFoundError as exc:
                    result.update(status="MISSING_REAL_SHARD", reason=str(exc))
                except ValueError as exc:
                    result.update(status="SHARD_REJECTED", reason=str(exc))
                except OSError as exc:
                    result.update(status="ENVIRONMENT_BLOCKED", reason=str(exc))
                except Exception as exc:
                    result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
            _atomic(directory / "preflight.json", _finish(result))
            results.append(result)
    summary = {
        "format": REPORT_FORMAT, "spec_identity": spec.identity, "package": binding,
        "shard_manifest_sha256": shard_digest, "status": _aggregate_status(results),
        "requested_device": device, "max_batches": max_batches,
        "timeout_seconds": timeout_seconds, "limitations": list(LIMITATIONS),
        "arms": [{"arm_id": item["arm_id"], "arm_identity": item["arm_identity"],
                  "status": item["status"], "reason": item["reason"],
                  "observed": item["observed"], "missing_evidence": item["missing_evidence"],
                  "result_path": f"arms/{item['arm_id']}/preflight.json"} for item in results],
        "missing_evidence": sorted({value for item in results for value in item["missing_evidence"]}),
    }
    _atomic(output / "preflight_summary.json", summary)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker" or not sys.flags.isolated or not sys.flags.no_site:
        raise SystemExit("Private worker requires an isolated, no-site interpreter")
    _worker(Path(sys.argv[2]))
