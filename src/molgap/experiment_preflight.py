"""Source-bound CPU diagnostics, loader-only by default; not scientific admission.

This file also serves as the isolated stdlib bootstrap. Family imports happen
only in a fresh -I -S interpreter after installing the unpacked-source guard.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import tempfile

__all__ = ["run_experiment_preflight", "validate_real_shard_manifest"]

MANIFEST_FORMAT = "molgap-real-shard-preflight-manifest-v1"
REPORT_FORMAT = "molgap-experiment-preflight-v1"
LOADER_MODE = "loader-only-v1"
MODEL_MODE = "model-smoke-v1"
MODEL_CHECKS = ("forward_checked", "backward_checked", "optimizer_step_checked",
                "checkpoint_roundtrip_checked")
CHECKS = (
    "package_verified", "shard_verified", "loader_batch_built", "forward_checked",
    "backward_checked", "optimizer_step_checked", "checkpoint_roundtrip_checked",
)
LIMITATIONS = [
    "CPU loader-only diagnostic; no model construction or numerical qualification.",
    "Frozen loader worker count is set to zero for isolated CPU inspection.",
    "No training, platform submission, scientific acceptance or downstream authority.",
    "Source and packed pickle inputs must be trusted; isolation is not a security sandbox.",
    "Manifest identities bind declarations, not independent scientific provenance.",
]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _load(raw):
    result = json.loads(raw, object_pairs_hook=_unique)
    if type(result) is not dict or _canonical(result) != raw:
        raise ValueError("Expected canonical JSON object (no newline or duplicate keys)")
    return result


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError("Unexpected manifest fields")


def _digest(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected lowercase SHA256")


def _relative(name):
    if (type(name) is not str or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute()):
        raise ValueError("Unsafe relative path")
    for part in name.split("/"):
        if (part in {"", ".", ".."} or part.endswith((".", " "))
                or any(ord(c) < 32 for c in part)
                or any(c in part for c in '<>"|?*')
                or part.split(".")[0].upper() in {
                    "CON", "PRN", "AUX", "NUL",
                    *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10)),
                }):
            raise ValueError("Unsafe relative path component")
    return name


def _no_links(path):
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Symlink/junction/reparse paths are forbidden")


def _regular(path):
    _no_links(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Expected regular file")
    return path


def _under(root, name):
    path = root / _relative(name)
    _no_links(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escaped root")
    return path


def _atomic(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".preflight-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_real_shard_manifest(spec, manifest_path, expected_sha256):
    """Validate authorization declarations, not files or loader success.

    A trusted caller must pin the digest separately. Each entry repeats exactly
    its arm's data identity and lists only authorized train/development files.
    Arms may be omitted; omitted arms remain MISSING_REAL_SHARD.
    """
    from .experiment_spec import ExperimentSpec
    from .screen_policy import canonical_fingerprint

    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec")
    _digest(expected_sha256)
    raw = _regular(Path(manifest_path).absolute()).read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("Shard manifest digest mismatch")
    manifest = _load(raw)
    _fields(manifest, "format spec_identity arms")
    if manifest["format"] != MANIFEST_FORMAT or manifest["spec_identity"] != spec.identity:
        raise ValueError("Shard manifest format/spec identity mismatch")
    if type(manifest["arms"]) is not list:
        raise ValueError("Expected manifest arms array")
    arms = {arm["arm_id"]: arm for arm in spec.to_dict()["arms"]}
    seen = set()
    for entry in manifest["arms"]:
        _fields(entry, "arm_id arm_identity data kind fixed_manifest files")
        arm_id = entry["arm_id"]
        if type(arm_id) is not str or arm_id not in arms or arm_id in seen:
            raise ValueError("Unknown or duplicate shard arm")
        seen.add(arm_id)
        arm = arms[arm_id]
        if entry["arm_identity"] != canonical_fingerprint(arm) or entry["data"] != arm["data"]:
            raise ValueError("Shard arm/data identity mismatch")
        if entry["kind"] != "real-packed-pcqm":
            raise ValueError("Only explicitly authorized real packed shards are supported")
        _fields(entry["fixed_manifest"], "path sha256")
        _relative(entry["fixed_manifest"]["path"])
        _digest(entry["fixed_manifest"]["sha256"])
        if type(entry["files"]) is not list or not entry["files"]:
            raise ValueError("Missing real shard files")
        roles = {role["role"] for role in arm["data"]["roles"]}
        paths = {entry["fixed_manifest"]["path"].casefold()}
        observed_roles = set()
        for item in entry["files"]:
            _fields(item, "path sha256 bytes role")
            name = _relative(item["path"])
            _digest(item["sha256"])
            if type(item["bytes"]) is not int or item["bytes"] <= 0:
                raise ValueError("Invalid shard byte count")
            if type(item["role"]) is not str or item["role"] not in roles & {"train", "development"}:
                raise ValueError("Protected or undeclared role")
            if name.casefold() in paths:
                raise ValueError("Duplicate/case-colliding shard path")
            paths.add(name.casefold())
            observed_roles.add(item["role"])
        if observed_roles != roles:
            raise ValueError("Incomplete authorized role coverage")
    return manifest


def _unpack(package, root):
    """Defense in depth after the existing package verifier; never extractall."""
    seen = set()
    with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
        for member in archive:
            name = _relative(member.name)
            if (member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
                    or member.sparse is not None or member.linkname or name.casefold() in seen):
                raise ValueError("Unsafe or duplicate archive member")
            seen.add(name.casefold())
            target = _under(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing archive member stream")
            with stream, target.open("xb") as handle:
                shutil.copyfileobj(stream, handle)


def _stage_files(entry, root, destination):
    _no_links(root)
    if not root.is_dir():
        raise FileNotFoundError("Real shard root missing")
    for item in [entry["fixed_manifest"], *entry["files"]]:
        source = _regular(_under(root, item["path"]))
        target = _under(destination, item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if _file_sha(target) != item["sha256"]:
            raise ValueError("Real shard file hash mismatch")
        if "bytes" in item and target.stat().st_size != item["bytes"]:
            raise ValueError("Real shard file size mismatch")


class _PackageOnly(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root.resolve()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "molgap" and not fullname.startswith("molgap."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(self.root):
            raise ImportError(f"Host/missing package import rejected: {fullname}")
        if spec.submodule_search_locations is not None and any(
            not Path(p).resolve().is_relative_to(self.root) for p in spec.submodule_search_locations
        ):
            raise ImportError(f"Host package search path rejected: {fullname}")
        return spec


def _origins(root):
    observed = {}
    for name, module in list(sys.modules.items()):
        if name == "molgap" or name.startswith("molgap."):
            path = getattr(module, "__file__", None)
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            if (not path or not origin or Path(path).resolve() != Path(origin).resolve()
                    or not Path(path).resolve().is_relative_to(root.resolve())):
                raise ImportError(f"Host import pollution: {name}")
            observed[name] = Path(path).resolve().relative_to(root.resolve()).as_posix()
    return observed


def _exact(left, right):
    """Compare checkpoint state exactly, including tensor/array byte identity."""
    import torch
    import numpy as np
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (left.dtype == right.dtype and left.shape == right.shape and left.device == right.device
                and left.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                == right.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


def _diagnostic_roundtrip(runtime, model, optimizer, scheduler, generator, path):
    import torch
    expected = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(),
                              "rng": runtime.capture_rng_state(loader_generator=generator)})
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(expected, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        # This private payload is diagnostic state, never a production checkpoint.
        with path.open("rb") as handle:
            observed = torch.load(handle, map_location="cpu", weights_only=False)
        if not _exact(expected, observed):
            raise ValueError("Diagnostic checkpoint serialized state mismatch")
        restored_model = runtime._make_model(initial_state_path=None, variant="reference").to("cpu")
        restored_optimizer = runtime.make_adamw_compat(
            restored_model.parameters(), lr=runtime.LEARNING_RATE,
            weight_decay=runtime.WEIGHT_DECAY, fused=False, foreach=False)
        restored_scheduler = runtime.FrozenEpochScheduler(restored_optimizer)
        restored_model.load_state_dict(observed["model"], strict=True)
        restored_optimizer.load_state_dict(observed["optimizer"])
        restored_scheduler.load_state_dict(observed["scheduler"])
        runtime.restore_rng_state(observed["rng"], loader_generator=generator)
        actual = {"model": restored_model.state_dict(), "optimizer": restored_optimizer.state_dict(),
                  "scheduler": restored_scheduler.state_dict(),
                  "rng": runtime.capture_rng_state(loader_generator=generator)}
        if not _exact(expected, actual):
            raise ValueError("Diagnostic checkpoint restored state mismatch")
        return sorted(observed["rng"])
    finally:
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def _model_smoke(runtime, arm, batch, stats, generator, path, result):
    import random
    import numpy as np
    import torch
    result["status"] = "MODEL_SMOKE_FAILED"
    random.seed(arm["initialization"]["seed"])
    np.random.seed(arm["initialization"]["seed"])
    torch.manual_seed(arm["initialization"]["seed"])
    model = runtime._make_model(initial_state_path=None, variant="reference").to("cpu")
    identity = runtime._state_sha256(model)
    if identity != arm["initialization"]["state_sha256"]:
        raise ValueError("Random initial state differs from declared identity")
    runtime._verify_model_identity(model)
    result.update(initial_state_checked=True, initial_state_sha256=identity)
    optimizer = runtime.make_adamw_compat(model.parameters(), lr=runtime.LEARNING_RATE,
                                          weight_decay=runtime.WEIGHT_DECAY,
                                          fused=False, foreach=False)
    scheduler = runtime.FrozenEpochScheduler(optimizer)
    scheduler.step(0)
    mean, std = stats
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid real train normalization")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = runtime._forward(model, batch)
    target = (batch.y.view(-1).float() - mean) / std
    if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
        raise ValueError("Invalid finite forward output")
    loss = torch.nn.functional.l1_loss(prediction, target)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite normalized-gap-L1")
    result.update(forward_checked=True, loss=float(loss.detach()))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("Missing or nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.GRADIENT_CLIP)
    if not bool(torch.isfinite(norm)):
        raise ValueError("Nonfinite gradient norm")
    result["backward_checked"] = True
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite updated parameters")
    if runtime._state_sha256(model) == identity:
        raise ValueError("Optimizer did not change model state")
    result["optimizer_step_checked"] = True
    result["rng_components"] = _diagnostic_roundtrip(
        runtime, model, optimizer, scheduler, generator, path)
    result["checkpoint_roundtrip_checked"] = True
    result["status"] = "MODEL_SMOKE_VERIFIED_ONLY"


def _worker(request):
    root = Path(request["source_root"])
    if any(name == "molgap" or name.startswith("molgap.") for name in sys.modules):
        raise ImportError("Bootstrap already imported host molgap")
    sys.path[:] = [str(root / "src"), *request["dependency_paths"], *sys.path]
    sys.meta_path.insert(0, _PackageOnly(root))
    result = {"status": "LOADER_FAILED", "shard_verified": False,
              "loader_batch_built": False, "import_origins": {}, "batches": {},
              "device": None, "error": None, **{key: False for key in MODEL_CHECKS},
              "initial_state_checked": False, "initial_state_sha256": None,
              "loss": None, "rng_components": None}
    try:
        mode = request.get("mode", LOADER_MODE)
        if mode not in {LOADER_MODE, MODEL_MODE}:
            raise ValueError("Unknown versioned preflight mode")
        if mode == MODEL_MODE:
            arm = request["arm"]
            if (arm["family"] != {"name": "gptrans_t", "version": "1"}
                    or arm["addons"] or arm["initialization"]["kind"] != "random"):
                raise ValueError("Unsupported fixed model dispatch")
        runtime = importlib.import_module("molgap.pcqm_gptrans_v4")
        result["import_origins"] = _origins(root)
        if (not all(callable(getattr(runtime, name, None)) for name in (
                "validate_fixed_assets", "_load_datasets", "_training_loader", "_development_loader"))
                or not isinstance(getattr(runtime, "MANIFEST_SHA256", None), str)):
            result["status"] = "UNSUPPORTED_FAMILY_PREFLIGHT"
            result["error"] = {"type": "UnsupportedLoader", "message": "Frozen GPTrans loader contract absent"}
            return result
        entry = request["entry"]
        data_root = Path(request["data_root"])
        fixed = _under(data_root, entry["fixed_manifest"]["path"])
        # Pinned frozen reference bytes prevent a synthetic pickle becoming a
        # real-shard observation merely by labeling it "real" in a manifest.
        if _file_sha(fixed) != runtime.MANIFEST_SHA256:
            raise ValueError("Not the frozen real PCQM manifest")
        records = json.loads(fixed.read_bytes())["geometry_shards"]
        declared = [{"file": f["path"], "sha256": f["sha256"], "bytes": f["bytes"], "role": f["role"]}
                    for f in entry["files"]]
        actual = [{key: row[key] for key in ("file", "sha256", "bytes", "role")} for row in records]
        if declared != actual:
            raise ValueError("Authorized shard list differs from frozen manifest")
        assets = runtime.validate_fixed_assets(data_root, fixed, verify_content=True)
        result["shard_verified"] = True
        runtime.LOADER_WORKERS = 0
        train_batch = stats = generator = None
        for role, paths in (("train", assets.train_paths), ("development", assets.development_paths)):
            graphs, shards = runtime._load_datasets(paths)
            expected_rows = runtime.TRAIN_ROWS if role == "train" else runtime.DEVELOPMENT_ROWS
            if len(graphs) != expected_rows:
                raise ValueError("Frozen loader row count mismatch")
            loader = (runtime._training_loader(graphs, epoch=0) if role == "train"
                      else runtime._development_loader(graphs))
            batch = next(iter(loader))
            import torch
            if (int(batch.num_graphs) != runtime.PHYSICAL_BATCH
                    or batch.x.ndim != 2 or batch.x.shape[1] != 9
                    or batch.edge_attr.ndim != 2 or batch.edge_attr.shape[1] != 3
                    or batch.y.numel() != batch.num_graphs
                    or not bool(torch.isfinite(batch.y).all()) or batch.x.device.type != "cpu"):
                raise ValueError("Invalid real loader batch")
            result["batches"][role] = {"graphs": int(batch.num_graphs), "rows": len(graphs),
                                       "device": str(batch.x.device)}
            result["device"] = str(batch.x.device)
            if role == "train" and request.get("mode", LOADER_MODE) == MODEL_MODE:
                train_batch = batch
                stats = runtime._target_stats(shards)
                generator = getattr(loader, "generator", None)
            del batch, loader, graphs, shards
        result["import_origins"] = _origins(root)
        result.update(status="LOADER_VERIFIED_ONLY", loader_batch_built=True)
        if request.get("mode", LOADER_MODE) == MODEL_MODE:
            _model_smoke(runtime, request["arm"], train_batch, stats, generator,
                         Path(request["diagnostic_path"]), result)
            result["import_origins"] = _origins(root)
    except Exception as exc:
        if result["status"] in {"LOADER_VERIFIED_ONLY", "MODEL_SMOKE_VERIFIED_ONLY"}:
            result["status"] = "MODEL_SMOKE_FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def _launch(source_root, data_root, entry, timeout_seconds, workspace, *, mode=LOADER_MODE, arm=None):
    # Copy only this infrastructure bootstrap, never host family source.
    bootstrap = workspace / "bootstrap.py"
    shutil.copyfile(Path(__file__), bootstrap)
    request = {"source_root": str(source_root), "data_root": str(data_root), "entry": entry,
               "mode": mode, "arm": arm, "diagnostic_path": str(workspace / "diagnostic.pt"),
               "dependency_paths": sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})}
    request_path = workspace / "request.json"
    response_path = workspace / "response.json"
    _atomic(request_path, request)
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith("PYTHON")}
    env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    with (workspace / "worker.log").open("wb") as log:
        child = subprocess.run([sys.executable, "-I", "-S", str(bootstrap),
                                str(request_path), str(response_path)],
                               cwd=workspace, env=env, stdout=log, stderr=log,
                               timeout=timeout_seconds, check=False)
    if child.returncode != 0:
        raise RuntimeError(f"Isolated loader exited {child.returncode}")
    result = _load(_regular(response_path).read_bytes())
    _fields(result, "status shard_verified loader_batch_built import_origins batches device error "
                    "forward_checked backward_checked optimizer_step_checked checkpoint_roundtrip_checked "
                    "initial_state_checked initial_state_sha256 loss rng_components")
    if result["status"] not in {"LOADER_VERIFIED_ONLY", "LOADER_FAILED", "UNSUPPORTED_FAMILY_PREFLIGHT",
                                "MODEL_SMOKE_FAILED", "MODEL_SMOKE_VERIFIED_ONLY"}:
        raise ValueError("Invalid worker status")
    if any(type(result[key]) is not bool for key in ("shard_verified", "loader_batch_built",
                                                    "initial_state_checked", *MODEL_CHECKS)):
        raise ValueError("Invalid worker observations")
    if result["status"] in {"LOADER_VERIFIED_ONLY", "MODEL_SMOKE_VERIFIED_ONLY"} or result["loader_batch_built"]:
        if (not result["shard_verified"] or not result["loader_batch_built"]
                or set(result["batches"]) != {"train", "development"}
                or "molgap.pcqm_gptrans_v4" not in result["import_origins"]
                or result["device"] != "cpu"):
            raise ValueError("Incomplete worker observations")
    if result["status"] in {"LOADER_VERIFIED_ONLY", "MODEL_SMOKE_VERIFIED_ONLY"} and result["error"] is not None:
        raise ValueError("Successful diagnostic contains an error")
    if mode == LOADER_MODE and (any(result[k] for k in MODEL_CHECKS) or result["initial_state_checked"]):
        raise ValueError("Default loader mode reported model execution")
    if result["status"] == "MODEL_SMOKE_VERIFIED_ONLY":
        if (mode != MODEL_MODE or not all(result[k] for k in MODEL_CHECKS)
                or not result["initial_state_checked"] or result["loss"] is None
                or not result["rng_components"]):
            raise ValueError("Incomplete model smoke observations")
    for name in result["import_origins"].values():
        _regular(_under(source_root, name))
    return result


def _blank(spec, arm, package_identity, manifest_digest, mode=LOADER_MODE):
    from .screen_policy import canonical_fingerprint
    return {"format": REPORT_FORMAT, "spec_identity": spec.identity,
            "package_identity": package_identity, "shard_manifest_sha256": manifest_digest,
            "arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm),
            "status": "NOT_RUN", **{key: False for key in CHECKS},
            "requested_device": "cpu", "device": None,
            "mode": mode, "initial_state_checked": False, "initial_state_sha256": None,
            "loss": None, "rng_components": None,
            "import_origins": {}, "batches": {}, "error": None,
            "limitations": list(LIMITATIONS), "missing_evidence": list(CHECKS)}


def run_experiment_preflight(spec, package_dir, output_root, *, expected_package_identity,
                             shard_manifest=None, shard_root=None,
                             expected_shard_manifest_sha256=None, timeout_seconds=300.0,
                             mode=LOADER_MODE):
    """Publish independent arm reports and a summary in a *new* output directory.

    The default constructs no model. Explicit model-smoke-v1 performs one CPU
    diagnostic step on verified real data; neither mode is scientific admission.
    Invalid output/spec/API arguments raise; artifact failures are structured.
    """
    from .experiment_package import SIDECARS, verify_experiment_source_package
    from .experiment_spec import ExperimentSpec

    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec")
    rebuilt = ExperimentSpec.from_json(spec.to_json())
    if rebuilt.to_json() != spec.to_json() or rebuilt.identity != spec.identity:
        raise ValueError("Invalid spec identity")
    spec = rebuilt
    _digest(expected_package_identity)
    if type(mode) is not str or mode not in {LOADER_MODE, MODEL_MODE}:
        raise ValueError("Unknown versioned preflight mode")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Invalid timeout")
    output = Path(output_root).absolute()
    _no_links(output)
    if output.exists() or not output.parent.is_dir() or ".." in output.parts:
        raise ValueError("Output must be new, with an existing parent and no traversal")
    for supplied in (package_dir, shard_root):
        if supplied is not None:
            path = Path(supplied).resolve()
            if output.resolve().is_relative_to(path) or path.is_relative_to(output.resolve()):
                raise ValueError("Output overlaps an input root")
    arms = spec.to_dict()["arms"]
    for arm in arms:
        _relative(arm["arm_id"])
        if arm["arm_id"].casefold() == "preflight.json":
            raise ValueError("Arm ID collides with summary path")
    if len({a["arm_id"].casefold() for a in arms}) != len(arms):
        raise ValueError("Case-colliding arm IDs")
    output.mkdir(exist_ok=False)
    reports = [_blank(spec, arm, None, None, mode) for arm in arms]
    if mode == MODEL_MODE:
        for report in reports:
            report["limitations"][0] = "CPU single-step model diagnostic; no numerical qualification."
    package_error = None
    manifest_error = None
    manifest = None
    with tempfile.TemporaryDirectory(prefix="molgap-preflight-") as temporary:
        workspace = Path(temporary)
        snapshot = workspace / "package"
        snapshot.mkdir()
        try:
            source = Path(package_dir).absolute()
            _no_links(source)
            if {p.name for p in source.iterdir()} != SIDECARS:
                raise ValueError("Package sidecar set mismatch")
            for name in SIDECARS:
                shutil.copyfile(_regular(source / name), snapshot / name)
            receipt = verify_experiment_source_package(snapshot)
            if receipt["spec_identity"] != spec.identity or receipt["package_identity"] != expected_package_identity:
                raise ValueError("Pinned package/spec identity mismatch")
            for report in reports:
                report.update(package_verified=True, package_identity=receipt["package_identity"])
        except Exception as exc:
            package_error = {"type": type(exc).__name__, "message": str(exc)}
        if shard_manifest is not None:
            try:
                manifest = validate_real_shard_manifest(spec, shard_manifest, expected_shard_manifest_sha256)
                for report in reports:
                    report["shard_manifest_sha256"] = expected_shard_manifest_sha256
            except Exception as exc:
                manifest_error = {"type": type(exc).__name__, "message": str(exc)}
        entries = {entry["arm_id"]: entry for entry in manifest["arms"]} if manifest else {}
        for index, (arm, report) in enumerate(zip(arms, reports)):
            if package_error:
                report.update(status="PACKAGE_INVALID", error=package_error)
            elif manifest_error:
                report.update(status="SHARD_MANIFEST_INVALID", error=manifest_error)
            elif arm["family"]["name"] != "gptrans_t":
                report["status"] = "UNSUPPORTED_FAMILY_PREFLIGHT"
                report["limitations"].append("No frozen K1 PCQM loader supported by this boundary.")
            elif arm["arm_id"] not in entries or shard_root is None:
                report["status"] = "MISSING_REAL_SHARD"
            elif mode == MODEL_MODE and (arm["addons"] or arm["initialization"]["kind"] != "random"):
                report["status"] = "UNSUPPORTED_MODEL_SMOKE"
                report["error"] = {"type": "UnsupportedModel", "message": "Only fixed random GPTrans reference dispatch is supported"}
            else:
                stage = workspace / f"arm-{index}"
                stage.mkdir()
                source_root, data_root = stage / "source", stage / "data"
                source_root.mkdir()
                data_root.mkdir()
                try:
                    _unpack(snapshot, source_root)
                    if not (source_root / "src/molgap/pcqm_gptrans_v4.py").is_file():
                        report["status"] = "UNSUPPORTED_FAMILY_PREFLIGHT"
                    else:
                        entry = entries[arm["arm_id"]]
                        _stage_files(entry, Path(shard_root).absolute(), data_root)
                        report.update(_launch(source_root, data_root, entry, timeout_seconds, stage,
                                              mode=mode, arm=arm))
                except FileNotFoundError as exc:
                    report.update(status="MISSING_REAL_SHARD", error={"type": type(exc).__name__, "message": str(exc)})
                except Exception as exc:
                    report.update(status="PREFLIGHT_FAILED", error={"type": type(exc).__name__, "message": str(exc)})
            report["missing_evidence"] = [key for key in CHECKS if not report[key]]
            directory = output / "arms" / arm["arm_id"]
            directory.mkdir(parents=True)
            _atomic(directory / "preflight.json", report)
    statuses = {r["status"] for r in reports}
    status = next(iter(statuses)) if len(statuses) == 1 else "MIXED_NONPASS"
    summary = {"format": REPORT_FORMAT, "spec_identity": spec.identity,
               "expected_package_identity": expected_package_identity,
               "expected_shard_manifest_sha256": expected_shard_manifest_sha256,
               "package_identity": reports[0]["package_identity"],
               "shard_manifest_sha256": reports[0]["shard_manifest_sha256"],
               "status": status, "requested_device": "cpu", "mode": mode, "arms": reports,
               "limitations": list(reports[0]["limitations"]),
               "missing_evidence": sorted({item for r in reports for item in r["missing_evidence"]})}
    _atomic(output / "preflight_summary.json", summary)
    return summary


if __name__ == "__main__":
    # Internal bootstrap only, not a public CLI or a platform entrypoint.
    _atomic(Path(sys.argv[2]), _worker(_load(Path(sys.argv[1]).read_bytes())))
