"""Source-bound CPU loader/model diagnostics; not scientific admission.

This file also serves as the isolated stdlib bootstrap. Family imports happen
only in a fresh -I -S interpreter after installing the unpacked-source guard.
"""
from __future__ import annotations

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
SMOKE_MODE = "model_smoke_v1"
SMOKE_FORMAT = "molgap-experiment-model-smoke-v1"
# Reviewed ae7674d source semantics, not caller-provided claims of real code.
SMOKE_SOURCE = {
    "pcqm_gptrans_v4": "7ec7c44dd766296d7aa683c35e9adab4f6d245d91e2506d674d17d3c5ad73d9c",
    "gptrans_adapter": "8a4c77ac91750b2c3f09fcdb876e9f20c82b87c06614253a6ab98ae4ad28e21a",
    "gptrans": "04edcb6f928617d1142c624d071b7d7accb15c92d2f66e3473ed666ca7e5b2b2",
    "gptrans_variants": "a84355f2ab4eaa9cc4c3269d4ba0646c81d03b09a600b1f5e64864951de6a35a",
    "v4_runtime": "0a973278b8c3b41ff9f36ed4b7a3f4fc7a068c76dc1c24f57d4bd01843bb7b51",
    "experiment_spec": "7fd97ba800c7241dec6737363ffa3d2eedbe8c014636ddd7a9c14c99abcb4c49",
    "training_reproducibility": "6608ee1a52e432c2f2a7f68e483f5cb5ec162f95d1d09d9d7b35d3ea7e3f48ba",
}
SMOKE_LIMITATIONS = [
    "CPU FP32 single-train-batch diagnostic; no runtime qualification or scientific admission.",
    "Seeded fresh initialization, not verified frozen initial-state artifact parity.",
    "Checkpoint unsupported: frozen helper emits certificate-bound scientific fields; no certificate supplied.",
]
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


def _smoke_source(root):
    for name, expected in SMOKE_SOURCE.items():
        raw = _regular(root / f"src/molgap/{name}.py").read_bytes()
        if _sha(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")) != expected:
            raise ValueError(f"Unsupported model-smoke source contract: {name}")


def _checkpoint_support(runtime):
    # _save_checkpoint requires a runtime certificate and labels scientific
    # fields. A CPU diagnostic must not fabricate those fields or a resume claim.
    return {"status": "CHECKPOINT_UNSUPPORTED", "checked": False,
            "helper_available": callable(getattr(runtime, "_save_checkpoint", None)),
            "reason": "No certified checkpoint contract for CPU model_smoke_v1"}


def _model_smoke(runtime, request, batch, stats, result):
    import torch
    from molgap.experiment_spec import ExperimentSpec
    from molgap.gptrans_adapter import build_gptrans_model, gptrans_metadata

    spec = ExperimentSpec.from_json(request["spec_json"])
    if spec.identity != request["spec_identity"]:
        raise ValueError("Worker spec identity mismatch")
    metadata = gptrans_metadata(spec, request["entry"]["arm_id"])
    torch.manual_seed(runtime.SEED)
    model = build_gptrans_model(spec, metadata.arm_id).to(device="cpu", dtype=torch.float32)
    result["import_origins"] = _origins(Path(request["source_root"]))
    model.train()
    optimizer = runtime.make_adamw_compat(
        model.parameters(), lr=runtime.LEARNING_RATE, weight_decay=runtime.WEIGHT_DECAY,
        fused=False, foreach=False)
    scheduler = runtime.FrozenEpochScheduler(optimizer)
    scheduler.step(0)
    ema = runtime.ExponentialMovingAverage(model)
    mean, std = stats
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid real training target statistics")
    optimizer.zero_grad(set_to_none=True)
    prediction = runtime._forward(model, batch)
    target = (batch.y.view(-1).float() - mean) / std
    if (tuple(prediction.shape) != (int(batch.num_graphs),)
            or prediction.shape != target.shape
            or not bool(torch.isfinite(prediction).all())
            or not bool(torch.isfinite(target).all())):
        raise ValueError("Invalid model prediction/target shape or finiteness")
    loss = torch.nn.functional.l1_loss(prediction, target)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite normalized-gap-L1 loss")
    result["observations"].update(
        variant=metadata.variant, loss=float(loss.detach()), loss_name="normalized-gap-l1",
        prediction_shape=list(prediction.shape), target_shape=list(target.shape),
        target_mean=mean, target_std=std)
    result["forward_checked"] = True
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("Missing or nonfinite gradients")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), runtime.GRADIENT_CLIP)
    if not bool(torch.isfinite(norm)):
        raise ValueError("Nonfinite gradient norm")
    result["observations"].update(gradient_norm=float(norm), gradient_tensors=len(gradients))
    result["backward_checked"] = True
    before = runtime.model_state_sha256(model)
    optimizer.step()
    ema.update(model)
    runtime.assert_finite_state_dict(model.state_dict(), label="smoke model")
    runtime.assert_finite_state_dict(ema.state_dict(), label="smoke EMA")
    states = list(optimizer.state.values())
    if not states:
        raise ValueError("Optimizer has no observed state")
    for state in states:
        if any(not bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v)):
            raise ValueError("Nonfinite optimizer state")
        if float(state["step"]) != 1:
            raise ValueError("Unexpected optimizer step cursor")
    after = runtime.model_state_sha256(model)
    if before == after:
        raise ValueError("Optimizer did not change model state")
    result["observations"].update(model_before_sha256=before, model_after_sha256=after,
                                  optimizer_steps=1, optimizer_state_entries=len(states))
    result["optimizer_step_checked"] = True
    result["checkpoint"] = _checkpoint_support(runtime)
    result["status"] = "MODEL_SMOKE_CHECKPOINT_UNSUPPORTED"


def _worker(request):
    root = Path(request["source_root"])
    if any(name == "molgap" or name.startswith("molgap.") for name in sys.modules):
        raise ImportError("Bootstrap already imported host molgap")
    mode = request.get("mode", "loader_only")
    if type(mode) is not str or mode not in {"loader_only", SMOKE_MODE}:
        raise ValueError("Unknown preflight mode")
    sys.path[:] = [str(root / "src"), *request["dependency_paths"], *sys.path]
    sys.meta_path.insert(0, _PackageOnly(root))
    result = {"status": "LOADER_FAILED", "shard_verified": False,
              "loader_batch_built": False, "import_origins": {}, "batches": {},
              "device": None, "error": None}
    smoke = mode == SMOKE_MODE
    if smoke:
        result.update({key: False for key in CHECKS[3:]})
        result.update(observations={}, checkpoint={"status": "NOT_RUN", "checked": False})
    try:
        if smoke:
            _smoke_source(root)
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
            if smoke and role == "train":
                train_batch, target_stats = batch, runtime._target_stats(shards)
            del batch, loader, graphs, shards
        result["import_origins"] = _origins(root)
        result.update(status="LOADER_VERIFIED_ONLY", loader_batch_built=True)
        if smoke:
            result["status"] = "MODEL_SMOKE_FAILED"
            _model_smoke(runtime, request, train_batch, target_stats, result)
            result["import_origins"] = _origins(root)
    except Exception as exc:
        if smoke:
            result["status"] = "MODEL_SMOKE_FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return result


def _launch(source_root, data_root, entry, timeout_seconds, workspace, *, mode="loader_only", spec=None):
    if type(mode) is not str or mode not in {"loader_only", SMOKE_MODE}:
        raise ValueError("Unknown preflight mode")
    # Copy only this infrastructure bootstrap, never host family source.
    bootstrap = workspace / "bootstrap.py"
    shutil.copyfile(Path(__file__), bootstrap)
    request = {"source_root": str(source_root), "data_root": str(data_root), "entry": entry,
               "dependency_paths": sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})}
    if mode == SMOKE_MODE:
        request.update(mode=mode, spec_json=spec.to_json(), spec_identity=spec.identity)
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
    smoke = mode == SMOKE_MODE
    extra = " forward_checked backward_checked optimizer_step_checked checkpoint_roundtrip_checked observations checkpoint" if smoke else ""
    _fields(result, "status shard_verified loader_batch_built import_origins batches device error" + extra)
    allowed = {"LOADER_VERIFIED_ONLY", "LOADER_FAILED", "UNSUPPORTED_FAMILY_PREFLIGHT"}
    if smoke:
        allowed = {"MODEL_SMOKE_FAILED", "MODEL_SMOKE_CHECKPOINT_UNSUPPORTED", "UNSUPPORTED_FAMILY_PREFLIGHT"}
    if result["status"] not in allowed:
        raise ValueError("Invalid worker status")
    if any(type(result[key]) is not bool for key in ("shard_verified", "loader_batch_built")):
        raise ValueError("Invalid worker observations")
    if result["status"] == "LOADER_VERIFIED_ONLY" and not result["loader_batch_built"]:
        raise ValueError("Incomplete loader observations")
    if smoke:
        flags = [result[key] for key in CHECKS[1:]]
        if any(type(flag) is not bool for flag in flags) or any(
                flags[i] and not all(flags[:i]) for i in range(len(flags))):
            raise ValueError("Invalid smoke observation ordering")
        if result["checkpoint_roundtrip_checked"]:
            raise ValueError("Unsupported checkpoint claim")
        if result["status"] == "MODEL_SMOKE_CHECKPOINT_UNSUPPORTED" and (
                not all(flags[:-1]) or result["error"] is not None
                or result["checkpoint"]["status"] != "CHECKPOINT_UNSUPPORTED"
                or result["checkpoint"]["checked"] is not False):
            raise ValueError("Incomplete smoke observations")
    if result["loader_batch_built"]:
        if (not result["shard_verified"] or (not smoke and result["error"] is not None)
                or set(result["batches"]) != {"train", "development"}
                or "molgap.pcqm_gptrans_v4" not in result["import_origins"]
                or result["device"] != "cpu"):
            raise ValueError("Incomplete worker observations")
        if not smoke and result["status"] != "LOADER_VERIFIED_ONLY":
            raise ValueError("Failed worker claimed completed loader observation")
    for name in result["import_origins"].values():
        _regular(_under(source_root, name))
    return result


def _blank(spec, arm, package_identity, manifest_digest):
    from .screen_policy import canonical_fingerprint
    return {"format": REPORT_FORMAT, "spec_identity": spec.identity,
            "package_identity": package_identity, "shard_manifest_sha256": manifest_digest,
            "arm_id": arm["arm_id"], "arm_identity": canonical_fingerprint(arm),
            "status": "NOT_RUN", **{key: False for key in CHECKS},
            "requested_device": "cpu", "device": None,
            "import_origins": {}, "batches": {}, "error": None,
            "limitations": list(LIMITATIONS), "missing_evidence": list(CHECKS)}


def run_experiment_preflight(spec, package_dir, output_root, *, expected_package_identity,
                             shard_manifest=None, shard_root=None,
                             expected_shard_manifest_sha256=None, timeout_seconds=300.0,
                             mode="loader_only"):
    """Publish independent arm reports and a summary in a *new* output directory.

    Default loader_only never constructs a model. Explicit model_smoke_v1
    performs one CPU train-batch step, without checkpoint certification.
    Invalid output/spec/API arguments raise; artifact failures are structured.
    """
    from .experiment_package import SIDECARS, verify_experiment_source_package
    from .experiment_spec import ExperimentSpec

    if type(mode) is not str or mode not in {"loader_only", SMOKE_MODE}:
        raise ValueError("Unknown preflight mode")
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec")
    rebuilt = ExperimentSpec.from_json(spec.to_json())
    if rebuilt.to_json() != spec.to_json() or rebuilt.identity != spec.identity:
        raise ValueError("Invalid spec identity")
    spec = rebuilt
    _digest(expected_package_identity)
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
    reports = [_blank(spec, arm, None, None) for arm in arms]
    limitations = list(LIMITATIONS)
    if mode == SMOKE_MODE:
        limitations = SMOKE_LIMITATIONS + LIMITATIONS[1:2] + LIMITATIONS[3:]
        for report in reports:
            report.update(format=SMOKE_FORMAT, mode=mode, limitations=list(limitations),
                          observations={}, checkpoint={"status": "NOT_RUN", "checked": False})
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
            else:
                stage = workspace / f"arm-{index}"
                stage.mkdir()
                source_root, data_root = stage / "source", stage / "data"
                source_root.mkdir()
                data_root.mkdir()
                try:
                    entry = entries[arm["arm_id"]]
                    _stage_files(entry, Path(shard_root).absolute(), data_root)
                    _unpack(snapshot, source_root)
                    if not (source_root / "src/molgap/pcqm_gptrans_v4.py").is_file():
                        report["status"] = "UNSUPPORTED_FAMILY_PREFLIGHT"
                    else:
                        if mode == SMOKE_MODE:
                            report.update(_launch(source_root, data_root, entry, timeout_seconds,
                                                  stage, mode=mode, spec=spec))
                        else:
                            report.update(_launch(source_root, data_root, entry, timeout_seconds, stage))
                except FileNotFoundError as exc:
                    report.update(status="MISSING_REAL_SHARD", error={"type": type(exc).__name__, "message": str(exc)})
                except Exception as exc:
                    report.update(status="PREFLIGHT_FAILED", error={"type": type(exc).__name__, "message": str(exc)})
            report["missing_evidence"] = [key for key in CHECKS if not report[key]]
            directory = output / arm["arm_id"]
            directory.mkdir()
            _atomic(directory / "preflight.json", report)
    statuses = {r["status"] for r in reports}
    status = next(iter(statuses)) if len(statuses) == 1 else "MIXED_NONPASS"
    summary = {"format": SMOKE_FORMAT if mode == SMOKE_MODE else REPORT_FORMAT, "spec_identity": spec.identity,
               "expected_package_identity": expected_package_identity,
               "expected_shard_manifest_sha256": expected_shard_manifest_sha256,
               "package_identity": reports[0]["package_identity"],
               "shard_manifest_sha256": reports[0]["shard_manifest_sha256"],
               "status": status, "requested_device": "cpu", "arms": reports,
               "limitations": limitations,
               "missing_evidence": sorted({item for r in reports for item in r["missing_evidence"]})}
    _atomic(output / "preflight.json", summary)
    return summary


if __name__ == "__main__":
    # Internal bootstrap only, not a public CLI or a platform entrypoint.
    _atomic(Path(sys.argv[2]), _worker(_load(Path(sys.argv[1]).read_bytes())))
