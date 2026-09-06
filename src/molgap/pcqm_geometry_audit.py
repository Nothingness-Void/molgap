"""Shard-resumable numerical diagnosis without fitting validation targets."""
from __future__ import annotations

import copy
import gc
import json
import math
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .pcqm_gap_architecture import make_pcqm_gap_encoder
from .pcqm_geometry_warmstart import (
    CANDIDATE, _forward_geometry, _load_source_checkpoint, load_pretrained_backbone,
)
from .pcqm_official_edge_state import (
    PackedGraphDataset, atomic_json, atomic_torch, sha256_file,
)

ACCEPTANCE_SHA = "650b1fbd14771888ba1c9342cdcba240b420bd91b0dced974ae654a4b65ab9d2"
SOURCE_SHA = "ca5118128e5d1626901d054fcc152863a2fb40678c1ae914be9ea42db7e447fb"
REFERENCE_SHA = "98c27df7291841f2147672d76d2924adbc0b96822938fab31239e5971260ea56"


def checked_acceptance(path: Path) -> dict:
    if sha256_file(path) != ACCEPTANCE_SHA:
        raise RuntimeError("Unexpected immutable geometry acceptance")
    report = json.loads(path.read_text())
    if report["status"] != "accepted" or report["counts"] != {"train": 3378606, "valid": 73545}:
        raise RuntimeError("Geometry roles/counts changed")
    return report


def aligned_payload(payload: dict, reference: dict) -> dict:
    """Reject duplicate, missing, or relabeled rows before comparing predictions."""
    result = {}
    for name, values in (("payload", payload), ("reference", reference)):
        ids = values["source_idx"].view(-1)
        if ids.numel() != torch.unique(ids).numel():
            raise RuntimeError(f"Duplicate {name} source_idx")
        order = ids.argsort()
        result[name] = {key: value.view(-1)[order] for key, value in values.items()}
        if not all(torch.isfinite(value).all() for value in result[name].values()):
            raise RuntimeError(f"Non-finite {name} payload")
    if not torch.equal(result["payload"]["source_idx"], result["reference"]["source_idx"]):
        raise RuntimeError("Source identity coverage differs")
    torch.testing.assert_close(result["payload"]["target_eV"], result["reference"]["target_eV"], rtol=0, atol=1e-6)
    return result


def _source_forward(model, batch):
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.random_walk_pe).view(-1)


def _dropouts(model):
    return {name: float(module.p) for name, module in model.named_modules()
            if isinstance(module, torch.nn.Dropout)}


def numeric_step(model, batch, mean, std, amp: bool) -> dict:
    """Use a fresh disposable copy and identical RNG for each precision arm."""
    model = copy.deepcopy(model).train()
    torch.manual_seed(903)
    torch.cuda.manual_seed_all(903)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    with torch.amp.autocast("cuda", enabled=amp, dtype=torch.float16):
        pred = _forward_geometry(model, batch)
        loss = torch.nn.functional.l1_loss(pred, (batch.y.view(-1) - mean) / std)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = {name: p.grad.detach().float().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
    finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in grads.values())
    if finite:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        norm = torch.tensor(float("inf"))
    updated_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    change = sum(float((p.detach() - initial[name]).float().square().sum()) for name, p in model.named_parameters()) ** 0.5
    return {"loss": float(loss.detach()), "finite": finite and updated_finite,
            "gradient_norm": float(norm), "update_l2": change, "gradients": grads}


def run_audit(graph_dir: Path, acceptance_path: Path, source_path: Path,
              config_path: Path, reference_path: Path, output_dir: Path) -> dict:
    acceptance = checked_acceptance(acceptance_path)
    if sha256_file(source_path) != SOURCE_SHA or sha256_file(reference_path) != REFERENCE_SHA:
        raise RuntimeError("Source checkpoint/prediction hash changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Audit needs a scheduled CUDA worker")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    checkpoint, source = _load_source_checkpoint(source_path, acceptance, config_path)
    model = make_pcqm_gap_encoder(CANDIDATE)
    mapping = load_pretrained_backbone(model, checkpoint["model"])
    source, model = source.cuda().eval(), model.cuda().eval()
    del checkpoint
    mean, std = float(acceptance["target_mean_gap"]), float(acceptance["target_std_gap"])
    identity = {"format": "pcqm-geometry-audit-v1", "source_sha256": SOURCE_SHA,
                "acceptance_sha256": ACCEPTANCE_SHA, "reference_sha256": REFERENCE_SHA,
                "config_sha256": sha256_file(config_path), "code_sha256": sha256_file(Path(__file__))}
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = output_dir / "contract.json"
    if contract.exists() and json.loads(contract.read_text()) != identity:
        raise RuntimeError("Audit resume identity changed")
    atomic_json(contract, identity)
    config = torch.load(config_path, map_location="cpu", weights_only=False)["config"]
    atomic_json(output_dir / "configuration.json", {
        "source_config": config, "source_dropouts": _dropouts(source),
        "candidate_dropouts": _dropouts(model), "mapping": mapping,
    })
    pieces = []
    for record in sorted(acceptance["shards"], key=lambda r: r["path"]):
        if record["role"] != "valid":
            continue
        path = graph_dir / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Changed graph shard: {path}")
        part = output_dir / "parts" / path.name
        if part.exists():
            saved = torch.load(part, map_location="cpu", weights_only=False)
            if saved["identity"] != identity or saved["graph_sha256"] != record["sha256"]:
                raise RuntimeError("Audit part identity changed")
            payload = saved["payload"]
        else:
            fields = {key: [] for key in ("source_idx", "target_eV", "source_fp32", "source_fp16", "candidate_fp32", "candidate_fp16")}
            dataset = PackedGraphDataset(path)
            loader = DataLoader(dataset, batch_size=192, shuffle=False, num_workers=0)
            with torch.no_grad():
                for batch in loader:
                    batch = batch.cuda()
                    fields["source_idx"].append(batch.source_idx.view(-1).cpu())
                    fields["target_eV"].append(batch.y.view(-1).cpu())
                    for label, encoder, forward in (("source", source, _source_forward), ("candidate", model, _forward_geometry)):
                        for amp in (False, True):
                            with torch.amp.autocast("cuda", enabled=amp, dtype=torch.float16):
                                value = forward(encoder, batch)
                            fields[f"{label}_{'fp16' if amp else 'fp32'}"].append((value.float() * std + mean).cpu())
            payload = {key: torch.cat(values) for key, values in fields.items()}
            atomic_torch(part, {"identity": identity, "graph_sha256": record["sha256"], "payload": payload})
            del loader, dataset
            gc.collect()
        if len(payload["source_idx"]) != record["rows"]:
            raise RuntimeError("Audit part row count changed")
        pieces.append(payload)
        print(f"audit valid shard={path.name} rows={record['rows']} saved", flush=True)
        atomic_json(output_dir / "progress.json", {"status": "evaluating", "completed_shards": len(pieces)})
    full = {key: torch.cat([p[key] for p in pieces]) for key in pieces[0]}
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    aligned = aligned_payload(full, reference)
    full, reference = aligned["payload"], aligned["reference"]
    if len(full["source_idx"]) != 73545:
        raise RuntimeError("Incomplete validation coverage")
    maes = {key: float((value.double() - full["target_eV"].double()).abs().mean())
            for key, value in full.items() if key not in ("source_idx", "target_eV")}
    equality = float((full["source_fp32"] - full["candidate_fp32"]).abs().max())
    precision_drift = float((full["candidate_fp16"] - full["candidate_fp32"]).abs().mean())
    reference_drift = float((full["source_fp16"] - reference["prediction_eV"]).abs().mean())
    train_record = next(r for r in acceptance["shards"] if r["role"] == "train")
    train_path = graph_dir / train_record["path"]
    if sha256_file(train_path) != train_record["sha256"]:
        raise RuntimeError("Changed numerical-probe train shard")
    dataset = PackedGraphDataset(train_path)
    probe_results = []
    for index, batch in enumerate(DataLoader(dataset, batch_size=192, shuffle=False, num_workers=0)):
        if index == 4:
            break
        batch = batch.cuda()
        a, b = numeric_step(model, batch, mean, std, False), numeric_step(model, batch, mean, std, True)
        ga, gb = a.pop("gradients"), b.pop("gradients")
        relative = math.sqrt(sum(float((ga[k] - gb[k]).double().square().sum()) for k in ga) /
                             max(1e-30, sum(float(g.double().square().sum()) for g in ga.values())))
        probe_results.append({"batch": index, "fp32": a, "fp16": b, "relative_gradient_l2": relative})
        atomic_json(output_dir / "numerical_steps.json", probe_results)
        print(f"audit numeric batch={index} fp32_finite={a['finite']} fp16_finite={b['finite']}", flush=True)
    passed = equality <= 2e-5 and reference_drift <= 5e-4 and all(p["fp32"]["finite"] for p in probe_results)
    result = {"status": "accepted" if passed else "rejected", "identity": identity,
              "rows": len(full["source_idx"]), "mae_eV": maes,
              "initial_fp32_max_difference_eV": equality,
              "source_fp16_mean_difference_vs_retained_eV": reference_drift,
              "candidate_fp16_mean_difference_vs_fp32_eV": precision_drift,
              "fp16_finite": all(p["fp16"]["finite"] for p in probe_results),
              "source_config_dropout": config["dropout"], "candidate_factory_dropout": 0.1,
              "numerical_steps": probe_results, "official_test_used": False,
              "parts": [{"path": str(p.relative_to(output_dir)), "sha256": sha256_file(p)} for p in sorted((output_dir / "parts").glob("*.pt"))]}
    atomic_json(output_dir / "result.json", result)
    atomic_json(output_dir / "progress.json", {"status": "complete", "accepted": passed})
    atomic_json(output_dir / "completion_manifest.json", {"status": "complete", "result_sha256": sha256_file(output_dir / "result.json")})
    if not passed:
        raise RuntimeError("Numerical audit gate failed; see result.json")
    return result
