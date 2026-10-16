"""Bounded structural audit; no dataset loader, training, ranking or submission."""
from dataclasses import asdict
import gc
from pathlib import Path
import statistics
import time

from .common.wire import canonical_fingerprint
from .fast_sandbox import synthetic_batch, _cpu_scope
from .runtime_profiling import RuntimeProfile
from .source_dependencies import discover_source_dependencies
from .training_reproducibility import sha256_file


def audit_structures(repo_root):
    import torch
    from .ppgt_model import PPGT, PPGTConfig
    from .esa_model import ESA, ESAConfig
    from .ppgt_features import feature_cost as ppgt_cost
    from .esa_features import feature_cost as esa_cost
    root = Path(repo_root).resolve()
    candidates = [(f"ppgt_deep_{h}", PPGT, PPGTConfig(hidden=h, heads=h//32), h == 256, 10493634)
                  for h in (320, 288, 256)]
    candidates += [(f"ppgt_shallow_{n}", PPGT, PPGTConfig(hidden=512, heads=16, blocks=n), n == 4, 10493634)
                   for n in (8, 7, 6, 5, 4)]
    candidates += [("ppgt_paper_structure", PPGT, PPGTConfig(hidden=512, heads=16), False, None),
                   ("esa_mixed", ESA, ESAConfig(), True, 7317634),
                   ("esa_compact", ESA, ESAConfig(hidden=256, layer_types=("M","M","S","P"), mlp_hidden=512), True, 7317634)]
    results = []
    with _cpu_scope(42, 1):
        batch = synthetic_batch()
        for name, cls, cfg, detail, cap in candidates:
            started = time.perf_counter()
            model = cls(cfg).eval()
            construction = time.perf_counter() - started
            count = sum(p.numel() for p in model.parameters())
            record = dict(name=name, config=asdict(cfg), config_sha256=canonical_fingerprint(asdict(cfg)),
                parameters=count, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                buffers=sum(p.numel() for p in model.buffers()), weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()),
                hypothetical_fp32_ema_bytes=4*count, ema_enabled=False, cap=cap,
                budget_status="REFERENCE_ONLY" if cap is None else "WITHIN_CAP" if count <= cap else "OVER_CAP",
                construction_seconds=construction, native_activation_peak_bytes=None)
            if detail:
                record["layer_parameters"] = [{"name": n, "parameters": sum(p.numel() for p in m.parameters(recurse=False))}
                    for n,m in model.named_modules() if any(True for _ in m.parameters(recurse=False))]
                linear_macs = [0]
                def linear_hook(module, args, out):
                    linear_macs[0] += out.numel() * module.in_features
                hooks = [m.register_forward_hook(linear_hook) for m in model.modules() if isinstance(m, torch.nn.Linear)]
                with torch.no_grad(): output = model(batch)
                for h in hooks: h.remove()
                attention_macs = (sum(2*n*n*cfg.hidden*cfg.blocks for n in (3,2)) if cls is PPGT else
                    sum(2*e*e*cfg.hidden*(len(cfg.layer_types)-1) + 2*cfg.seeds*e*cfg.hidden for e in (4,2)))
                record["synthetic_batch_macs"] = {"linear": linear_macs[0], "attention_qk_and_av": attention_macs,
                    "sum": linear_macs[0]+attention_macs, "rough_flops_twice_macs": 2*(linear_macs[0]+attention_macs),
                    "excluded": "norm/embedding/activation/SPE/softmax and CPU feature derivation; not an exact FLOP counter"}
                profile = RuntimeProfile()
                profile.metadata.update(scope="CPU_SYNTHETIC_INFERENCE_ONLY", graph_nodes=[3,2], directed_edges=[4,2])
                with torch.no_grad():
                    for _ in range(3):
                        with profile.measure("validation"): model(batch)
                record["cpu_profile"] = profile.to_dict()
                record["median_inference_seconds"] = statistics.median(profile.samples["validation"])
                assert output.shape == (2,) and bool(torch.isfinite(output).all())
            results.append(record)
            del model
            gc.collect()
    sources = {}
    for module in ("ppgt", "esa"):
        closure = discover_source_dependencies(root, [f"src/molgap/{module}_model.py"], require_static=True)
        sources[module] = {**closure.to_dict(), "raw_file_sha256": {p:sha256_file(root / p) for p in closure.source_files}}
    main = next(r for r in results if r["name"] == "ppgt_deep_256")["parameters"] + next(r for r in results if r["name"] == "esa_mixed")["parameters"]
    assert main <= 17811268
    return {"format":"molgap-modern2d-structural-audit-v1", "scope":"CPU_SYNTHETIC_ONLY",
            "torch_version":str(torch.__version__), "results":results, "main_combined_parameters":main,
            "combined_cap":17811268, "sources":sources,
            "feature_formula_examples":{f"n{n}":{"ppgt":ppgt_cost(n), "esa_chain":esa_cost(2*(n-1))} for n in (16,64,128)},
            "native_memory_throughput":"NOT_MEASURED", "real_size_distribution":"NOT_READ",
            "scientific_training_authority":"NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING"}
