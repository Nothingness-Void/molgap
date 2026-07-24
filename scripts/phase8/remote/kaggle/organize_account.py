"""Normalize private Kaggle dataset metadata without changing dataset slugs.

The command is dry-run by default. Pass ``--apply`` to update metadata in place.
No dataset files or versions are uploaded, and existing notebook references keep
working because owner/slug identifiers are preserved.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


DATASETS = {
    "nothingnessvoid/1m-full": {
        "title": "MolGap | Reference | Original 1M Assets",
        "subtitle": "Unpromoted 1M continuation assets retained for reproducibility",
        "description": "Reference-only Phase 8 bundle: original 1M CSV, 2D embeddings, ETKDG 3D graphs, SchNet checkpoint, and dual-GPS fusion checkpoint. The 1M candidate failed the global acceptance gate and is not the production model.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-1m-external-eval-models": {
        "title": "MolGap | Eval | 1M Model Assets",
        "subtitle": "Frozen checkpoints used by paired Phase 8 acceptance evaluations",
        "description": "Evaluation-only checkpoint bundle for original 500K, original 1M, repair-v2, SchNet, GPS7, GPS9, and fusion comparisons. Keep private and immutable for reproducible paired metrics.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-1m-replay-fusion-model": {
        "title": "MolGap | Archive | Replay Fusion Negative",
        "subtitle": "Rejected replay-weighted 1M fusion checkpoint",
        "description": "Archived negative experiment. Replay-weighted frozen-embedding fusion did not repair the 1M PCQM regression and must not be used as a production model.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-pcqm4mv2-valid-5k": {
        "title": "MolGap | Eval | PCQM4Mv2 Valid 5K",
        "subtitle": "Fixed external Gap proxy labels for paired acceptance tests",
        "description": "Immutable 5K PCQM4Mv2 validation subset used only for paired external Gap evaluation. This is a local proxy test, not an OGB leaderboard submission.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-1m-external-eval": {
        "title": "MolGap | Eval | Common OOD and Hard Labels",
        "subtitle": "Fixed common, OOD-1000, and P8-hard labeled evaluation table",
        "description": "Immutable B3LYP evaluation labels used for paired Phase 8 model comparisons. The common table is development evidence after residual-guided acquisition and is not a newly sealed promotion set.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-runtime-source": {
        "title": "MolGap | Active | Phase 8 Runtime Source",
        "subtitle": "Reusable inference and evaluation source bundle for cloud jobs",
        "description": "Small private source bundle used by Kaggle evaluation kernels. Reusable model logic mirrors src/molgap; experiment-specific wrappers remain in scripts/phase8/remote/kaggle.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-phase8-repair-v2-checkpoints": {
        "title": "MolGap | Archive | Repair V2 Fetch Checkpoints",
        "subtitle": "Durable acquisition rounds retained for reconstruction only",
        "description": "Archived repair-v2 acquisition checkpoints. The reconciled 726,966-row candidate union supplied the rejected repair-v2 and additive 1.5M experiments. Do not treat these files as a new training set.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/pyg-3d-graphs-etkdg-300k": {
        "title": "MolGap | Archive | ETKDG 3D Graphs 300K",
        "subtitle": "Historical Phase 7/8 ETKDG graph cache",
        "description": "Archived 300K ETKDG PyG graph cache retained for historical reproduction. Current scaling experiments use the named Phase 8 caches instead.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/pyg-2d-graphs-bond-300k": {
        "title": "MolGap | Archive | Bond 2D Graphs 300K",
        "subtitle": "Historical Phase 7/8 molecular graph cache",
        "description": "Archived 300K bond-aware PyG graph cache retained for historical reproduction. It is not the current 1M training cache.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-hybrid": {
        "title": "MolGap | Archive | Legacy Hybrid Graph Cache",
        "subtitle": "Historical 300K hybrid experiment asset",
        "description": "Legacy descriptor-enhanced 3D graph asset from the early hybrid experiment. Retained only for Phase 7/early Phase 8 reproducibility.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-2m-fetch-base": {
        "title": "MolGap | Reference | Acquisition Exclusion Base",
        "subtitle": "Rejected 1.5M table plus prior repair-union identities",
        "description": "Reference bundle for future acquisition exclusion. It contains the rejected additive 1.5M labeled table and the 726,966-row prior repair identity union. Despite the historical slug, this is not a completed 2M dataset.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
    "nothingnessvoid/molgap-residual-target-round01-recovered": {
        "title": "MolGap | Active | Residual Target Round 01",
        "subtitle": "Recovered 45,457-row residual-focused B3LYP acquisition",
        "description": "Recovered residual-target acquisition with 45,457 unique labeled molecules. It contributes to the controlled original-1M plus 97,798-row uniform pilot; use the newer scaffold-sealed split for promotion evaluation.",
        "keywords": ["chemistry", "graph neural network", "pre-trained model"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Update Kaggle metadata")
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def read_downloaded_metadata(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw) if isinstance(raw, str) else raw


def metadata_matches(api: KaggleApi, ref: str, desired: dict, root: Path) -> bool:
    verify = root / "verify"
    verify.mkdir(exist_ok=True)
    api.dataset_metadata(ref, str(verify))
    current = read_downloaded_metadata(verify / "dataset-metadata.json")
    return all(values_match(key, current.get(key), value) for key, value in desired.items())


def values_match(key: str, current, desired) -> bool:
    if key == "keywords":
        return set(current or []) == set(desired or [])
    return current == desired


def main() -> None:
    args = parse_args()
    api = KaggleApi()
    api.authenticate()
    rows = []
    with tempfile.TemporaryDirectory(prefix="molgap-kaggle-metadata-") as temporary:
        root = Path(temporary)
        for ref, desired in DATASETS.items():
            target = root / ref.rsplit("/", 1)[1]
            target.mkdir()
            api.dataset_metadata(ref, str(target))
            metadata_path = target / "dataset-metadata.json"
            current = read_downloaded_metadata(metadata_path)
            before = {key: current.get(key) for key in ("title", "subtitle", "description", "keywords")}
            current.update(desired)
            metadata_path.write_text(json.dumps(json.dumps(current)), encoding="utf-8")
            changed = not all(values_match(key, before.get(key), value) for key, value in desired.items())
            if args.apply and changed:
                try:
                    api.dataset_metadata_update(ref, str(target))
                except TypeError:
                    # Kaggle API 1.8.2 can parse a successful empty-errors response
                    # as a string. Verify the remote state instead of trusting it.
                    if not metadata_matches(api, ref, desired, target):
                        raise
            rows.append({"ref": ref, "changed": changed, "applied": bool(args.apply and changed), **desired})
            print(f"{'APPLY' if args.apply else 'DRY'} {ref}: {desired['title']}", flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"applied": args.apply, "datasets": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
