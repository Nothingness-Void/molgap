"""Canonical identity compatibility for the accepted OGB silicon exceptions.

This is NOT a chemistry repair or a graph builder. The existing scale-cache
fallback preserves source topology with sanitize=False/strict=False. Here that
strategy is used only for identity, limited to the frozen, already audited
structures; an unknown structure never acquires an accepted identity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .pcqm_k1_scale import (
    UNSANITIZED_OGB_INDEX_SHA256, UNSANITIZED_OGB_SOURCE_INDICES, index_sha256,
)
from .research_memory.trace import file_digest


COMPAT_MOLECULE_RULE = "rdkit-canonical-isomeric-smiles-clear-atom-maps-known-ogb-si-v2"


def _key(mol):
    from rdkit import Chem
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _silicon_fallback(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("unparsable SMILES syntax")
    problems = Chem.DetectChemistryProblems(mol)
    if not problems or any(
        p.GetType() != "AtomValenceException" or
        mol.GetAtomWithIdx(p.GetAtomIdx()).GetAtomicNum() != 14
        for p in problems
    ):
        raise ValueError("not exclusively a known-class Si valence exception")
    # Exactly the accepted cache strategy; no charge/bond/H/stereo correction.
    mol.UpdatePropertyCache(strict=False)
    return mol


def load_fallback_keys(path, expected_sha256):
    """Pin the old audit or an independently graph-verified same-class ledger."""
    path = Path(path)
    if file_digest(path) != expected_sha256:
        raise ValueError("ROLE_BLOCKED: fallback ledger SHA differs")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    entries = ledger["entries"]
    indices = [e["source_idx"] for e in entries]
    if ledger.get("format") == "molgap-pcqm-unified-accepted-si-ledger-v1":
        proof_path = path.with_name("exception_graph_audit.json")
        if file_digest(proof_path) != ledger["graph_proof_sha256"]:
            raise ValueError("ROLE_BLOCKED: exception graph proof SHA differs")
        proof = json.loads(proof_path.read_text())
        for entry in proof["entries"]:
            if entry.get("accepted_si_identity") is True and (
                    entry.get("identity_stable") is not True or
                    entry.get("feature_matches") != {"x":True,"edge_index":True,"edge_attr":True}):
                raise ValueError("ROLE_BLOCKED: incomplete graph/identity proof")
        verified = [{k:e[k] for k in ("source_idx","smiles")} for e in proof["entries"]
                    if e.get("accepted_si_identity") is True]
        if (not entries or entries != verified or indices != sorted(set(indices)) or
                any(proof.get(k) != ledger.get(k) for k in
                    ("source_manifest_sha256","source_acceptance_sha256","identity_audit_sha256"))):
            raise ValueError("ROLE_BLOCKED: expanded Si ledger differs from verified evidence")
        return frozenset(_key(_silicon_fallback(e["smiles"])) for e in entries)
    if (ledger.get("format") != "molgap-pcqm-k1-scale500k-parser-fallback-audit-v1" or
            tuple(indices) != UNSANITIZED_OGB_SOURCE_INDICES or
            index_sha256(indices) != UNSANITIZED_OGB_INDEX_SHA256 or
            ledger.get("ordered_index_sha256") != UNSANITIZED_OGB_INDEX_SHA256):
        raise ValueError("ROLE_BLOCKED: not the accepted OGB exception ledger")
    return frozenset(_key(_silicon_fallback(e["smiles"])) for e in entries)


def molecular_identity(smiles, *, fallback_keys):
    """Return (canonical key, anomaly evidence); no fallback flag in the key.

    Known graph-isomorphic aliases are covered too: restricting by source index
    or raw SMILES alone would miss duplicates across training/holdout roles.
    Unknown anomalies return None for a complete audit, NEVER a made-up key.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return _key(mol), None
    evidence = {"smiles_sha256": hashlib.sha256(smiles.encode()).hexdigest(),
                "status": "BLOCKED", "canonical_sha256": None}
    try:
        mol = _silicon_fallback(smiles)
        key = _key(mol)
        evidence["canonical_sha256"] = hashlib.sha256(key.encode()).hexdigest()
        if key not in fallback_keys:
            raise ValueError("Si structure absent from the frozen accepted ledger")
        # A second ordering and a serialization round trip must retain identity.
        reordered = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        round_trip = Chem.MolFromSmiles(key, sanitize=False)
        if round_trip is None:
            raise ValueError("canonical identity cannot round trip")
        round_trip.UpdatePropertyCache(strict=False)
        if (_key(reordered) != key or _key(round_trip) != key or
                (round_trip.GetNumAtoms(), round_trip.GetNumBonds()) !=
                (mol.GetNumAtoms(), mol.GetNumBonds())):
            raise ValueError("unstable fallback molecular identity")
        evidence.update(status="KNOWN_FALLBACK", reason="accepted OGB Si topology; no chemistry edits")
        return key, evidence
    except ValueError as exc:
        evidence["reason"] = str(exc)
        return None, evidence


def audit_exception_graphs(*, boundary, fixed_root, identity_root, output_root, rule, audit_sha256):
    """CPU-only read-only proof against accepted graph bytes; never repair them.

    Only individually verified Si structures enter the expanded ledger. Other
    chemistry classes remain blocked even if their cached graph is readable.
    """
    import csv
    import gzip
    import itertools
    import os
    import re
    import torch
    from .common.io import publish_immutable_bytes
    from .pcqm_unified_roles import _safe, _pin, validate_rule
    from .pcqm_fixed_datasets import _inside
    from .packed_graph_dataset import PackedGraphDataset
    from .pcqm_k1_scale_cache import _unsanitized_ogb_payload
    from .research_memory.trace import json_bytes
    from rdkit import Chem, rdBase

    fixed, identity, output = (_safe(boundary,p) for p in (fixed_root,identity_root,output_root))
    if _inside(output,fixed) or _inside(output,identity) or _inside(fixed,output) or _inside(identity,output):
        raise ValueError("Exception proof requires a separate confined output")
    if rdBase.rdkitVersion != rule["rdkit_version"]:
        raise ValueError("Frozen molecular canonicalization runtime changed")
    for name,key in (("manifest.json","source_manifest_sha256"),("acceptance.json","source_acceptance_sha256")):
        if file_digest(fixed/name) != rule[key]:
            raise ValueError("Old fixed dataset identity changed")
    if json.loads((identity/"rule.json").read_text()) != rule:
        raise ValueError("Identity audit rule changed")
    audit_path=identity/"molecular_identity_audit.json"
    if file_digest(audit_path) != audit_sha256:
        raise ValueError("Identity audit SHA changed")
    audit=json.loads(audit_path.read_text())
    if audit["rule_identity"] != validate_rule(rule) or audit["completed_rows"] != rule["official_train_rows"]:
        raise ValueError("Complete bound official-train audit required")
    source=json.loads((fixed/"manifest.json").read_text())
    wanted={e["source_idx"]:e for e in audit["entries"]}
    if not wanted or len(wanted)!=len(audit["entries"]) or any(type(i) is not int or not 0<=i<rule["official_train_rows"] for i in wanted):
        raise ValueError("Exception outside official training membership")
    molecules={}
    for record in source["store_assets"]["source"]:
        match=re.fullmatch(r"source_(\d+)_(\d+)\.csv\.gz",Path(record["file"]).name)
        if not match:
            continue
        start,stop=map(int,match.groups())
        selected={i for i in wanted if start<=i<stop}
        if not selected:
            continue
        with gzip.open(_pin(fixed,record),"rt",newline="") as stream:
            # Never ask the reader for an official-validation row, including
            # in the mixed terminal source chunk. No label value is exported.
            for row in itertools.islice(csv.DictReader(stream),min(max(selected)+1,rule["official_train_rows"])-start):
                idx=int(row["idx"])
                if idx in selected:
                    smiles=row["smiles"]
                    if hashlib.sha256(smiles.encode()).hexdigest()!=wanted[idx]["smiles_sha256"]:
                        raise ValueError("Audited source SMILES changed")
                    molecules[idx]=(smiles,record["sha256"])
    if set(molecules)!=set(wanted):
        raise ValueError("Missing exception source membership")
    entries=[]
    for record in source["store_assets"]["topology"]:
        if record.get("role")!="train":
            continue
        start,stop=record["source_idx_min"],record["source_idx_max"]+1
        selected=sorted(i for i in wanted if start<=i<stop)
        if not selected:
            continue
        shard=PackedGraphDataset(_pin(fixed,record),mmap=True)
        for idx in selected:
            smiles,source_sha=molecules[idx]
            entry={"source_idx":idx,"smiles":smiles,"source_chunk_sha256":source_sha,
                   "topology_chunk_sha256":record["sha256"],"accepted_si_identity":False}
            raw=Chem.MolFromSmiles(smiles,sanitize=False)
            problems=[] if raw is None else list(Chem.DetectChemistryProblems(raw))
            entry["chemistry_problems"]=[
                {"type":p.GetType(),"message":p.Message()} for p in problems]
            try:
                if raw is None or not problems or any(p.GetType()!="AtomValenceException" for p in problems):
                    raise ValueError("Not a valence-only source exception")
                mol=raw;mol.UpdatePropertyCache(strict=False);key=_key(mol)
                if wanted[idx]["canonical_sha256"] is not None and hashlib.sha256(key.encode()).hexdigest()!=wanted[idx]["canonical_sha256"]:
                    raise ValueError("Canonical identity differs from the original full audit")
                for shift in range(mol.GetNumAtoms()):
                    order=list(range(shift,mol.GetNumAtoms()))+list(range(shift))
                    if _key(Chem.RenumberAtoms(mol,order))!=key:
                        raise ValueError("Atom-order-dependent canonical identity")
                round_trip=Chem.MolFromSmiles(key,sanitize=False)
                if round_trip is None:
                    raise ValueError("Identity round-trip cannot parse")
                round_trip.UpdatePropertyCache(strict=False)
                if _key(round_trip)!=key or (round_trip.GetNumAtoms(),round_trip.GetNumBonds())!=(mol.GetNumAtoms(),mol.GetNumBonds()):
                    raise ValueError("Identity round-trip verification failed")
                payload=_unsanitized_ogb_payload(smiles)
                graph=shard.get(idx-start)
                matches={name:torch.equal(getattr(graph,name),torch.as_tensor(payload[field],dtype=torch.long))
                         for name,field in (("x","node_feat"),("edge_index","edge_index"),("edge_attr","edge_feat"))}
                entry.update(feature_matches=matches,identity_stable=True,canonical_smiles=key)
                if (int(graph.source_idx.item())!=idx or graph.num_nodes!=payload["num_nodes"] or not all(matches.values())):
                    raise ValueError("Accepted graph atom/bond tensors differ from the unchanged source fallback")
                if any(mol.GetAtomWithIdx(p.GetAtomIdx()).GetAtomicNum()!=14 for p in problems):
                    raise ValueError("Non-Si valence exception: separate authority required even with matching graph")
                entry["accepted_si_identity"]=True
            except (ValueError,RuntimeError) as exc:
                entry["reason"]=str(exc)
            entries.append(entry)
        del shard
    entries.sort(key=lambda e:e["source_idx"])
    if [e["source_idx"] for e in entries]!=sorted(wanted):
        raise ValueError("Missing or duplicate topology proof membership")
    verified=[{k:e[k] for k in ("source_idx","smiles")} for e in entries if e["accepted_si_identity"]]
    report={"format":"molgap-pcqm-exception-graph-proof-v1","status":"ALL_EXCEPTIONS_VERIFIED" if len(verified)==len(entries) else "PARTIAL_SI_PROOF_ONLY",
            "source_manifest_sha256":rule["source_manifest_sha256"],"source_acceptance_sha256":rule["source_acceptance_sha256"],
            "identity_audit_sha256":audit_sha256,"verifier_sha256":file_digest(Path(__file__)),"entries":entries,
            "verified_si_rows":len(verified),"unresolved_rows":len(entries)-len(verified),
            "graphs_rebuilt":False,"source_labels_modified":False,"gpu_used":False,
            "official_validation_read":False,"test_roles_read":False,"data_role_accepted":False}
    output.mkdir(parents=True,exist_ok=True)
    if os.name!="nt":os.chmod(output,0o700)
    publish_immutable_bytes(output/"exception_graph_audit.json",json_bytes(report))
    if verified:
        ledger={"format":"molgap-pcqm-unified-accepted-si-ledger-v1","entries":verified,
                **{k:report[k] for k in ("source_manifest_sha256","source_acceptance_sha256","identity_audit_sha256")},
                "graph_proof_sha256":file_digest(output/"exception_graph_audit.json")}
        publish_immutable_bytes(output/"fallback_ledger.json",json_bytes(ledger))
    return report
