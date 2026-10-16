"""Versioned indexed roles over accepted PCQM graphs; no graph construction.

CPU preparation is separate from training. The SQLite identity inventory and
audit payload stay in the private preparation root, never the training export.
Old fixed views and the existing training ABI remain unchanged.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from functools import partial
from pathlib import Path
import sqlite3
import struct

from .common.io import publish_immutable_bytes
from .common.wire import canonical_fingerprint
from .cross_scale_plan import digest, exact, number
from .pcqm_fixed_datasets import sha256_file, _inside
from .research_memory.paths import repo_local_path
from .research_memory.trace import atomic_write, json_bytes
from .pcqm_molecular_identity import COMPAT_MOLECULE_RULE, EXPLICIT_H_MOLECULE_RULE, TWO_P_MOLECULE_RULE, molecular_identity, load_fallback_keys


FORMAT = "molgap-pcqm4mv2-unified-roles-v2"
ALGORITHM = "sha256-molecule-group-rank-v1"
MOLECULE_RULE = "rdkit-canonical-isomeric-smiles-clear-atom-maps-v1"
TRAIN_ROLES = ("train_100k", "train_500k", "train_1m", "full_minus_holdout")
ALL_ROLES = (*TRAIN_ROLES, "common_dev", "independent_audit")
GRAPH_FIELDS = frozenset({"x", "edge_index", "edge_attr", "random_walk_pe", "source_idx", "y"})
SOURCE_METADATA_FIELDS = frozenset({"smiles_sanitized", "split_code"})
COMPATIBLE_RULES = {COMPAT_MOLECULE_RULE, EXPLICIT_H_MOLECULE_RULE, TWO_P_MOLECULE_RULE}
EXPLICIT_H_RULES = {EXPLICIT_H_MOLECULE_RULE, TWO_P_MOLECULE_RULE}


def freeze_role_inventory(repo_root):
    """Freeze existing canonical RML events plus accepted fixed-role manifests.

    Ambiguous historical internal development is conservatively excluded as
    the UNION of the three documented fixed roles, never guessed as one row.
    Any other role name is unresolved and blocks actual generation.
    """
    from .research_memory.validate import validate_repository_records
    root = Path(repo_root).resolve()
    records = validate_repository_records(root)["records"]
    pins, exclusions, unresolved = [], set(), []
    fixed = root/"platforms/_records/ims/pcqm_fixed_datasets_v1"
    for path in sorted(fixed.glob("*.manifest.json")):
        manifest = json.loads(path.read_text())
        pins.append({"path":path.relative_to(root).as_posix(),"sha256":sha256_file(path)})
        role = manifest.get("roles",{}).get("development")
        if role:
            exclusions.add((role["source_idx_start"],role["source_idx_stop"]))
    aliases = {
        "internal_development_100k":(100000,150000),
        "internal_development_100k_first_128_rows":(100000,150000),
        "internal_development_100000_150000":(100000,150000),
        "internal_development_500k":(500000,550000),
        "internal_development_500k_550k":(500000,550000),
        "internal_development_500000_550000":(500000,550000),
    }
    non_evaluation={"train","internal_train","internal_train_0_500k","official_train_prefix_0_100000", "official_train_prefix_0_500000"}
    outside_pool={"official_validation","test_dev","test_challenge"}
    for kind in ("roles","evidence"):
        for path, record in records[kind]:
            pins.append({"path":path.relative_to(root).as_posix(),"sha256":sha256_file(path)})
            names=[record["role_name"]] if kind=="roles" else list(record.get("role_use",{}))
            for name in names:
                if name in aliases:
                    exclusions.add(aliases[name])
                elif name == "internal_development":
                    exclusions.update(((100000,150000),(500000,550000),(1000000,1050000)))
                elif name not in non_evaluation | outside_pool:
                    unresolved.append({"path":path.relative_to(root).as_posix(),"role":name})
    return {"format":"molgap-pcqm-unified-role-inventory-v2","pins":pins,
        "excluded_ranges":[list(x) for x in sorted(exclusions)],"unresolved_roles":unresolved,
        "status":"ROLE_BLOCKED" if unresolved else "METADATA_ROLE_INVENTORY_FROZEN",
        "ambiguous_development_resolution":"conservative union of all documented fixed development roles",
        "old_full_already_consumed_entire_official_train":True,"official_labels_opened":False}


def validate_rule(rule):
    extra = {"fallback_ledger_sha256", "molecule_parser_sha256"} if rule.get("molecule_rule") in COMPATIBLE_RULES else set()
    exact(rule, {"format", "algorithm", "molecule_rule", "rdkit_version", "seed", "official_train_rows",
        "train_sizes", "common_dev_rows", "independent_audit_rows", "excluded_ranges",
        "source_manifest_sha256", "source_acceptance_sha256", "role_inventory_sha256",
        "generator_sha256", "source_commit", "scope"} | extra, "unified rule")
    if (rule["format"] != FORMAT or rule["algorithm"] != ALGORITHM or rule["molecule_rule"] not in {MOLECULE_RULE, *COMPATIBLE_RULES} or
            rule["scope"] not in {"CPU_SYNTHETIC_ONLY", "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY"}):
        raise ValueError("Unknown immutable dataset rule")
    for key in ("seed", "official_train_rows", "common_dev_rows", "independent_audit_rows"):
        number(rule[key], key, integer=True, minimum=0 if key == "seed" else 1)
    if (type(rule["train_sizes"]) is not list or len(rule["train_sizes"]) != 3 or
            rule["train_sizes"] != sorted(set(rule["train_sizes"]))):
        raise ValueError("Three increasing nested training sizes required")
    for size in rule["train_sizes"]:
        number(size, "train size", integer=True, minimum=1)
    if rule["train_sizes"][-1] >= rule["official_train_rows"]:
        raise ValueError("No eligible official train pool remains")
    for start, stop in rule["excluded_ranges"]:
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= rule["official_train_rows"]:
            raise ValueError("Excluded role outside official train")
    for key in ("source_manifest_sha256", "source_acceptance_sha256", "role_inventory_sha256", "generator_sha256"):
        digest(rule[key], key)
    for key in extra:
        digest(rule[key], key)
    if type(rule["rdkit_version"]) is not str or not rule["rdkit_version"]:
        raise ValueError("Frozen RDKit identity required")
    if (type(rule["source_commit"]) is not str or len(rule["source_commit"]) != 40 or
            any(c not in "0123456789abcdef" for c in rule["source_commit"])):
        raise ValueError("Source commit required")
    return canonical_fingerprint(rule)


def canonical_molecule(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("ROLE_BLOCKED: unparsable molecular identity")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _molecule_task(row, *, compatible=False, fallback_keys=(), explicit_hydrogens=False, allow_phosphorus=False):
    idx, smiles, target, seed = row
    key, anomaly = molecular_identity(smiles, fallback_keys=fallback_keys,explicit_hydrogens=explicit_hydrogens,
                                     allow_phosphorus=allow_phosphorus) if compatible else (canonical_molecule(smiles), None)
    molecule = hashlib.sha256(key.encode()).hexdigest() if key is not None else None
    rank = hashlib.sha256(f"{ALGORITHM}:{seed}:{molecule}".encode()).hexdigest() if molecule is not None else None
    return idx, molecule, rank, target, anomaly


def _excluded(idx, rule):
    return idx < rule["train_sizes"][-1] or any(a <= idx < b for a, b in rule["excluded_ranges"])


def inventory_rows(database, rows, rule, *, map_rows=map, fallback_keys=()):
    """One transaction per bounded source shard; a crash cannot acknowledge half.

    SQLite insertion fails on repeated source IDs. The driver skips only a
    hash-bound completed source shard, not a guessed row cursor.
    """
    validate_rule(rule)
    from rdkit import rdBase
    if rdBase.rdkitVersion != rule["rdkit_version"]:
        raise ValueError("Frozen molecular canonicalization runtime changed")
    database.execute("CREATE TABLE IF NOT EXISTS rows (idx INTEGER PRIMARY KEY, molecule TEXT, rank TEXT, target REAL, excluded INTEGER)")
    database.execute("CREATE TABLE IF NOT EXISTS molecular_anomalies (idx INTEGER PRIMARY KEY, evidence TEXT NOT NULL)")
    tasks = ((idx, smiles, target, rule["seed"]) for idx, smiles, target in rows)
    task = partial(_molecule_task, compatible=rule["molecule_rule"] in COMPATIBLE_RULES, fallback_keys=fallback_keys,
                   explicit_hydrogens=rule["molecule_rule"] in EXPLICIT_H_RULES,
                   allow_phosphorus=rule["molecule_rule"] == TWO_P_MOLECULE_RULE)
    for idx, molecule, rank, target, anomaly in map_rows(task, tasks):
        number(idx, "source index", integer=True)
        if idx >= rule["official_train_rows"]:
            raise ValueError("ROLE_BLOCKED: non-training source index")
        number(target, "Gap label")
        database.execute("INSERT INTO rows VALUES (?,?,?,?,?)", (idx, molecule, rank, target, int(_excluded(idx, rule))))
        if anomaly is not None:
            database.execute("INSERT INTO molecular_anomalies VALUES (?,?)", (idx, json.dumps(anomaly, sort_keys=True)))


def molecular_audit(database, rule):
    total, valid = database.execute("SELECT COUNT(*),COUNT(molecule) FROM rows").fetchone()
    entries = [{"source_idx": idx, **json.loads(raw)} for idx, raw in
               database.execute("SELECT idx,evidence FROM molecular_anomalies ORDER BY idx")]
    blocked = sum(e["status"] == "BLOCKED" for e in entries)
    complete = total == rule["official_train_rows"]
    return {"rule_identity": validate_rule(rule), "completed_rows": total, "valid_identity_rows": valid,
            "status": "IDENTITIES_VERIFIED" if complete and valid == total and not blocked else
                      "ROLE_BLOCKED" if blocked else "INCOMPLETE",
            "known_fallback_rows": sum(e["status"] == "KNOWN_FALLBACK" for e in entries),
            "blocked_rows": blocked, "entries": entries, "rows_skipped_or_replaced": 0,
            "official_validation_read": False, "test_roles_read": False, "gpu_used": False}


def choose_roles(database, rule):
    """Hash-rank unique molecule groups; selection never receives predictions."""
    validate_rule(rule)
    if database.execute("SELECT COUNT(*) FROM rows WHERE molecule IS NULL OR rank IS NULL").fetchone()[0]:
        raise ValueError("ROLE_BLOCKED: unresolved molecular identities; inspect molecular_identity_audit.json")
    count, lo, hi = database.execute("SELECT COUNT(*),MIN(idx),MAX(idx) FROM rows").fetchone()
    if (count, lo, hi) != (rule["official_train_rows"], 0, rule["official_train_rows"]-1):
        raise ValueError("ROLE_BLOCKED: official training membership is incomplete")
    database.execute("CREATE INDEX IF NOT EXISTS molecule_idx ON rows(molecule)")
    needed = rule["common_dev_rows"] + rule["independent_audit_rows"]
    groups = _ranked_molecule_groups(database, rule)
    if len(groups) != needed:
        raise ValueError("ROLE_BLOCKED: insufficient disjoint, unique molecular groups")
    dev = sorted(r[0] for r in groups[:rule["common_dev_rows"]])
    audit = sorted(r[0] for r in groups[rule["common_dev_rows"]:])
    held_molecules = {r[1] for r in groups}
    # Every copy of a held molecule is excluded from Full, not only its chosen row.
    full = [idx for idx, molecule in database.execute("SELECT idx,molecule FROM rows ORDER BY idx") if molecule not in held_molecules]
    roles = {name: list(range(size)) for name, size in zip(TRAIN_ROLES[:3], rule["train_sizes"])}
    roles.update(common_dev=dev, independent_audit=audit, full_minus_holdout=full)
    validate_role_members(database, roles, rule)
    return roles


def _ranked_molecule_groups(database, rule):
    return database.execute("SELECT MIN(idx), molecule, MIN(rank) FROM rows GROUP BY molecule HAVING MAX(excluded)=0 ORDER BY MIN(rank),MIN(idx) LIMIT ?",
        (rule["common_dev_rows"]+rule["independent_audit_rows"],)).fetchall()


def validate_role_members(database, roles, rule):
    if set(roles) != set(ALL_ROLES):
        raise ValueError("Missing dataset roles")
    for name, ids in roles.items():
        if type(ids) is not list or not ids or ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < rule["official_train_rows"] for i in ids):
            raise ValueError("Source index ordering/membership tampered: " + name)
    for name, size in zip(TRAIN_ROLES[:3], rule["train_sizes"]):
        if roles[name] != list(range(size)):
            raise ValueError("Preserved training identity changed")
    sets = {k: set(v) for k, v in roles.items()}
    if not sets["train_100k"] < sets["train_500k"] < sets["train_1m"] <= sets["full_minus_holdout"]:
        raise ValueError("Training cohorts not nested")
    for role, count in (("common_dev", rule["common_dev_rows"]), ("independent_audit", rule["independent_audit_rows"])):
        if len(roles[role]) != count or any(_excluded(i, rule) for i in roles[role]):
            raise ValueError("ROLE_BLOCKED: reserved role contains consumed/training members")
        if any(sets[role] & sets[t] for t in TRAIN_ROLES) or sets["common_dev"] & sets["independent_audit"]:
            raise ValueError("ROLE_BLOCKED: role intersection")
    molecules = {i: m for i, m in database.execute("SELECT idx,molecule FROM rows")}
    if len(molecules)!=rule["official_train_rows"] or any(m is None for m in molecules.values()):
        raise ValueError("ROLE_BLOCKED: incomplete molecular membership")
    dev_m, audit_m = ({molecules[i] for i in roles[r]} for r in ("common_dev", "independent_audit"))
    train_m = {molecules[i] for i in roles["full_minus_holdout"]}
    if (len(dev_m) != len(roles["common_dev"]) or len(audit_m) != len(roles["independent_audit"]) or
            dev_m & audit_m or (dev_m | audit_m) & train_m):
        raise ValueError("ROLE_BLOCKED: exact duplicate molecular leakage")
    held_molecules=dev_m | audit_m
    expected_full=[i for i in sorted(molecules) if molecules[i] not in held_molecules]
    if roles["full_minus_holdout"]!=expected_full:
        raise ValueError("ROLE_BLOCKED: Full is not the exact held-molecule complement")
    return {"nested": True, "train_development_intersection": 0, "train_audit_intersection": 0,
            "development_audit_intersection": 0, "cross_role_exact_molecule_intersection": 0,
            "old_full_independent_audit_applicability": "NOT_APPLICABLE_ALREADY_TRAINED_ON_ROWS"}


def _index_bytes(ids):
    return b"".join(struct.pack("<q", i) for i in ids)


def _safe(boundary, path):
    path = Path(path).resolve()
    if not _inside(path, Path(boundary)) or path == Path(boundary).resolve():
        raise ValueError("Path outside authorized dataset boundary")
    return path


def _pin(root, record):
    path = repo_local_path(root, record["file"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("Source content SHA mismatch: " + record["file"])
    return path


def inspect_unified_source_topology(*, boundary, fixed_root, output_root, rule):
    """Observe one SHA-pinned train shard's ABI; no role or data acceptance."""
    import os
    from .packed_graph_dataset import PackedGraphDataset
    fixed,output=(_safe(boundary,p) for p in (fixed_root,output_root))
    if _inside(output,fixed) or _inside(fixed,output):
        raise ValueError("Separate topology observation output required")
    for name,key in (("manifest.json","source_manifest_sha256"),("acceptance.json","source_acceptance_sha256")):
        if sha256_file(fixed/name)!=rule[key]:raise ValueError("Old fixed dataset identity changed")
    source=json.loads((fixed/'manifest.json').read_text())
    record=next(r for r in source['store_assets']['topology'] if r.get('role')=='train')
    shard=PackedGraphDataset(_pin(fixed,record),mmap=True)
    sample=shard.get(0)
    report={'status':'SOURCE_TOPOLOGY_OBSERVATION_ONLY','source_file':record['file'],
        'source_sha256':record['sha256'],'expected_rows':record['source_idx_max']-record['source_idx_min']+1,
        'observed_rows':len(shard),'packed_fields':sorted(shard._data.keys()),'graph_fields':sorted(sample.keys()),
        'fields':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} if hasattr(v,'shape') else {'type':type(v).__name__}
                  for k,v in sample.to_dict().items()},'official_validation_read':False,'test_roles_read':False,
        'gpu_used':False,'data_role_accepted':False,'graphs_rebuilt':False}
    output.mkdir(parents=True,exist_ok=True)
    if os.name!='nt':os.chmod(output,0o700)
    publish_immutable_bytes(output/'source_topology_observation.json',json_bytes(report))
    return report


def _project_source_graph(graph):
    """Drop only validated source provenance, never model tensors or geometry."""
    import torch
    from torch_geometric.data import Data
    unknown=set(graph.keys())-(GRAPH_FIELDS|SOURCE_METADATA_FIELDS|{"num_nodes"})
    if unknown:raise ValueError("Non-2D/unknown source graph fields: "+str(sorted(unknown)))
    for name in SOURCE_METADATA_FIELDS & set(graph.keys()):
        value=getattr(graph,name)
        allowed=(0,1) if name=="smiles_sanitized" else (0,1,2)
        if not torch.is_tensor(value) or value.dtype!=torch.int8 or value.numel()!=1 or int(value.item()) not in allowed:
            raise ValueError("Invalid source provenance metadata: "+name)
    return Data(**{k:v for k,v in graph.to_dict().items() if k not in SOURCE_METADATA_FIELDS})


def _reuse_complete_identity(boundary, output, rule, reuse):
    """Explicit immutable input copy; existing SQLite chunk recovery stays owner.

    Only source/generator provenance may differ. Parser, rule, data, roles,
    runtime and all scientific identity fields must be exactly unchanged.
    """
    parent=_safe(boundary,reuse["root"])
    if _inside(output,parent) or _inside(parent,output):raise ValueError("Separate identity reuse output required")
    original=json.loads(repo_local_path(parent,'rule.json').read_text())
    if any(original.get(k)!=v for k,v in rule.items() if k not in {'source_commit','generator_sha256'}):
        raise ValueError("Identity reuse scientific/parser/data contract changed")
    audit_path=repo_local_path(parent,'molecular_identity_audit.json');db_path=repo_local_path(parent,'private_identity.sqlite')
    if sha256_file(audit_path)!=reuse['audit_sha256'] or sha256_file(db_path)!=reuse['database_sha256']:
        raise ValueError("Identity reuse source bytes changed")
    if (db_path.with_name(db_path.name+'-wal')).exists():
        raise ValueError("Identity reuse requires a closed checkpoint without WAL")
    db=sqlite3.connect('file:'+db_path.as_posix()+'?mode=ro&immutable=1',uri=True)
    try:
        audit=molecular_audit(db,original)
        if json.loads(audit_path.read_text())!=audit or audit['status']!='IDENTITIES_VERIFIED':
            raise ValueError("Identity reuse requires complete verified membership")
    finally:db.close()
    binding={'parent_rule_identity':validate_rule(original),'rule_identity':validate_rule(rule),
             'parent_root':str(parent),'database_sha256':reuse['database_sha256'],'audit_sha256':reuse['audit_sha256'],
             'scientific_identity_unchanged':True,'status':'IMMUTABLE_IDENTITY_INPUT_ONLY_NOT_DATA_ACCEPTANCE'}
    dest=output/'private_identity.sqlite';receipt=output/'identity_reuse.json'
    if receipt.exists():
        if json.loads(receipt.read_text())!=binding or not dest.exists():raise ValueError("Identity reuse binding changed")
        return
    if dest.exists():
        if sha256_file(dest)!=reuse['database_sha256']:raise ValueError("Unbound identity reuse destination")
    else:
        # Remote CPU only: reuse atomic no-overwrite IO, not a second recovery
        # implementation. These source bytes are not graph/model reconstruction.
        raw=db_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=reuse['database_sha256']:raise ValueError("Identity reuse changed during copy")
        publish_immutable_bytes(dest,raw)
    publish_immutable_bytes(receipt,json_bytes(binding))


def prepare_unified_roles(*, boundary, fixed_root, output_root, rule, workers=1, role_inventory=None,
                          fallback_ledger=None, identity_only=False, identity_reuse=None):
    """IMS CPU owner: reuse accepted row/topology bytes, derive role-only chunks.

    No official validation/test shard is selected. Checkpoints are per source
    shard and per derived graph chunk. Output is private and never overwritten.
    """
    import multiprocessing
    import os
    import time
    from .cross_scale_inputs import index_digest
    boundary, fixed, output = Path(boundary).resolve(), _safe(boundary, fixed_root), _safe(boundary, output_root)
    if output == fixed or _inside(output, fixed):
        raise ValueError("Cannot write into old fixed dataset")
    rule_id = validate_rule(rule)
    fallback_keys = ()
    if rule["molecule_rule"] in COMPATIBLE_RULES:
        from . import pcqm_molecular_identity
        if sha256_file(Path(pcqm_molecular_identity.__file__)) != rule["molecule_parser_sha256"]:
            raise ValueError("Frozen molecular parser bytes changed")
        if fallback_ledger is None:
            raise ValueError("ROLE_BLOCKED: frozen fallback ledger required")
        ledger_path = _safe(boundary, fallback_ledger)
        fallback_keys = load_fallback_keys(ledger_path, rule["fallback_ledger_sha256"],
                                          explicit_hydrogens=rule["molecule_rule"] in EXPLICIT_H_RULES,
                                          allow_phosphorus=rule["molecule_rule"]==TWO_P_MOLECULE_RULE)
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("format") in {"molgap-pcqm-unified-accepted-si-ledger-v1","molgap-pcqm-unified-accepted-si-two-p-ledger-v1"} and any(
                ledger.get(k) != rule[k] for k in ("source_manifest_sha256","source_acceptance_sha256")):
            raise ValueError("ROLE_BLOCKED: Si proof belongs to different source assets")
    if rule["scope"] != "CPU_SYNTHETIC_ONLY" and sha256_file(Path(__file__)) != rule["generator_sha256"]:
        raise ValueError("Actual generator bytes differ from the frozen source")
    if rule["scope"] != "CPU_SYNTHETIC_ONLY":
        if role_inventory is None:
            raise ValueError("ROLE_BLOCKED: hash-bound historical role inventory required")
        inventory_path = _safe(boundary, role_inventory)
        if sha256_file(inventory_path) != rule["role_inventory_sha256"]:
            raise ValueError("ROLE_BLOCKED: historical role inventory SHA differs")
        inventory = json.loads(inventory_path.read_text())
        if (inventory.get("status") != "METADATA_ROLE_INVENTORY_FROZEN" or
                inventory.get("unresolved_roles") != [] or
                inventory.get("excluded_ranges") != rule["excluded_ranges"]):
            raise ValueError("ROLE_BLOCKED: exclusion rule is not bound to accepted role inventory")
    for name, key in (("manifest.json", "source_manifest_sha256"), ("acceptance.json", "source_acceptance_sha256")):
        if sha256_file(fixed/name) != rule[key]:
            raise ValueError("Old fixed dataset identity changed")
    source = json.loads((fixed/"manifest.json").read_text())
    accepted = json.loads((fixed/"acceptance.json").read_text())
    if source.get("status") != "complete" or accepted.get("status") != "accepted" or accepted.get("verify_content") is not True:
        raise ValueError("Source dataset lacks content acceptance")
    output.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(output, 0o700)
    publish_immutable_bytes(output/"rule.json", json_bytes(rule))
    if (output/"acceptance.json").exists():
        previous = json.loads((output/"acceptance.json").read_text())
        if previous["rule_identity"] != rule_id:
            raise ValueError("Accepted dataset identity changed")
        for binding in previous["roles"].values():
            if sha256_file(repo_local_path(output, binding["path"])) != binding["sha256"]:
                raise ValueError("Accepted role manifest identity changed")
        accept_unified_roles(output, rule=rule)
        return previous
    started, cpu_start = time.monotonic(), time.process_time()
    if identity_reuse is not None:
        _reuse_complete_identity(boundary,output,rule,identity_reuse)
    db = sqlite3.connect(output/"private_identity.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (path TEXT PRIMARY KEY, sha TEXT, rows INTEGER)")
    db.commit()
    source_records = [r for r in source["store_assets"]["source"] if r["file"].endswith(".csv.gz")]
    next_idx = 0
    pool = multiprocessing.Pool(workers) if workers > 1 else None
    try:
        for record in source_records:
            if next_idx >= rule["official_train_rows"]:
                break
            path = _pin(fixed, record)
            previous = db.execute("SELECT sha,rows FROM chunks WHERE path=?", (record["file"],)).fetchone()
            if previous is not None:
                if previous[0] != record["sha256"]:
                    raise ValueError("Resume source chunk identity changed")
                next_idx += previous[1]
                continue
            # The last source file may also contain official validation rows.
            # Stop BEFORE asking the CSV reader for the first such row.
            with gzip.open(path, "rt", newline="") as stream:
                reader = csv.DictReader(stream)
                remaining = rule["official_train_rows"] - next_idx
                rows = []
                for row in itertools.islice(reader, remaining):
                    idx = int(row.get("idx", row.get("source_idx", -1)))
                    if idx != next_idx + len(rows):
                        raise ValueError("Source shard order is not official train order")
                    rows.append((idx, row["smiles"], float(row.get("homolumogap", row.get("gap", "nan")))))
                with db:
                    mapper = (lambda f, items: pool.imap(f, items, chunksize=256)) if pool else map
                    inventory_rows(db, rows, rule, map_rows=mapper, fallback_keys=fallback_keys)
                    db.execute("INSERT INTO chunks VALUES (?,?,?)", (record["file"], record["sha256"], len(rows)))
                next_idx += len(rows)
            atomic_write(output/"progress.json", json_bytes({"stage": "MOLECULAR_IDENTITY", "completed_rows": next_idx,
                "rule_identity": rule_id, "status": "RUNNING", "gpu_used": False}))
            if rule["molecule_rule"] in COMPATIBLE_RULES:
                atomic_write(output/"molecular_identity_progress.json", json_bytes(molecular_audit(db, rule)))
        if rule["molecule_rule"] in COMPATIBLE_RULES:
            audit = molecular_audit(db, rule)
            publish_immutable_bytes(output/"molecular_identity_audit.json", json_bytes(audit))
            if audit["status"] != "IDENTITIES_VERIFIED":
                raise ValueError("ROLE_BLOCKED: unresolved molecular identities; inspect molecular_identity_audit.json")
            if identity_only:
                return audit
        elif identity_only:
            raise ValueError("Identity-only audit requires the versioned compatible rule")
        with db:
            roles = choose_roles(db, rule)
        checks = validate_role_members(db, roles, rule)
        for role, ids in roles.items():
            publish_immutable_bytes(output/"members"/(role+".i64le"), _index_bytes(ids))
        # Role manifests are computed from actual member arrays, never ranges
        # pretending random held-out rows are contiguous.
        graph_records = _derive_graph_chunks(fixed, output, source, db, roles, rule_id)
        manifests = {}
        for role, ids in roles.items():
            payload = {"format": FORMAT, "status": "complete", "version_identity": rule_id, "role": role,
                "rows": len(ids), "source_index_sha256": index_digest(ids), "ordering": "ascending-source-index",
                "source_index_file": "members/"+role+".i64le", "source_index_file_sha256": sha256_file(output/"members"/(role+".i64le")),
                "source_manifest_sha256": rule["source_manifest_sha256"], "source_acceptance_sha256": rule["source_acceptance_sha256"],
                "role_inventory_sha256": rule["role_inventory_sha256"], "molecule_rule": rule["molecule_rule"],
                "generator_sha256": rule["generator_sha256"], "source_commit": rule["source_commit"],
                "feature_schema": "ogb-atom9-bond3-rwse16-v1", "pure_2d": True,
                "graph_chunks": graph_records[role], "protected": role == "independent_audit",
                "access": "FROZEN_CANDIDATE_ONE_TIME_AUDIT_ONLY" if role == "independent_audit" else "TRAIN_OR_COMMON_DEVELOPMENT",
                "label_statistics_published": False, "checks": checks}
            payload["logical_content_sha256"] = canonical_fingerprint({"members": payload["source_index_sha256"],
                "chunks": [{k: c[k] for k in ("row_sha256", "target_sha256", "topology_sha256")} for c in graph_records[role]]})
            path = output/"roles"/(role+".json")
            publish_immutable_bytes(path, json_bytes(payload))
            manifests[role] = {"path": "roles/"+role+".json", "sha256": sha256_file(path),
                "rows": len(ids), "source_index_sha256": payload["source_index_sha256"],
                "logical_content_sha256": payload["logical_content_sha256"]}
        report = {"format": FORMAT+"-acceptance", "status": "DATA_ROLE_ACCEPTED", "scope": rule["scope"],
            "rule_identity": rule_id, "roles": manifests, "checks": checks,
            "official_validation_read": False, "test_roles_read": False, "gpu_used": False,
            "graphs_rebuilt": False, "audit_label_statistics_exposed": False,
            "preparation_wall_seconds": time.monotonic()-started,
            "parent_cpu_seconds": time.process_time()-cpu_start, "whole_allocation_cpu_hours": None,
            "mirrors": {"kaggle1": "NOT_PUBLISHED", "kaggle2": "NOT_PUBLISHED"},
            "training_authority": "NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING"}
        if rule["molecule_rule"] in COMPATIBLE_RULES:
            report["molecular_identity_audit_sha256"] = sha256_file(output/"molecular_identity_audit.json")
        from .research_memory.schemas import validate_role_event
        events=[]
        for role in ALL_ROLES:
            event={"schema":"molgap-role-event-v1","role_event_id":"DATA-PCQM-V2-"+role.replace("_","-"),
                "trajectory_id":"DATA-PCQM-UNIFIED-V2-CPU-PREPARATION","action_id":"PREPARE-IDENTITY",
                "run_id":"IMS-UNIFIED-V2-CPU","dataset_identity":rule_id,
                "role_name":role,"row_manifest_hash":manifests[role]["source_index_sha256"],
                "access_kind":"labels_read","selection_used":False,
                "evidence_ref":manifests[role]["path"]}
            validate_role_event(event);events.append(event)
        # Preparation events use the existing event shape, kept with the data
        # evidence rather than invented scientific trajectories in canonical RML.
        publish_immutable_bytes(output/"preparation_role_events.json",json_bytes({"events":events,
            "purpose":"CPU identity/content validation only", "metrics_computed":False,
            "canonical_research_trajectory_published":False}))
        report["preparation_role_events_sha256"]=sha256_file(output/"preparation_role_events.json")
        # A fresh independent content/role validation is mandatory before status
        # publication; synthetic scope never asserts real DATA_ROLE_ACCEPTED.
        accept_unified_roles(output, rule=rule)
        if rule["scope"] == "CPU_SYNTHETIC_ONLY":
            report["status"] = "SYNTHETIC_DATA_ROLES_VERIFIED"
        publish_immutable_bytes(output/"acceptance.json", json_bytes(report))
        return report
    finally:
        if pool:
            pool.close(); pool.join()
        db.close()


def _derive_graph_chunks(fixed, output, source, db, roles, rule_id):
    import torch
    from torch_geometric.data import InMemoryDataset
    from .packed_graph_dataset import PackedGraphDataset
    from .cross_scale_inputs import index_digest, target_digest
    from .cross_scale_checkpoint import state_digest
    records = {r: [] for r in ALL_ROLES}
    member_sets = {r: set(ids) for r, ids in roles.items()}
    # The three small training scales share Full's chunks only when all their
    # rows are present; each role gets an explicit non-contiguous offset map.
    for record in source["store_assets"]["topology"]:
        if record.get("role") != "train":
            continue
        path = _pin(fixed, record)
        start, stop = record["source_idx_min"], record["source_idx_max"]+1
        chunk_id = f"{start:07d}_{stop:07d}"
        receipt = output/"chunks"/(chunk_id+".json")
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            if prior["source_sha256"] != record["sha256"] or prior["rule_identity"] != rule_id:
                raise ValueError("Graph resume identity changed")
            for role, item in prior["roles"].items():
                _pin(output, item); records[role].append(item)
            continue
        shard = PackedGraphDataset(path, mmap=True)
        if len(shard) != stop-start or not set(shard._data.keys()) <= GRAPH_FIELDS | SOURCE_METADATA_FIELDS | {"num_nodes"}:
            raise ValueError("Non-2D/unknown topology field or row count: "+str(sorted(shard._data.keys())))
        source_ids = shard._data.source_idx.reshape(-1)
        if source_ids.dtype != torch.int64 or not torch.equal(source_ids, torch.arange(start, stop)):
            raise ValueError("Source topology membership/order differs")
        raw_targets = dict(db.execute("SELECT idx,target FROM rows WHERE idx>=? AND idx<?", (start, stop)))
        chosen = defaultdict(list)
        for offset in range(len(shard)):
            graph = _project_source_graph(shard.get(offset))
            idx = start+offset
            if set(graph.keys()) - (GRAPH_FIELDS | {"num_nodes"}):
                raise ValueError("Forbidden 3D or unknown graph field")
            if (graph.x.ndim != 2 or graph.x.shape[1] != 9 or graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 3 or
                    graph.random_walk_pe.shape != (graph.num_nodes, 16) or not torch.isfinite(graph.random_walk_pe).all() or
                    graph.y.numel() != 1 or not torch.isfinite(graph.y).all() or
                    not torch.equal(graph.y.reshape(-1), torch.tensor([raw_targets[idx]], dtype=graph.y.dtype))):
                raise ValueError("Topology/feature/label identity differs from accepted source")
            for role in ("full_minus_holdout", "common_dev", "independent_audit"):
                if idx in member_sets[role]:
                    chosen[role].append(graph)
        chunk_roles = {}
        for role, graphs in chosen.items():
            payload = InMemoryDataset.collate(graphs)
            relative = f"payloads/{role}/{chunk_id}.pt"
            dest = output/relative
            # Tensor serialization uses the existing atomic writer, not another
            # checkpoint or graph-builder protocol.
            import io
            buffer = io.BytesIO(); torch.save(payload, buffer)
            if not dest.exists():
                publish_immutable_bytes(dest, buffer.getvalue())
            indices = [int(g.source_idx.item()) for g in graphs]
            top = {k: v for k, v in payload[0].to_dict().items() if k not in {"source_idx", "y"}}
            item = {"file": relative, "sha256": sha256_file(dest), "rows": len(graphs),
                "row_sha256": index_digest(indices), "target_sha256": target_digest([float(g.y.item()) for g in graphs]),
                "topology_sha256": state_digest({"data": top, "slices": {k:v for k,v in (payload[1] or {}).items() if k not in {"source_idx", "y"}}}),
                "parent_sha256": record["sha256"], "source_idx_min": indices[0], "source_idx_max": indices[-1],
                "offsets": None}
            records[role].append(item); chunk_roles[role] = item
            if role == "full_minus_holdout":
                for train in TRAIN_ROLES[:3]:
                    offsets = [j for j, idx in enumerate(indices) if idx in member_sets[train]]
                    if not offsets:
                        continue
                    # Below 1M no new holdout can occur; consequently chunks
                    # containing these nested roles cannot expose audit labels.
                    subset = [graphs[j] for j in offsets]
                    subset_payload = InMemoryDataset.collate(subset)
                    subset_top = {k:v for k,v in subset_payload[0].to_dict().items() if k not in {"source_idx", "y"}}
                    view = {**item, "offsets": offsets, "rows": len(offsets),
                        "row_sha256": index_digest([indices[j] for j in offsets]),
                        "target_sha256": target_digest([float(g.y.item()) for g in subset]),
                        "topology_sha256": state_digest({"data": subset_top, "slices": {k:v for k,v in (subset_payload[1] or {}).items() if k not in {"source_idx", "y"}}})}
                    records[train].append(view); chunk_roles[train] = view
        publish_immutable_bytes(receipt, json_bytes({"rule_identity": rule_id, "source_sha256": record["sha256"], "roles": chunk_roles}))
        atomic_write(output/"progress.json", json_bytes({"stage": "DERIVED_GRAPH_CHUNKS", "last_source_stop": stop,
            "status": "RUNNING", "rule_identity": rule_id, "gpu_used": False}))
    return records


def accept_unified_roles(root, *, rule):
    """Independent CPU acceptance reopens only the new train-derived assets."""
    import torch
    from .packed_graph_dataset import PackedGraphDataset
    from .cross_scale_inputs import index_digest, target_digest
    from .cross_scale_checkpoint import state_digest
    root = Path(root).resolve(); rule_id = validate_rule(rule)
    if json.loads((root/"rule.json").read_text()) != rule:
        raise ValueError("Frozen rule identity changed")
    db = sqlite3.connect("file:"+(root/"private_identity.sqlite").as_posix()+"?mode=ro", uri=True)
    roles = {}
    try:
        if rule["molecule_rule"] in COMPATIBLE_RULES:
            audit = json.loads((root/"molecular_identity_audit.json").read_text())
            if audit != molecular_audit(db, rule) or audit["status"] != "IDENTITIES_VERIFIED":
                raise ValueError("ROLE_BLOCKED: molecular identity audit changed or incomplete")
            if (root/"acceptance.json").exists():
                accepted = json.loads((root/"acceptance.json").read_text())
                if accepted.get("molecular_identity_audit_sha256") != sha256_file(root/"molecular_identity_audit.json"):
                    raise ValueError("ROLE_BLOCKED: accepted molecular identity audit SHA changed")
        for role in ALL_ROLES:
            manifest = json.loads((root/"roles"/(role+".json")).read_text())
            if manifest["version_identity"] != rule_id or manifest["role"] != role:
                raise ValueError("Dataset version/role changed")
            data = (root/manifest["source_index_file"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != manifest["source_index_file_sha256"] or len(data) % 8:
                raise ValueError("Source index file content changed")
            ids = [v[0] for v in struct.iter_unpack("<q", data)]
            if index_digest(ids) != manifest["source_index_sha256"] or len(ids) != manifest["rows"]:
                raise ValueError("Source index identity changed")
            decoded = []
            for item in manifest["graph_chunks"]:
                shard = PackedGraphDataset(_pin(root, item), mmap=True)
                offsets = item["offsets"] if item["offsets"] is not None else range(len(shard))
                graphs = [shard.get(i) for i in offsets]
                got = [int(g.source_idx.item()) for g in graphs]
                if len(got) != item["rows"] or index_digest(got) != item["row_sha256"]:
                    raise ValueError("Graph row identity/order changed")
                if target_digest([float(g.y.item()) for g in graphs]) != item["target_sha256"]:
                    raise ValueError("Label content changed")
                from torch_geometric.data import InMemoryDataset
                packed = InMemoryDataset.collate(graphs)
                top = {k:v for k,v in packed[0].to_dict().items() if k not in {"source_idx", "y"}}
                if set(top) - {"x", "edge_index", "edge_attr", "random_walk_pe", "num_nodes"}:
                    raise ValueError("Forbidden graph input field")
                actual = state_digest({"data": top, "slices": {k:v for k,v in (packed[1] or {}).items() if k not in {"source_idx", "y"}}})
                if actual != item["topology_sha256"]:
                    raise ValueError("Topology content changed")
                decoded.extend(got)
            if decoded != ids:
                raise ValueError("Non-contiguous row map differs from manifest")
            roles[role] = ids
        checks=validate_role_members(db, roles, rule)
        groups=_ranked_molecule_groups(db,rule)
        if (roles["common_dev"]!=sorted(g[0] for g in groups[:rule["common_dev_rows"]]) or
                roles["independent_audit"]!=sorted(g[0] for g in groups[rule["common_dev_rows"]:])):
            raise ValueError("ROLE_BLOCKED: held-out membership differs from frozen hash ranking")
        return checks
    finally:
        db.close()


def authorize_role(manifest, *, operation, candidate_frozen=False, selection_frozen=False):
    """Preparation authority never becomes audit scoring authority implicitly."""
    if type(candidate_frozen) is not bool or type(selection_frozen) is not bool:
        raise TypeError("Role gate flags must be exact bool")
    role = manifest.get("role")
    allowed = {r: {"training_membership", "prediction_input", "labels_read"} for r in TRAIN_ROLES}
    allowed["common_dev"] = {"prediction_input", "labels_read", "metric_computed", "selection_used"}
    # Scoring needs a separate one-time RML transaction; this API deliberately
    # cannot open audit labels, even with both frozen assertions.
    allowed["independent_audit"] = set()
    if (role not in allowed or operation not in allowed[role] or manifest.get("status") != "complete" or
            manifest.get("format") != FORMAT or manifest.get("protected") is not False):
        raise ValueError("ROLE_BLOCKED: operation requires separate owning authority/event")
    return {"role": role, "operation": operation, "training_authority": False}


def verify_mirror_identity(master, mirror):
    fields = ("format", "version_identity", "role", "rows", "source_index_sha256", "ordering",
              "logical_content_sha256", "feature_schema", "pure_2d", "protected")
    if any(master.get(k) is None or master.get(k) != mirror.get(k) for k in fields):
        raise ValueError("Mirror differs from IMS logical data identity")
    if master["role"] == "independent_audit":
        raise ValueError("ROLE_BLOCKED: audit is not a normal platform mirror")
    return {"status": "LOGICAL_IDENTITY_MATCH_ONLY", "byte_acceptance": "NOT_EVALUATED"}
