"""Bounded identity/role counterexamples; no real corpus or training."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest
from rdkit import Chem

from molgap.pcqm_molecular_identity import (
    COMPAT_MOLECULE_RULE, load_fallback_keys, molecular_identity,
)
from molgap.pcqm_unified_roles import (
    canonical_molecule, inventory_rows, choose_roles, molecular_audit,
    prepare_unified_roles, accept_unified_roles, validate_rule,
)
from molgap.pcqm_fixed_datasets import sha256_file
from molgap import pcqm_molecular_identity as parser_owner
from molgap.research_memory.trace import json_bytes
from test_pcqm_unified_roles import rule, source_fixture


LEDGER = Path(__file__).resolve().parents[1]/"experiments/pcqm_k1_scale500k/parser_fallback_audit_v4.json"
ENTRIES = json.loads(LEDGER.read_text())["entries"]
KEYS = load_fallback_keys(LEDGER, sha256_file(LEDGER))


def compatible_rule(r=None):
    result = deepcopy(r or rule())
    result.update(molecule_rule=COMPAT_MOLECULE_RULE,
                  fallback_ledger_sha256=sha256_file(LEDGER),
                  molecule_parser_sha256=sha256_file(Path(parser_owner.__file__)))
    return result


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e:str(e["source_idx"]))
def test_accepted_exception_round_trip_and_all_atom_orders(entry):
    smiles = entry["smiles"]
    with pytest.raises(ValueError):
        canonical_molecule(smiles)  # historical strict v1 stays strict
    key, evidence = molecular_identity(smiles, fallback_keys=KEYS)
    assert key and evidence["status"] == "KNOWN_FALLBACK"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    for offset in range(mol.GetNumAtoms()):
        order = list(range(offset, mol.GetNumAtoms()))+list(range(offset))
        other = Chem.RenumberAtoms(mol, order)
        for atom in other.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx()+1)
        alias = Chem.MolToSmiles(other, canonical=False, isomericSmiles=True)
        assert molecular_identity(alias, fallback_keys=KEYS)[0] == key


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "C[Si](C)(C)C", "[13CH3]C",
                                    "[CH3:9][CH2:1]O", "F[C@](Cl)(Br)I", "C[N+](C)(C)C", "C/C=C/C"])
def test_normal_identity_is_exactly_unchanged(smiles):
    assert molecular_identity(smiles, fallback_keys=KEYS) == (canonical_molecule(smiles), None)


def test_stereo_isotope_charge_distinctions_are_retained():
    for a,b in [("F[C@](Cl)(Br)I","F[C@@](Cl)(Br)I"), ("[13CH3]C","CC"),
                ("[NH4+]","N"), ("C/C=C/C","C/C=C\\C")]:
        assert molecular_identity(a,fallback_keys=KEYS)[0] != molecular_identity(b,fallback_keys=KEYS)[0]


@pytest.mark.parametrize("smiles", ["C[Si](C)(C)(C)C", "CN(C)(C)C", "not-a-molecule", "c1cccc1"])
def test_unknown_si_other_element_and_syntax_never_acquire_identity(smiles):
    key,evidence=molecular_identity(smiles,fallback_keys=KEYS)
    assert key is None and evidence["status"] == "BLOCKED"
    assert evidence["smiles_sha256"] and evidence["reason"]


def test_frozen_rule_and_ledger_cannot_be_silently_substituted(tmp_path):
    before=rule();after=compatible_rule()
    assert validate_rule(before) != validate_rule(after)
    with pytest.raises(ValueError):
        load_fallback_keys(LEDGER,"0"*64)
    ledger=json.loads(LEDGER.read_text());ledger["entries"].pop()
    path=tmp_path/"ledger.json";path.write_bytes(json_bytes(ledger))
    with pytest.raises(ValueError):
        load_fallback_keys(path,sha256_file(path))


def test_unknown_anomaly_full_inventory_is_durable_but_selection_is_blocked():
    r=compatible_rule();db=sqlite3.connect(":memory:")
    rows=[(i,"C"*(i+1),float(i)/10) for i in range(30)]
    rows[0]=(0,ENTRIES[0]["smiles"],0.0)
    rows[10]=(10,"C[Si](C)(C)(C)C",1.0)
    inventory_rows(db,rows,r,fallback_keys=KEYS)
    audit=molecular_audit(db,r)
    assert audit["completed_rows"]==30 and audit["valid_identity_rows"]==29
    assert audit["known_fallback_rows"]==1 and audit["blocked_rows"]==1
    assert [e["source_idx"] for e in audit["entries"]]==[0,10]
    with pytest.raises(ValueError,match="unresolved molecular"):
        choose_roles(db,r)


def test_known_exception_alias_of_training_member_cannot_enter_holdout():
    r=compatible_rule();db=sqlite3.connect(":memory:")
    rows=[(i,"C"*(i+1),float(i)/10) for i in range(30)]
    key=molecular_identity(ENTRIES[0]["smiles"],fallback_keys=KEYS)[0]
    rows[0]=(0,ENTRIES[0]["smiles"],0.0);rows[29]=(29,key,2.9)
    inventory_rows(db,rows,r,fallback_keys=KEYS)
    roles=choose_roles(db,r)
    assert 29 not in roles["common_dev"]+roles["independent_audit"]
    assert roles["train_1m"]==list(range(6))


def fixture_ledger(tmp_path):
    path=tmp_path/"fallback.json";path.write_bytes(LEDGER.read_bytes())
    return path


def test_compatible_full_prepare_identity_only_resume_and_audit_tamper(tmp_path):
    fixed,r=source_fixture(tmp_path);r=compatible_rule(r)
    ledger=fixture_ledger(tmp_path);out=tmp_path/"compatible"
    kwargs=dict(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r,fallback_ledger=ledger)
    audit=prepare_unified_roles(**kwargs,identity_only=True)
    assert audit["status"]=="IDENTITIES_VERIFIED"
    assert not (out/"members").exists() and not (out/"acceptance.json").exists()
    report=prepare_unified_roles(**kwargs)
    assert report["status"]=="SYNTHETIC_DATA_ROLES_VERIFIED"
    assert prepare_unified_roles(**kwargs)==report
    assert accept_unified_roles(out,rule=r)["nested"]
    path=out/"molecular_identity_audit.json"
    tampered=json.loads(path.read_text());tampered["known_fallback_rows"]=99
    path.write_bytes(json_bytes(tampered))
    with pytest.raises(ValueError,match="molecular identity audit"):
        accept_unified_roles(out,rule=r)


def test_parser_byte_pin_and_legacy_output_cannot_be_reused(tmp_path):
    fixed,r=source_fixture(tmp_path);out=tmp_path/"out"
    prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r)
    compat=compatible_rule(r);ledger=fixture_ledger(tmp_path)
    with pytest.raises((ValueError,FileExistsError)):
        prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=compat,fallback_ledger=ledger)
    compat["molecule_parser_sha256"]="0"*64
    with pytest.raises(ValueError,match="parser bytes"):
        prepare_unified_roles(boundary=tmp_path,fixed_root=fixed,output_root=tmp_path/"other",rule=compat,fallback_ledger=ledger)


def test_multiprocess_identity_inventory_has_the_same_result(tmp_path):
    fixed,r=source_fixture(tmp_path);r=compatible_rule(r);ledger=fixture_ledger(tmp_path)
    kwargs=dict(boundary=tmp_path,fixed_root=fixed,rule=r,fallback_ledger=ledger,identity_only=True)
    one=prepare_unified_roles(**kwargs,output_root=tmp_path/"one",workers=1)
    two=prepare_unified_roles(**kwargs,output_root=tmp_path/"two",workers=2)
    assert one==two and two["status"]=="IDENTITIES_VERIFIED"


def test_interrupted_identity_transaction_never_acknowledges_half_a_chunk(tmp_path,monkeypatch):
    import molgap.pcqm_unified_roles as owner
    fixed,r=source_fixture(tmp_path);r=compatible_rule(r);ledger=fixture_ledger(tmp_path)
    out=tmp_path/"interrupted"
    kwargs=dict(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r,fallback_ledger=ledger,identity_only=True)
    original=owner.inventory_rows
    def crash(db,rows,rule,**kw):
        original(db,rows[:5],rule,**kw)
        raise OSError("simulated mid-identity transaction interruption")
    with monkeypatch.context() as patch:
        patch.setattr(owner,"inventory_rows",crash)
        with pytest.raises(OSError,match="mid-identity"):
            prepare_unified_roles(**kwargs)
    db=sqlite3.connect(out/"private_identity.sqlite")
    assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]==0
    # SQLite also rolls back the first table creation in the failed transaction.
    if db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='rows'").fetchone()[0]:
        assert db.execute("SELECT COUNT(*) FROM rows").fetchone()[0]==0
    db.close()
    assert prepare_unified_roles(**kwargs)["completed_rows"]==30


def test_unknown_real_fixture_collects_all_exceptions_without_materializing_roles(tmp_path):
    import gzip
    fixed,r=source_fixture(tmp_path)
    lines=["idx,smiles,homolumogap"]+[f"{i},{'C'*(i+1)},{float(i)/10}" for i in range(30)]
    lines[1]="0,C[Si](C)(C)(C)C,0.0";lines[11]="10,CN(C)(C)C,1.0"
    with gzip.open(fixed/"rows.csv.gz","wt") as stream:
        stream.write("\n".join(lines)+"\n30,PROTECTED,not-a-label\n")
    manifest=json.loads((fixed/"manifest.json").read_text())
    manifest["store_assets"]["source"][0]["sha256"]=sha256_file(fixed/"rows.csv.gz")
    (fixed/"manifest.json").write_bytes(json_bytes(manifest))
    r["source_manifest_sha256"]=sha256_file(fixed/"manifest.json");r=compatible_rule(r)
    out=tmp_path/"unknown";ledger=fixture_ledger(tmp_path)
    kwargs=dict(boundary=tmp_path,fixed_root=fixed,output_root=out,rule=r,fallback_ledger=ledger)
    for _ in range(2):
        with pytest.raises(ValueError,match="unresolved molecular"):
            prepare_unified_roles(**kwargs)
    audit=json.loads((out/"molecular_identity_audit.json").read_text())
    assert audit["completed_rows"]==30 and audit["blocked_rows"]==2
    assert not (out/"members").exists() and not (out/"acceptance.json").exists()
