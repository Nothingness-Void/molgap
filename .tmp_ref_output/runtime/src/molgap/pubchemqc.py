"""Reusable PubChemQC streaming, filtering, and molecule identity helpers."""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


HF_REPO = "molssiai-hub/pubchemqc-b3lyp"
HF_CONFIG = "b3lyp_pm6"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}"
HF_TREE = f"{HF_API}/tree/main/data/{{config}}/train"
HF_RESOLVE = (
    f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
    "data/{config}/train/{file}"
)
USER_AGENT = "molgap-archive-r02-router/1.0"
ALLOWED_ELEMENTS = frozenset({"C", "H", "N", "O", "S", "F", "Cl"})


@dataclass(frozen=True)
class PubChemQCFilter:
    allowed_elements: tuple[str, ...] = tuple(sorted(ALLOWED_ELEMENTS))
    min_mw: float = 200.0
    max_mw: float = 1000.0
    state: str = "S0"
    charge: int = 0
    multiplicity: int = 1
    alpha_beta_tolerance_ev: float = 1e-6
    gap_consistency_tolerance_ev: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_dataset_metadata(timeout: int = 60) -> dict:
    req = urllib.request.Request(HF_API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = json.load(response)
    return {
        "repo": HF_REPO,
        "revision": raw.get("sha"),
        "last_modified": raw.get("lastModified"),
    }


def list_hf_files(config: str = HF_CONFIG, timeout: int = 60) -> list[dict]:
    req = urllib.request.Request(
        HF_TREE.format(config=config), headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = json.load(response)
    files = []
    for item in raw:
        if item.get("type") != "file":
            continue
        files.append({
            "name": item["path"].rsplit("/", 1)[-1],
            "size": int(item.get("size") or item.get("lfs", {}).get("size") or 0),
            "oid": item.get("oid") or item.get("lfs", {}).get("oid"),
        })
    return sorted(files, key=lambda row: row["name"])


def read_http_range(
    url: str, start: int, end: int, timeout: int = 120, retries: int = 3
) -> bytes:
    """Read a bounded byte range and reject servers that ignore nonzero ranges."""
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                if start > 0 and status != 206:
                    raise RuntimeError(f"server ignored byte range (HTTP {status})")
                return response.read(end - start + 1)
        except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"range request failed after {retries} attempts: {last_error}")


def iter_json_objects(buffer: bytes, starts_at_zero: bool = False) -> Iterator[dict]:
    """Yield complete objects from a full JSON prefix or an arbitrary byte window."""
    text = buffer.decode("utf-8", errors="ignore")
    depth = 0
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    value = json.loads(text[start:index + 1])
                    if isinstance(value, dict):
                        yield value
                except json.JSONDecodeError:
                    pass
                start = None


def formula_elements(formula: object) -> set[str]:
    if not isinstance(formula, str):
        return set()
    elements = set()
    index = 0
    while index < len(formula):
        if formula[index].isupper():
            symbol = formula[index]
            index += 1
            while index < len(formula) and formula[index].islower():
                symbol += formula[index]
                index += 1
            elements.add(symbol)
        else:
            index += 1
    return elements


def _float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def molecule_identity(smiles: object) -> tuple[str, str] | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = ""
    return canonical, inchikey


def pubchemqc_record(obj: dict, rules: PubChemQCFilter) -> tuple[dict | None, str]:
    """Validate one raw row and return a slim Router candidate plus reject reason."""
    mw = _float(obj.get("pubchem-molecular-weight"))
    if mw is None or not rules.min_mw <= mw <= rules.max_mw:
        return None, "mw"
    formula = obj.get("pubchem-molecular-formula")
    elements = formula_elements(formula)
    if not elements or not elements.issubset(rules.allowed_elements):
        return None, "elements"
    if obj.get("state") != rules.state:
        return None, "state"
    try:
        if int(obj.get("pubchem-charge")) != rules.charge:
            return None, "charge"
        if int(obj.get("multiplicity")) != rules.multiplicity:
            return None, "multiplicity"
    except (TypeError, ValueError):
        return None, "spin_charge_missing"

    alpha = [_float(obj.get(f"energy-alpha-{target}")) for target in ("homo", "lumo", "gap")]
    beta = [_float(obj.get(f"energy-beta-{target}")) for target in ("homo", "lumo", "gap")]
    if any(value is None for value in alpha + beta):
        return None, "labels"
    homo, lumo, gap = alpha
    if gap <= 0 or abs((lumo - homo) - gap) > rules.gap_consistency_tolerance_ev:
        return None, "gap_consistency"
    if max(abs(a - b) for a, b in zip(alpha, beta)) > rules.alpha_beta_tolerance_ev:
        return None, "alpha_beta"

    identity = molecule_identity(obj.get("pubchem-isomeric-smiles"))
    if identity is None:
        return None, "smiles"
    canonical, inchikey = identity
    mol = Chem.MolFromSmiles(canonical)
    atoms = list(mol.GetAtoms())
    cid = obj.get("cid")
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        return None, "cid"
    return {
        "cid": cid,
        "mw": mw,
        "formula": formula,
        "smiles": obj.get("pubchem-isomeric-smiles"),
        "canonical_smiles": canonical,
        "inchikey": inchikey,
        "homo": homo,
        "lumo": lumo,
        "gap": gap,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": Descriptors.MolLogP(mol),
        "formal_charge": sum(atom.GetFormalCharge() for atom in atoms),
        "has_s": int("S" in elements),
        "has_p": int("P" in elements),
        "has_cl": int("Cl" in elements),
        "has_f": int("F" in elements),
    }, "kept"
