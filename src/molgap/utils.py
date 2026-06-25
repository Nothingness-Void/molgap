"""
Shared utilities for the MolGap pipeline.

The helpers in this file are intentionally small and dependency-light so that
all stage scripts can reuse the same SMILES handling, split handling, and
metric reporting logic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from rdkit import Chem, RDLogger

    RDLogger.logger().setLevel(RDLogger.ERROR)
except Exception:  # pragma: no cover - scripts will fail clearly when RDKit is needed
    Chem = None


from .constants import (
    REPO_ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    TARGET_COLS, METADATA_COLS,
)

DEFAULT_SPLIT_PATH = RESULTS_DIR / "common" / "train_valid_test_split_indices.npz"


def ensure_dirs(*paths: os.PathLike | str) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def require_rdkit() -> None:
    """Raise a clear error if RDKit is unavailable."""
    if Chem is None:
        raise ImportError(
            "RDKit is required for this step. Install it first, e.g. `pip install rdkit`."
        )


def safe_mol(smiles: object):
    """Return an RDKit Mol from a SMILES string, or None on invalid input."""
    require_rdkit()
    if not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonicalize_smiles(smiles: object) -> str | None:
    """Return canonical isomeric SMILES, or None if parsing fails."""
    mol = safe_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def murcko_scaffold_smiles(smiles: object) -> str | None:
    """Return Bemis-Murcko scaffold SMILES for a molecule, or None on failure."""
    require_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = safe_mol(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return "NO_SCAFFOLD"
        return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def calc_morgan_bits(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return Morgan fingerprint bits as a uint8 numpy array."""
    require_rdkit()
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ConvertToNumpyArray

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr


def calc_maccs_keys(mol) -> np.ndarray:
    """Return MACCS keys as a uint8 numpy array (166 bits, index 1-166)."""
    require_rdkit()
    from rdkit.Chem import MACCSkeys
    from rdkit.DataStructs import ConvertToNumpyArray

    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr[1:]


def calc_atompair_bits(mol, n_bits: int = 2048) -> np.ndarray:
    """Return hashed atom pair fingerprint as a uint8 numpy array."""
    require_rdkit()
    from rdkit.Chem import rdMolDescriptors
    from rdkit.DataStructs import ConvertToNumpyArray

    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr


def calc_torsion_bits(mol, n_bits: int = 2048) -> np.ndarray:
    """Return hashed topological torsion fingerprint as a uint8 numpy array."""
    require_rdkit()
    from rdkit.Chem import rdMolDescriptors
    from rdkit.DataStructs import ConvertToNumpyArray

    fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr


def calc_rdkit_descriptors(mol) -> dict[str, float]:
    """Calculate all RDKit 2D descriptors with `desc_` prefixes."""
    require_rdkit()
    from rdkit.Chem import Descriptors

    names = [name for name, _ in Descriptors._descList]
    try:
        desc = Descriptors.CalcMolDescriptors(mol)
    except Exception:
        desc = {}
        for name, func in Descriptors._descList:
            try:
                desc[name] = func(mol)
            except Exception:
                desc[name] = np.nan

    result = {}
    for name in names:
        value = desc.get(name, np.nan)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        if np.isinf(value):
            value = np.nan
        result[f"desc_{name}"] = value
    return result


def calc_gasteiger_descriptors(mol) -> dict[str, float]:
    """Compute aggregate Gasteiger charge descriptors for a molecule."""
    require_rdkit()
    from rdkit.Chem import AllChem

    prefix = "gasteiger_"
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return {f"{prefix}{k}": np.nan for k in [
            "mean", "std", "min", "max", "range",
            "abs_mean", "abs_max", "pos_count", "neg_count",
            "most_pos", "most_neg",
        ]}

    charges = []
    for atom in mol.GetAtoms():
        c = atom.GetDoubleProp('_GasteigerCharge')
        if c != c or abs(c) > 1e6:
            c = 0.0
        charges.append(c)

    charges = np.array(charges)
    pos = charges[charges > 0]
    neg = charges[charges < 0]

    return {
        f"{prefix}mean": float(charges.mean()),
        f"{prefix}std": float(charges.std()),
        f"{prefix}min": float(charges.min()),
        f"{prefix}max": float(charges.max()),
        f"{prefix}range": float(charges.max() - charges.min()),
        f"{prefix}abs_mean": float(np.abs(charges).mean()),
        f"{prefix}abs_max": float(np.abs(charges).max()),
        f"{prefix}pos_count": float(len(pos)),
        f"{prefix}neg_count": float(len(neg)),
        f"{prefix}most_pos": float(pos.max()) if len(pos) > 0 else 0.0,
        f"{prefix}most_neg": float(neg.min()) if len(neg) > 0 else 0.0,
    }


FUSION_CONTEXT_FEATURES = [
    "mw", "heavy_atoms", "fragments", "hetero_atoms", "rotatable_bonds",
    "ring_count", "aromatic_rings", "conjugated_bonds", "frac_csp3", "tpsa",
    "formal_charge", "has_cl", "has_f", "has_s", "has_salt", "is_charged",
]


def calc_fusion_context_features(smiles: object) -> np.ndarray:
    """Return fixed lightweight context features for descriptor-aware fusion."""
    require_rdkit()
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = safe_mol(smiles)
    if mol is None:
        return np.zeros(len(FUSION_CONTEXT_FEATURES), dtype=np.float32)
    try:
        Chem.RemoveStereochemistry(mol)
    except Exception:
        pass

    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    elements = {atom.GetSymbol() for atom in atoms}
    fragments = len(Chem.GetMolFrags(mol))
    formal_charge = sum(atom.GetFormalCharge() for atom in atoms)
    values = {
        "mw": float(Descriptors.MolWt(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "fragments": float(fragments),
        "hetero_atoms": float(sum(1 for atom in atoms if atom.GetSymbol() not in {"C", "H"})),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "conjugated_bonds": float(sum(1 for bond in bonds if bond.GetIsConjugated())),
        "frac_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "formal_charge": float(formal_charge),
        "has_cl": float("Cl" in elements),
        "has_f": float("F" in elements),
        "has_s": float("S" in elements),
        "has_salt": float(fragments > 1),
        "is_charged": float(formal_charge != 0),
    }
    arr = np.array([values[name] for name in FUSION_CONTEXT_FEATURES], dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def build_feature_row_from_smiles(
    smiles: object,
    radius: int = 2,
    n_bits: int = 2048,
) -> dict[str, float] | None:
    """Build fingerprint + RDKit descriptor feature row from a SMILES string."""
    mol = safe_mol(smiles)
    if mol is None:
        return None
    row: dict[str, float] = {}

    bits = calc_morgan_bits(mol, radius=radius, n_bits=n_bits)
    for bit_idx, bit_value in enumerate(bits):
        row[f"morgan_{bit_idx}"] = int(bit_value)

    maccs = calc_maccs_keys(mol)
    for bit_idx, bit_value in enumerate(maccs):
        row[f"maccs_{bit_idx}"] = int(bit_value)

    ap = calc_atompair_bits(mol, n_bits=n_bits)
    for bit_idx, bit_value in enumerate(ap):
        row[f"atompair_{bit_idx}"] = int(bit_value)

    tt = calc_torsion_bits(mol, n_bits=n_bits)
    for bit_idx, bit_value in enumerate(tt):
        row[f"torsion_{bit_idx}"] = int(bit_value)

    row.update(calc_rdkit_descriptors(mol))
    row.update(calc_gasteiger_descriptors(mol))
    return row


def _worker_init(radius: int, n_bits: int) -> None:
    """Cache parameters in worker process globals."""
    global _WORKER_RADIUS, _WORKER_NBITS
    _WORKER_RADIUS = radius
    _WORKER_NBITS = n_bits


def _worker_func(smiles: str) -> dict[str, float] | None:
    """Picklable worker for multiprocessing feature generation."""
    return build_feature_row_from_smiles(smiles, radius=_WORKER_RADIUS, n_bits=_WORKER_NBITS)


def build_feature_rows_parallel(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    n_jobs: int | None = None,
) -> list[tuple[int, dict[str, float]]]:
    """Build feature rows in parallel. Returns list of (original_index, row_dict)."""
    import multiprocessing as mp

    if n_jobs is None:
        n_jobs = min(32, max(1, mp.cpu_count() - 1))

    results: list[tuple[int, dict[str, float]]] = []
    with mp.Pool(n_jobs, initializer=_worker_init, initargs=(radius, n_bits)) as pool:
        from tqdm import tqdm
        for idx, row in enumerate(tqdm(
            pool.imap(_worker_func, smiles_list, chunksize=256),
            total=len(smiles_list), desc="Features", unit="mol",
        )):
            if row is not None:
                results.append((idx, row))
    return results


def save_json(data: dict, path: os.PathLike | str) -> None:
    """Write a dictionary as pretty UTF-8 JSON."""
    path = Path(path)
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_saved_split_indices(
    n_samples: int,
    split_path: os.PathLike | str = DEFAULT_SPLIT_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load saved train/valid/test indices if they match the current dataset."""
    split_path = Path(split_path)
    if not split_path.exists():
        return None
    try:
        with np.load(split_path, allow_pickle=False) as d:
            train_idx = d["train_idx"].astype(int)
            valid_idx = d["valid_idx"].astype(int)
            test_idx = d["test_idx"].astype(int)
            saved_n = int(d["n_samples"][0]) if "n_samples" in d else None
    except Exception:
        return None

    if saved_n is not None and saved_n != n_samples:
        return None

    all_idx = np.concatenate([train_idx, valid_idx, test_idx])
    if len(all_idx) != len(np.unique(all_idx)):
        return None
    if np.any(all_idx < 0) or np.any(all_idx >= n_samples):
        return None
    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        return None
    return train_idx, valid_idx, test_idx


def create_split_indices(
    n_samples: int,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible train/valid/test indices."""
    if n_samples < 10:
        raise ValueError("Need at least 10 samples for a train/valid/test split.")
    all_idx = np.arange(n_samples)
    train_valid_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_state
    )
    valid_fraction_of_train_valid = valid_size / (1.0 - test_size)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_fraction_of_train_valid,
        random_state=random_state,
    )
    return train_idx, valid_idx, test_idx


def load_or_create_split_indices(
    n_samples: int,
    split_path: os.PathLike | str = DEFAULT_SPLIT_PATH,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load a matching split if available; otherwise create and save one."""
    split_path = Path(split_path)
    loaded = load_saved_split_indices(n_samples, split_path)
    if loaded is not None:
        train_idx, valid_idx, test_idx = loaded
        return train_idx, valid_idx, test_idx, f"loaded existing split: {split_path}"

    train_idx, valid_idx, test_idx = create_split_indices(
        n_samples=n_samples,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )
    ensure_dirs(split_path.parent)
    np.savez(
        split_path,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        n_samples=np.array([n_samples], dtype=int),
        valid_size=np.array([valid_size], dtype=float),
        test_size=np.array([test_size], dtype=float),
        random_state=np.array([random_state], dtype=int),
    )
    return train_idx, valid_idx, test_idx, f"created new split: {split_path}"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: Iterable[str] = TARGET_COLS) -> dict:
    """Return MAE/RMSE/R2 for each target and average MAE/RMSE/R2."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict[str, dict[str, float]] = {}

    maes = []
    rmses = []
    r2s = []
    for i, target in enumerate(targets):
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
        metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2}
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    metrics["average"] = {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)),
    }
    return metrics


def load_model_bundle(path: os.PathLike | str) -> dict:
    """Load a joblib model bundle saved by 04_train_baseline.py."""
    return joblib.load(path)


def load_split_indices_or_raise(
    n_samples: int,
    split_path: os.PathLike | str = DEFAULT_SPLIT_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved split indices and raise a clear error if unavailable/invalid."""
    split = load_saved_split_indices(n_samples, split_path)
    if split is None:
        raise FileNotFoundError(
            f"No valid split indices found at {split_path}. "
            "Run scripts/pipeline/04_train_baseline.py first."
        )
    return split


def get_feature_target_arrays(df, feature_cols: list[str] | None = None):
    """Return X, y, and feature columns from a feature DataFrame."""
    if feature_cols is None:
        required = set(METADATA_COLS + TARGET_COLS)
        feature_cols = [c for c in df.columns if c not in required]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)
    return X, y, feature_cols


def flatten_metrics(model_name: str, split_name: str, metrics: dict) -> dict:
    """Flatten nested metrics into one CSV-friendly row."""
    row = {"model": model_name, "split": split_name}
    for target, values in metrics.items():
        for metric_name, value in values.items():
            row[f"{target}_{metric_name}"] = value
    return row


# ── PM6 geometry + Gasteiger charges for inference ────────────


def generate_pm6_coords_mopac(smiles: str):
    """Run PM6 geometry optimization via mopactools. Returns (atomic_numbers, coords) or None."""
    require_rdkit()
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol_h = AllChem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
        return None

    n = mol_h.GetNumAtoms()
    conf = mol_h.GetConformer()
    atomic_nums = [mol_h.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)]
    coords = []
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords.extend([p.x, p.y, p.z])

    try:
        from mopactools.api import MopacSystem, MopacState, from_data

        system = MopacSystem()
        system.natom = n
        system.natom_move = n
        system.model = "PM6"
        system.charge = 0
        system.spin = 0
        system.atom = [mol_h.GetAtomWithIdx(i).GetSymbol() for i in range(n)]
        system.coord = np.array(coords, dtype=np.float64)

        state = MopacState()
        from_data(system, state, relax=True)

        opt_coords = list(system.coord[:n * 3])
        return atomic_nums, opt_coords
    except Exception:
        return None


def compute_gasteiger_charges(mol) -> list[float]:
    """Compute Gasteiger partial charges for an RDKit mol."""
    require_rdkit()
    from rdkit.Chem import AllChem
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        c = atom.GetDoubleProp('_GasteigerCharge')
        charges.append(0.0 if (c != c) or abs(c) > 1e6 else c)  # NaN/inf guard
    return charges
