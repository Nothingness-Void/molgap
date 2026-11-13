"""Local 2D-only featurization for the direction screen (rows 0..149,999 of PCQM4Mv2 raw data.csv.gz).

Outputs artifacts/features_v1.npz + feature_list.json. Inputs for models are
ecfp_packed/desc/aux16/tags; target_eV and source_idx are for alignment/analysis only.
Standardization statistics use training rows 0..99,999 only.
"""
import argparse, csv, gzip, json, math, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

CSV = "D:/molgap-data/ogb_pcqm4mv2_raw/extracted/pcqm4m-v2/raw/data.csv.gz"
N_ROWS, N_TRAIN = 150_000, 100_000
AUX_NAMES = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
             "RingCount", "NumAromaticRings", "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms",
             "BertzCT", "LabuteASA", "qed", "NumValenceElectrons", "BalabanJ"]
TAG_NAMES = ["radical", "rings", "arom_rings", "heavy_atoms", "hetero_atoms", "charged"]
_state = {}


def _init():
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    _state["descs"] = list(Descriptors._descList)
    _state["gen"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _one(smi):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rd
    descs = _state["descs"]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = np.packbits(_state["gen"].GetFingerprintAsNumPy(m).astype(np.uint8))
    d = np.zeros(len(descs), np.float64)
    for i, (_, f) in enumerate(descs):
        try:
            v = float(f(m))
        except Exception:
            v = 0.0
        d[i] = v if math.isfinite(v) else 0.0
    atoms = m.GetAtoms()
    tags = [int(sum(a.GetNumRadicalElectrons() > 0 for a in atoms) > 0),
            rd.CalcNumRings(m), rd.CalcNumAromaticRings(m), m.GetNumHeavyAtoms(),
            rd.CalcNumHeteroatoms(m), sum(a.GetFormalCharge() != 0 for a in atoms)]
    return fp, d, tags


def _chunk(smis):
    return [_one(s) for s in smis]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--out", default=str(Path(__file__).parent / "artifacts"))
    a = ap.parse_args()
    t0 = time.time()
    idx, smi, tgt = [], [], []
    with gzip.open(a.csv, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i = int(row["idx"])
            if i >= N_ROWS:
                break
            idx.append(i); smi.append(row["smiles"]); tgt.append(float(row["homolumogap"]))
    assert idx == list(range(N_ROWS)), "csv rows are not 0..N-1 in order"
    _init()
    names = [n for n, _ in _state["descs"]]
    D = len(names)
    chunks = [smi[i:i + 500] for i in range(0, N_ROWS, 500)]
    res = []
    with Pool(a.workers, initializer=_init) as p:
        for k, r in enumerate(p.imap(_chunk, chunks)):
            res.extend(r)
            if k % 50 == 0:
                print(f"{len(res)}/{N_ROWS} {time.time()-t0:.0f}s", flush=True)
    ecfp = np.zeros((N_ROWS, 256), np.uint8)
    desc = np.zeros((N_ROWS, D), np.float64)
    tags = np.zeros((N_ROWS, 6), np.int16)
    bad = []
    for i, r in enumerate(res):
        if r is None:
            bad.append(i); continue
        ecfp[i], desc[i], tags[i] = r[0], r[1], r[2]
    ok = np.ones(N_ROWS, bool); ok[bad] = False
    tr = ok[:N_TRAIN]
    mu = desc[:N_TRAIN][tr].mean(0)
    sd = desc[:N_TRAIN][tr].std(0)
    sd[sd < 1e-8] = 1.0
    z = np.clip((desc - mu) / sd, -10, 10)  # clip heavy tails (Ipc, etc.) at 10 sigma
    z[~ok] = 0.0
    aux_cols = [names.index(n) for n in AUX_NAMES]
    desc32 = z.astype(np.float32)
    aux16 = desc32[:, aux_cols].copy()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features_v1.npz", source_idx=np.arange(N_ROWS, dtype=np.int64),
                        ecfp_packed=ecfp, desc=desc32, aux16=aux16, tags=tags,
                        target_eV=np.asarray(tgt, np.float32))
    (out / "feature_list.json").write_text(json.dumps({
        "rows": N_ROWS, "std_stat_rows": f"0..{N_TRAIN-1} (training rows only, parse-ok rows)",
        "smiles_parse_failures": len(bad), "parse_failure_idx": bad[:100],
        "ecfp": "Morgan radius 2 (ECFP4), 2048 bits, np.packbits big-endian -> u8[N,256]",
        "desc_names": names, "desc_mean": mu.tolist(), "desc_std": sd.tolist(),
        "desc_note": "NaN/Inf and exceptions -> 0 before standardization; standardized values clipped to +-10; failed rows all zero",
        "aux16_names": AUX_NAMES, "aux16_desc_columns": aux_cols,
        "tags_columns": TAG_NAMES,
        "tags_note": "radical = 1 if any atom has NumRadicalElectrons>0 else 0 (binary); rings=RingCount; arom_rings=NumAromaticRings; heavy_atoms; hetero_atoms=NumHeteroatoms (non C/H); charged=count of atoms with nonzero formal charge",
        "target_eV": "homolumogap; alignment/analysis only, never a model input",
    }, indent=1))
    print(f"done {time.time()-t0:.0f}s fails={len(bad)} D={D}")


if __name__ == "__main__":
    main()
