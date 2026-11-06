"""NO_TRAIN probe: do RDKit resonance forms change K1's actual OGB inputs and the frozen K1's Gap prediction?

Uses only fixed-100K train shard 0 (no dev/protected role) and the frozen reference checkpoint; no label is read.
Run: .venv\Scripts\python.exe -I experiments/k1_resonance_invariance_review/probe_resonance_inputs.py <shard0.pt> <selected_model.pt> <out.json> [N]
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, torch
from rdkit import Chem, RDLogger
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Batch, Data
from molgap.data.packed_graph import PackedGraphDataset
from molgap.family_model_ops import k1_forward
from molgap.qm9_neural_atom import make_encoder

RDLogger.DisableLog("rdApp.*")
CHARGES = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
BONDS = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def rebuild(g):
    m = Chem.RWMol()
    for row in g.x.tolist():
        a = Chem.Atom(row[0] + 1)
        a.SetFormalCharge(CHARGES[row[3]] if row[3] < len(CHARGES) else 0)
        a.SetNumExplicitHs(row[4]); a.SetNoImplicit(True); a.SetNumRadicalElectrons(row[5])
        m.AddAtom(a)
    seen = set()
    for (i, j), e in zip(g.edge_index.t().tolist(), g.edge_attr.tolist()):
        if (j, i) in seen or i == j: continue
        seen.add((i, j)); m.AddBond(i, j, BONDS[e[0]] if e[0] < 4 else Chem.BondType.SINGLE)
    return m.GetMol()


def featurize(mol, n, edge_pairs):
    """Round-trip through SMILES (what any alternate input string would go through); keep original atom order via map numbers."""
    for a in mol.GetAtoms(): a.SetAtomMapNum(a.GetIdx() + 1)
    mm = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mm is None: return None
    x = np.zeros((n, 9), dtype=np.int64); idx = {}
    for a in mm.GetAtoms():
        k = a.GetAtomMapNum() - 1; idx[k] = a.GetIdx(); x[k] = atom_to_feature_vector(a)
    ea = np.zeros((len(edge_pairs), 3), dtype=np.int64)
    for r, (i, j) in enumerate(edge_pairs):
        b = mm.GetBondBetweenAtoms(idx[i], idx[j])
        if b is None: return None
        ea[r] = bond_to_feature_vector(b)
    return x, ea


def main(shard, ckpt, out, n=3000):
    d = PackedGraphDataset(Path(shard), mmap=True)
    model = make_encoder("neural_atom_k1")
    p = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(p["model"] if "model" in p and "weights" not in p else p["model"]); model.eval()
    stats = dict(rows=0, rebuilt_ok=0, base_x_match=0, base_e_match=0, multi_forms=0, differing_input=0, forms=[], differing_forms=[])
    deltas, charge_change, bond_change, arom_change, per_mol_range = [], 0, 0, 0, []
    def predict(g, x, ea):
        data = Data(x=torch.as_tensor(x), edge_index=g.edge_index, edge_attr=torch.as_tensor(ea), random_walk_pe=g.random_walk_pe)
        b = Batch.from_data_list([data])
        with torch.no_grad(): return float(k1_forward(model, b))
    for i in range(n):
        g = d[i]; stats["rows"] += 1
        try:
            mol = rebuild(g); Chem.SanitizeMol(mol)
        except Exception:
            continue
        stats["rebuilt_ok"] += 1
        pairs = [(a, b) for a, b in g.edge_index.t().tolist()]
        base = featurize(Chem.Mol(mol), g.x.shape[0], pairs)
        if base is None: continue
        stats["base_x_match"] += int((base[0][:, [0, 3, 4, 5, 7]] == g.x.numpy()[:, [0, 3, 4, 5, 7]]).all())
        stats["base_e_match"] += int((base[1][:, [0, 2]] == g.edge_attr.numpy()[:, [0, 2]]).all())
        try:
            sup = Chem.ResonanceMolSupplier(mol, Chem.KEKULE_ALL, 50)
            forms = [sup[k] for k in range(len(sup))] if len(sup) > 0 else []
        except Exception:
            forms = []
        stats["forms"].append(len(forms))
        if len(forms) < 2: continue
        stats["multi_forms"] += 1
        y0 = predict(g, base[0], base[1]); outs, distinct = [y0], 0
        for f in forms:
            alt = featurize(Chem.Mol(f), g.x.shape[0], pairs)
            if alt is None: continue
            if not (np.array_equal(alt[0], base[0]) and np.array_equal(alt[1], base[1])):
                distinct += 1
                charge_change += int((alt[0][:, 3] != base[0][:, 3]).any())
                bond_change += int((alt[1][:, 0] != base[1][:, 0]).any())
                arom_change += int((alt[0][:, 7] != base[0][:, 7]).any())
                outs.append(predict(g, alt[0], alt[1]))
        if distinct:
            stats["differing_input"] += 1; stats["differing_forms"].append(distinct)
            deltas += [abs(o - y0) for o in outs[1:]]; per_mol_range.append(max(outs) - min(outs))
    res = dict(stats)
    res.pop("forms"); res.pop("differing_forms")
    df = np.array(stats["differing_forms"]) if stats["differing_forms"] else np.array([0])
    res.update(frac_multi_forms=stats["multi_forms"] / max(1, stats["rebuilt_ok"]), frac_differing_input=stats["differing_input"] / max(1, stats["rebuilt_ok"]),
               mean_distinct_forms_when_differing=float(df.mean()), formal_charge_pattern_changed=charge_change, bond_type_pattern_changed=bond_change, aromatic_flag_changed=arom_change,
               eV_abs_delta_mean=float(np.mean(deltas)) if deltas else None, eV_abs_delta_median=float(np.median(deltas)) if deltas else None,
               eV_abs_delta_p95=float(np.percentile(deltas, 95)) if deltas else None, eV_range_mean=float(np.mean(per_mol_range)) if per_mol_range else None,
               eV_range_p95=float(np.percentile(per_mol_range, 95)) if per_mol_range else None, n_pred_pairs=len(deltas))
    Path(out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3000)
