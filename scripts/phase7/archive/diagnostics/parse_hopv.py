"""Quick parse of HOPV15 dataset to check usable entries."""
import json

records = []
with open("data/commercial/HOPV_15_revised_2.data", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if (line and not line.startswith("InChI") and "," not in line
            and not line[0].isdigit() and not line.startswith("Conformer")):
        smiles = line
        if i+2 < len(lines):
            parts = lines[i+2].strip().split(",")
            if len(parts) == 13:
                records.append({
                    "smiles": smiles,
                    "type": parts[2],
                    "homo_exp": parts[5],
                    "lumo_exp": parts[6],
                    "gap_exp": parts[7],
                    "optical_gap": parts[8],
                })
    i += 1

print(f"Total records: {len(records)}")

valid = [r for r in records if r["type"] == "molecule"
         and r["homo_exp"] != "nan" and r["lumo_exp"] != "nan"
         and r["gap_exp"] != "nan"]
print(f"Small molecules with HOMO+LUMO+Gap: {len(valid)}")

for r in valid[:5]:
    print(f"  {r['smiles'][:60]}  H={r['homo_exp']}  L={r['lumo_exp']}  G={r['gap_exp']}")

# Also check: how many with just HOMO+LUMO (gap can be computed)
valid2 = [r for r in records if r["type"] == "molecule"
          and r["homo_exp"] != "nan" and r["lumo_exp"] != "nan"]
print(f"Small molecules with HOMO+LUMO (gap computable): {len(valid2)}")
