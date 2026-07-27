"""Test fetching MW>500 molecules from the full PubChemQC b3lyp_pm6 dataset."""
import urllib.request, io, json, re
import ijson

url = ("https://huggingface.co/datasets/molssiai-hub/pubchemqc-b3lyp/"
       "resolve/main/data/b3lyp_pm6/train/000000001-000253696.json")
headers = {"User-Agent": "curl/8", "Range": "bytes=0-20000000"}
req = urllib.request.Request(url, headers=headers)
print("Downloading 20MB from first file of full dataset...")
with urllib.request.urlopen(req, timeout=120) as resp:
    buf = resp.read()
print(f"Downloaded {len(buf)} bytes")

ELEMENTS = {"C", "H", "O", "N", "S", "F", "Cl"}

def formula_elements(f):
    return set(re.findall(r"[A-Z][a-z]?", str(f)))

count = 0
big_mw = 0
examples = []
try:
 for obj in ijson.items(io.BytesIO(buf), "item"):
    count += 1
    mw = obj.get("pubchem-molecular-weight")
    formula = obj.get("pubchem-molecular-formula", "")
    if mw and float(mw) > 500:
        elems = formula_elements(formula)
        if elems and elems.issubset(ELEMENTS):
            big_mw += 1
            if big_mw <= 5:
                smi = obj.get("pubchem-isomeric-smiles", "")
                examples.append(f"  MW={float(mw):.1f} formula={formula} cid={obj.get('cid')} smi={smi[:60]}")

except Exception:
 pass
print(f"\nScanned {count} molecules from first 20MB")
print(f"Found {big_mw} with MW>500 + CHONSFCl only")
if examples:
    print("\nExamples:")
    for e in examples:
        print(e)

# Estimate: if 20MB has X big molecules, full file (5GB) has ~250x more
ratio = 5000 / 20  # 5GB / 20MB
est_per_file = int(big_mw * ratio)
print(f"\nEstimate per file (~5GB): ~{est_per_file} big MW molecules")
print(f"Estimate across 430 files: ~{est_per_file * 430} total")
