"""Diagnostic: stream-parse one HF file from the top, find where OOD molecules
appear (after the training-set quota is exhausted)."""
from __future__ import annotations

import re
import urllib.request

import ijson
import numpy as np

from fetch_ood_1000 import (list_files, load_training_cids, HF_BASE, USER_AGENT,
                            ELEMENTS, MW_MIN, MW_MAX)


def formula_elements(formula):
    return set(re.findall(r"[A-Z][a-z]?", formula))


def main():
    train_cids = load_training_cids()
    files = list_files()
    np.random.seed(0)
    np.random.shuffle(files)

    fn = files[0]
    url = HF_BASE.format(file=fn)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"Streaming {fn} from top...\n")

    n_seen = n_match = n_train = n_ood = 0
    first_ood_at = None
    bytes_read = 0

    with urllib.request.urlopen(req, timeout=120) as resp:
        for obj in ijson.items(resp, "item"):
            n_seen += 1
            mw_raw = obj.get("pubchem-molecular-weight")
            formula = obj.get("pubchem-molecular-formula")
            cid = obj.get("cid")
            if mw_raw is None or formula is None or cid is None:
                continue
            mw = float(mw_raw)
            if not (MW_MIN <= mw <= MW_MAX):
                continue
            els = formula_elements(formula)
            if not els or not els.issubset(ELEMENTS):
                continue
            n_match += 1
            if int(cid) in train_cids:
                n_train += 1
            else:
                n_ood += 1
                if first_ood_at is None:
                    first_ood_at = n_match
                    print(f"  First OOD at match #{n_match} (record #{n_seen}), cid={cid}")
            if n_match >= 3000:
                break

    print(f"\n  records scanned: {n_seen}")
    print(f"  MW+element matches: {n_match}")
    print(f"  in_train: {n_train}   OOD: {n_ood}")
    print(f"  first OOD at match #: {first_ood_at}")


if __name__ == "__main__":
    main()
