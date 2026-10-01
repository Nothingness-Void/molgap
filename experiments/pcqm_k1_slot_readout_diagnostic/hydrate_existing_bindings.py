"""Link pre-existing validated artifacts needed by repository-wide RML checks."""
from pathlib import Path
import re
import os
from molgap.research_memory.validate import validate_repository_records

ROOT=Path(__file__).resolve().parents[2]
ORIGINS=[Path('D:/文档/molgap'),Path('D:/w/k1-width256')]

def main():
    linked=[]
    for _ in range(40):
        try:
            validate_repository_records(ROOT)
            print({'status':'EXISTING_BINDINGS_VALIDATED','links':linked})
            return
        except ValueError as error:
            match=re.search(r'required repository pointer is missing: (.+)',str(error))
            if not match:raise
            pointer=match[1]
            destination=(ROOT/pointer).resolve(); destination.relative_to(ROOT)
            origin=next((root/pointer for root in ORIGINS if (root/pointer).is_file()),None)
            if origin is None:raise
            # Read-only acceptance inputs are shared; no source or scientific edits.
            destination.parent.mkdir(parents=True,exist_ok=True)
            os.link(origin,destination)
            linked.append(pointer)
    raise RuntimeError('Existing artifact hydration exceeded bounded limit')

if __name__=='__main__':main()
