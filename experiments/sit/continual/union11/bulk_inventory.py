"""Hash immutable bulk artifacts on the pod before removing local duplicates."""
import hashlib
from pathlib import Path
from .common import ART

def main():
    base=Path(__file__).parent
    paths=list((base/'artifacts').rglob('*'))
    current=base/'artifacts_v2'
    paths += list(current.glob('**/generated/**/*'))
    paths += list(current.glob('**/search_*.png'))
    # Active logs are excluded from sync, but not removed by this hash plan:
    # they change while training and should not be checked against a moving copy.
    selected=sorted({p for p in paths if p.is_file() and p.suffix!='.pt'})
    with (ART/'pod_bulk_sha256.txt').open('w') as f:
        for path in selected:
            f.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(base))+'\n')
    print('POD_BULK_INVENTORY',len(selected),flush=True)

if __name__=='__main__':main()
