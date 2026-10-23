"""Only metadata paths are rewritten; tensor values are untouched."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def resolve(value):
    if not isinstance(value,str):return value
    if value.startswith('@bundle/'):return str(ROOT/value[len('@bundle/'):])
    if value.startswith('@dataset/'):
        path=os.environ.get('COLONGRAPH_DATASET')
        if not path:raise RuntimeError('Set dataset through u7.py --data')
        return str(Path(path).resolve())
    return value

def portable(value):
    if isinstance(value,str) and Path(value).is_absolute():
        try:return '@bundle/'+str(Path(value).relative_to(ROOT))
        except ValueError:pass
    return value
