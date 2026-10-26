#!/usr/bin/env python3
"""Verify this release is source-only and matches its relative SHA256 manifest."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'.pth','.pt','.ckpt','.pkl','.npy','.npz','.ply','.mp4','.onnx','.so','.pyc'}
EXCLUDE={'.git','.env','.venv','output','build','dist'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()

def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not EXCLUDE.intersection(p.relative_to(ROOT).parts))

def main():
    paths=files();bad=[str(p.relative_to(ROOT)) for p in paths if p.suffix in FORBIDDEN or '__pycache__' in p.parts]
    assert not bad,'Not a clean source release: '+', '.join(bad)
    for name in ['assets','reference_results','verification']:assert not (ROOT/name).exists(),name
    for name in ['submodules/simple-knn/setup.py',
                 'submodules/diff-gaussian-rasterization-radegs/third_party/glm/glm/glm.hpp',
                 'submodules/diff-gaussian-rasterization-texture/third_party/glm/glm/glm.hpp']:
        assert (ROOT/name).is_file(), 'Missing build dependency: '+name
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    actual={str(p.relative_to(ROOT)) for p in paths if p.name!='SOURCE_MANIFEST.json'}
    assert actual==set(manifest['files']), 'Source file list differs from manifest'
    for name,h in manifest['files'].items():assert sha(ROOT/name)==h,name
    print(f"PASS: {len(manifest['files'])} source files; no checkpoints/results/caches")

if __name__=='__main__':main()
