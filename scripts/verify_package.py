#!/usr/bin/env python3
"""Verify immutable code/assets by relative SHA256; no CUDA or data required."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
q=json.loads((ROOT/'MANIFEST.json').read_text());bad=[]
for rel,h in q['files'].items():
 p=ROOT/rel
 if not p.is_file() or sha(p)!=h:bad.append(rel)
if bad:raise SystemExit('FAILED: '+', '.join(bad))
print('PASS:',len(q['files']),'files;',q['bytes'],'bytes; code/checkpoints/masks/references unchanged')
