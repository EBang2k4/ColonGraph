#!/usr/bin/env python3
"""Maintainer command after editing the source release."""
import json
from verify_source import ROOT,files,sha
if __name__=='__main__':
    content={str(p.relative_to(ROOT)):sha(p) for p in files() if p.name!='SOURCE_MANIFEST.json' and p.suffix!='.pyc' and '__pycache__' not in p.parts}
    (ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(baseline='M2_nearest16 clean source',files=content),indent=2)+'\n')
    print('Source manifest:',len(content),'files')
