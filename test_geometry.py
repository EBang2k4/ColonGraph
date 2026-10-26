#!/usr/bin/env python3
"""Same fixed-checkpoint evaluator, with GT geometry explicitly enabled."""
import runpy,sys
if __name__=='__main__':
    if '--geometry' not in sys.argv:sys.argv.append('--geometry')
    runpy.run_module('metrics',run_name='__main__')
