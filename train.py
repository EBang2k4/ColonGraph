#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Train M2_nearest16 from scratch: RGB/depth coarse100 + fine30k. No input checkpoints.')
    p.add_argument('-s','--source_path',type=Path,required=True)
    p.add_argument('-m','--model_path',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=30000)
    p.add_argument('--save_iterations',type=int,nargs='*',default=[10000,20000,30000])
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--smoke',action='store_true',help='Test all stage transitions with100 coarse +12 fine updates; NOT a benchmark run')
    args=p.parse_args()
    assert args.smoke or args.iterations==30000,'Final baseline uses30k fine updates'
    from baseline.training import train
    train(args)
if __name__=='__main__':main()
