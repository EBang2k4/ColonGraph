#!/usr/bin/env python3
import argparse
from pathlib import Path

def main(lookat_only=False):
    p=argparse.ArgumentParser(description='Render this run, including dataset/generated lookat cameras.')
    p.add_argument('-s','--source_path',type=Path,required=True);p.add_argument('-m','--model_path',type=Path,required=True)
    p.add_argument('--iteration',type=int,default=-1);p.add_argument('--max_frames',type=int)
    p.add_argument('--skip_train',action='store_true');p.add_argument('--skip_test',action='store_true');p.add_argument('--skip_video',action='store_true')
    p.add_argument('--lookat',action='store_true')
    args=p.parse_args()
    if lookat_only:args.lookat=True;args.skip_train=args.skip_test=args.skip_video=True
    from baseline.rendering import export
    export(args)
if __name__=='__main__':main()
