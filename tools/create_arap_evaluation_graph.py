#!/usr/bin/env python3
"""Freeze the K=8 canonical backbone graph shared by K0/A0/K1/A1."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from utils.node_arap import CanonicalNodeARAPGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="c1_sigmoid2_t4_v4")
    parser.add_argument("--output", type=Path,
                        default=Path("output/arap_backbone_eval_k8.pth"))
    args = parser.parse_args()
    source_path = (Path("output/baseline1_nodes2000") / args.scene /
                   "point_cloud/iteration_1/deformation_nodes.pth")
    source = torch.load(source_path, map_location="cpu", weights_only=True)["graph"]
    if len(source["node_xyz"]) != 2000:
        raise ValueError("Evaluation graph source must contain exactly 2,000 backbone nodes")
    graph = CanonicalNodeARAPGraph.build(source["node_xyz"], k=8)
    payload = graph.state_dict_payload()
    payload.update({"scene": args.scene, "source": str(source_path),
                    "source_graph": source})
    if args.output.exists():
        previous = torch.load(args.output, map_location="cpu", weights_only=True)
        tensor_keys = ("canonical_xyz", "neighbor_indices", "neighbor_mask",
                       "edge_weights", "local_scale")
        if previous.get("k") != 8 or any(
                not torch.equal(previous[key], payload[key]) for key in tensor_keys):
            raise RuntimeError("Existing ARAP evaluation graph differs from canonical source")
        print(f"Verified existing evaluation graph: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved fixed K=8 evaluation graph: {args.output}")
    print(f"nodes=2000 max_degree={graph.neighbor_indices.shape[1]}")


if __name__ == "__main__":
    main()
