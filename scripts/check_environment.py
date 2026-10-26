#!/usr/bin/env python3
"""Check the Python environment, CUDA device and bundled extension imports."""

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "docs/tested_environment.json").read_text())
    report = {"python": sys.version.split()[0], "packages": {}, "extensions": {}}
    errors = []
    if sys.version_info[:2] != (3, 9):
        errors.append("The tested runtime uses Python 3.9")
    for name, version in expected["packages"].items():
        try:
            actual = importlib.metadata.version(name)
            report["packages"][name] = actual
            if actual != version:
                errors.append(f"{name}: expected {version}, found {actual}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}")
    for name in (
        "torch",
        "cuvs.neighbors.ivf_flat",
        "simple_knn._C",
        "diff_gaussian_rasterization._C",
        "diff_gaussian_rasterization_texture._C",
    ):
        try:
            module = importlib.import_module(name)
            report["extensions"][name] = "OK"
            if name == "torch":
                report["cuda_available"] = module.cuda.is_available()
                report["torch_cuda"] = module.version.cuda
                if not report["cuda_available"]:
                    errors.append("No CUDA device is available to Torch")
                else:
                    report["gpu"] = module.cuda.get_device_name(0)
        except Exception as exc:
            report["extensions"][name] = repr(exc)
            errors.append(f"Cannot import {name}: {exc}")
    report.update(pass_=not errors, errors=errors)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
