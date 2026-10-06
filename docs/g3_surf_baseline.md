# G3-Surf stable code baseline

Branch: `baseline/g3-surf-stable` in `EBang2k4/ColonGraph`.
This is the **fixed G3-Surf joint-training baseline**, not an adaptive,
r16, Rank4, CameraLight or appearance-only refit checkpoint.

## Architecture and frozen experiment choices

- B1-bbox: 2,000 Standard-FPS canonical nodes, shared node MLP width128/depth3,
  bbox-normalized input, direct translation blending, `use_do=False`.
- Gaussian geometry K4: fixed inverse-distance weights, minimally repaired
  for the surface graph. Only19/364500 Gaussian rows were repaired.
- Node connectivity: surface/geodesic-proxy KNN6, symmetrized union (degree
  can exceed6), frozen spatial RBF and trajectory gate from the common2k state.
- Node motion: base MLP + G3 zero-initialized residual decoder, hidden64,
  initialization seed4242. No G4 motion guard, ARAP, isometry or edge loss.
- Appearance: R6 node8D + Gaussian embedding16D + continuous time PE13D;
  `37 -> 64 -> 64 -> 3`, `delta=0.5*tanh`, `color=base*(1+delta)`.
  Appearance uses its separate original fixed2000-node K4; no r16 residual.
- Joint geometry/deformation/appearance optimization; depth weight.25,
  original saved Gaussian KNN99 weight.01; `lambda_res=0`, DSSIM/LPIPS loss0,
  color smoothness.0001 and image TV.001, same optimizer/LR schedule.
- Adaptive nodes, Weighted FPS, Gaussian densification/pruning, new opacity
  branches and camera/view-conditioning are disabled.

The code is copied byte-for-byte from the working implementation for runtime
modules. Shared feature-flagged compatibility helpers remain present, but the
baseline launcher rejects their activation. No debug-code removal or numerical
refactor was performed. `configs/baselines/g3_surf.json` records all four
parameter groups, including explicit inactive/default compatibility flags.

## Recorded c1_sigmoid2_t4_v4 result

36 test views; source experiment `output/g3_connectivity_G3-Surf/c1_sigmoid2_t4_v4`.

| PSNR | SSIM | LPIPS-Alex | Depth MSE | branch train min | render FPS | peak train VRAM MiB |
|---:|---:|---:|---:|---:|---:|---:|
|27.05709076|0.70273048|0.54089481|0.0021381152|80.98|78.35|4016.74|

Depth is the existing **normalized depth metric**, not calibrated physical
depth. The checkpoint audit passed: canonical nodes, geometry/appearance K4,
saved KNN99, graph topology/weights and all reloaded trainable tensors exact.
Rendered reload max difference was0.00034624 (accepted1e-3 tolerance).
CUDA atomic accumulation means full retraining is not promised bitwise exact.
Timing refers to this continuation process, not all earlier source preparation.

Small archived result/audit JSONs are under `docs/g3_surf_artifacts/`.
`source_manifest.json` records SHA256s for external checkpoints/references and
the copied runtime files. Large binaries, datasets and renders are not in Git.

## Environment

Recorded environment: Python3.9.25, PyTorch2.6.0+cu126, torchvision0.21.0+cu126,
NumPy1.26.4, cuvs-cu12 24.8.0, fpsample1.0.2, mmcv1.6.0, lpips0.1.4.
Miniconda environment `colonsplat`; use `/usr/local/cuda-12.6` explicitly
because the system default `nvcc` can point to a different CUDA release.
Keep the repository license/acknowledgements and original rasterizer unchanged.

```bash
conda create -n colonsplat python=3.9
conda activate colonsplat
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install 'numpy==1.26.4' 'setuptools<81'
pip install -r requirements.txt
pip install cuvs-cu12==24.8.0 --extra-index-url https://pypi.nvidia.com
git submodule update --init --recursive
# The vendored rasterizer's nested .gitmodules is not a top-level gitlink.
# The recorded build uses GLM headers from the conda environment instead.
conda install -c conda-forge glm
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-radegs
pip install --no-build-isolation ./submodules/simple-knn
```

## Run / replay

The exact experiment continues the shared **pre-G3 fine2k** model, Adam,
CPU/CUDA/Python/NumPy RNG and camera stack to30k (28,000 matched updates).
The source pipeline performed100 coarse +2k fine first. Restarting from a
fresh initializer is **not** this matched protocol.

Obtain these external artifacts separately, preserving their directory tree:

- Dataset `data/coloncrafter/c1_sigmoid2_t4_v4` (RGB, depth, cameras).
- `output/node_connectivity_common_warmup/c1_sigmoid2_t4_v4/point_cloud/iteration_2000`
  and its `fine_state_2000.pth` (including optimizer/RNG/camera stack).
- `output/baseline1_nodes2000/c1_sigmoid2_t4_v4/point_cloud/iteration_1`.
- `output/adaptive_deformation_matched_knn99.pth`.
- `output/g3_connectivity_ablation/c1_sigmoid2_t4_v4/references/G3-Surf.pth`.
- `output/persistent_split_study/c1_sigmoid2_t4_v4/fixed_g3_motion_seed1.json`
  (the historical safety/monitor reference, not the training seed).

The exact hashes are in the manifest. References should not be regenerated
from a different geometry or checkpoint. The completed source30k checkpoint
remains on the original machine at
`/mnt/storage/Manh/ColonSplat-adaptive-deformation/output/g3_connectivity_G3-Surf/c1_sigmoid2_t4_v4/point_cloud/iteration_30000`;
`fine_state_30000.pth` is alongside `cfg_args` in its model directory.

The new standalone launcher is independent of old R0/Rcap queues. Default is
a dry run; **packaging does not start training**.

```bash
# Print the command only; artifact-root may point to the original machine tree.
python tools/run_g3_surf_baseline.py train --artifact-root /path/to/artifacts
# Explicitly run, refusing to overwrite any existing output directory.
python tools/run_g3_surf_baseline.py train --artifact-root /path/to/artifacts --execute
python tools/run_g3_surf_baseline.py evaluate --artifact-root /path/to/artifacts --execute
python tools/run_g3_surf_baseline.py audit --artifact-root /path/to/artifacts --execute
```

Use `--data /path/to/scene` and `--output /path/to/new/run` if needed. Evaluate
and audit write diagnostics inside that output directory. For read-only replay
of an archived checkpoint, use `tools.experiment_common.load_scene` plus the
existing renderer; do not invoke the training launcher.

## Verification

```bash
python -m unittest discover -s tests -p 'test_g3_surf_baseline.py' -v
python -m unittest discover -s tests -p 'test_graph_forward.py' -v
python -m unittest discover -s tests -p 'test_graph_determinism.py' -v
python -m unittest discover -s tests -p 'test_connectivity_ablation.py' -v
python -m unittest discover -s tests -p 'test_residual_appearance.py' -v
```

Tests cover the fixed baseline contract/path relocation, zero-init graph
residuals, graph/appearance checkpoint roundtrips, matched RNG handling and
the optional deterministic graph mode (not enabled in this archived run).

Read-only checkpoint replay (run from the original artifact root so its
historical `cfg_args` relative paths are valid):

```bash
cd /path/to/artifacts
python /path/to/ColonGraph/tools/verify_g3_surf_backup.py \
  output/g3_connectivity_G3-Surf/c1_sigmoid2_t4_v4
```

Serialized state equality is bit-exact; CUDA graph aggregation and rasterizer
replay use numerical tolerances. Packaging does not enable deterministic graph
mode or change the old run to make these audits pass.
