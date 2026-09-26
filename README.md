# ColonSplat: Reconstruction of Peristaltic Motion in Colonoscopy with Dynamic Gaussian Splatting  

## Clone repository

```bash
git clone --recursive https://github.com/wMito/ColonSplat.git
cd ColonSplat
```

## Environment

Install conda environment. Please note we use gaussian rasterizer from https://github.com/HKUST-SAIL/RaDe-GS.  

```bash
conda create -n env python=3.9 
conda activate env

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

python -m pip install "setuptools<81"
pip install cuvs-cu12 --extra-index-url https://pypi.nvidia.com

pip install -r requirements.txt

pip install --no-build-isolation submodules/diff-gaussian-rasterization-radegs
pip install --no-build-isolation submodules/simple-knn

```

## To run training, testing and visualisation simply run:
```bash
# Train on C3VDv2
bash run.sh
# Train on DynamicColon
bash run_custom_ds.sh
```

## Stable B1-bbox + R6 baseline

The reproducible backup baseline is tagged `b1-bbox-r6-stable-v1`. It uses
standard deterministic FPS, a fixed 2,000-node K=4 inverse-distance graph,
direct node translation with bbox-normalized node-MLP coordinates, and the R6
node-conditioned residual appearance branch (8-D node feature, 16-D Gaussian
embedding, 13-D time encoding, `37-64-64-3`, `0.3*tanh`, multiplicative color,
`lambda_res=0`). Weighted FPS and adaptive-node changes are intentionally not
part of this revision.

Run the matched protocol with:

```bash
bash run_joint_b1_bbox_r6.sh full c1_sigmoid2_t4_v4
```

The completed `c1_sigmoid2_t4_v4` run reached PSNR `27.4416`, SSIM `0.70838`,
LPIPS-Alex `0.52820`, and Depth MSE `0.00210943`. Its local checkpoint remains
at `output/joint_b1_bbox_r6/c1_sigmoid2_t4_v4/point_cloud/iteration_30000`;
generated checkpoints and datasets are intentionally excluded from Git. See
[`docs/joint_b1_bbox_r6_results.md`](docs/joint_b1_bbox_r6_results.md) for the
protocol, efficiency measurements, and checkpoint audit.

## Weighted-FPS 2,000 + 400 baseline

This branch adds a fixed-budget geometry variant that preserves all 2,000
Standard-FPS coverage nodes and appends 400 total-gradient importance nodes.
The graph is fixed at K=4 and geometry, shared node deformation, and R6
appearance are trained jointly for 30k fine iterations after 100 coarse
iterations. The best completed run reached PSNR `27.45388`, SSIM `0.70893`,
LPIPS-Alex `0.52434`, and Depth MSE `0.00164310`.

Run it with `bash run_fps_plus_importance.sh`. See
[`docs/weighted_fps_2000_plus_400_baseline.md`](docs/weighted_fps_2000_plus_400_baseline.md)
for the exact protocol, local artifact paths, audits, and reproducibility note.

## Data:  

All data used in our work is available here: https://zenodo.org/records/18763383


## Acknowledgements:  
We thank the authors of publicly available repositories: 
- [ENDO-4DGX](https://github.com/lastbasket/Endo-4DGX). 
- [EndoPlanar](https://github.com/ThatphumCpre/EndoPlanar)
- [SurgicalGS](https://github.com/neneyork/SurgicalGS)
- [Endo-4DGS](https://github.com/lastbasket/Endo-4DGS)
- [Deform3DGS](https://github.com/jinlab-imvr/Deform3DGS)
- [RADE-GS](https://github.com/HKUST-SAIL/RaDe-GS)
- [C3VDv2](https://durrlab.github.io/C3VDv2/) - we used C3VDv2 meshes to create DynamicColon.


## Implementation notes on fair comparison with baselines
- We noticed diffrent baselines have differently implemented `LPIPS` measurement (input normalization). We pass image in range 0-1 and use normalization inside model. We applied the same computation for all baselines (see how we do it in `utils/image_utils.py/lpips_score`).  
- For fair comparison of geometry, we do not use densification and pruning for baseline methods on DynamicColon. We mostly do not want to prune Gaussians which could potentially lower the scores CH and HD95 for other baselines. We densely initialize Gaussians with point cloud from Blender which should be sufficient for reconstruction.  
- Baseline methods handle depth differently - some relying on normalized depth, others on metric depth - their original depth losses are not always directly compatible with AnyDepth/ColonCrafter depth maps. This in practice led to convergence issues. To ensure a fair comparison, we apply the same L1 depth loss on normalized depth maps across all methods. We also tune depth loss weights and training iterations for each baseline to obtain the most satisfactory renderings and consistent geometry for each method.
