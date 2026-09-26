# Weighted-FPS baseline: 2,000 coverage + 400 importance nodes

This branch packages the strongest completed additive-node configuration for
`c1_sigmoid2_t4_v4`. It retains the complete 2,000-node Standard-FPS coverage
basis and appends 400 Gaussians selected using total-loss position-gradient
importance collected during the 100-iteration coarse stage.

## Fixed protocol

- 100 coarse iterations
- 2,000 Standard-FPS coverage nodes + 400 importance nodes
- fixed canonical Gaussian-to-node K=4 graph with normalized inverse-distance weights
- B1 direct node deformation and bbox-normalized shared node MLP
- original Gaussian KNN-99 loss (`0.01`)
- normalized depth loss weight `0.25`
- R6 appearance: 8-D node feature, 16-D Gaussian embedding, 13-D time PE,
  `37 -> 64 -> 64 -> 3`, `0.3*tanh`, multiplicative composition
- joint geometry, node-deformation and appearance optimization for 30,000 fine iterations
- no fine-stage Gaussian densification or pruning

Run the matched pipeline with:

```bash
bash run_fps_plus_importance.sh
```

The controlled script expects the audited coarse-importance artifact at:

```text
output/joint_b1_bbox_r6_weighted_fps_a050/c1_sigmoid2_t4_v4/weighted_fps_sampling.pth
```

Generated datasets, importance tensors, renders and checkpoints are excluded
from Git. The best local checkpoint remains at:

```text
output/joint_b1_bbox_r6_fps2000_plus_importance400/c1_sigmoid2_t4_v4/point_cloud/iteration_30000
```

## Results

| Run | PSNR | SSIM | LPIPS-Alex | Depth MSE |
|---|---:|---:|---:|---:|
| Original best | 27.453882 | 0.708933 | 0.524343 | 0.00164310 |
| Independent reproduction | 27.414158 | 0.707545 | 0.523755 | 0.00225910 |

Sampling, node placement, K4 construction, checkpoint reload and joint-R6
audits passed. The original Gaussian KNN-99 graph is built with cuVS IVF-flat
and was not persisted by the historical run. The two runs therefore begin
with slightly different KNN-99 losses despite identical Gaussian, deformation
node, K4 and appearance initialization. Persist `closest_point_indices` when
strict run-to-run geometry reproducibility is required.
