# Joint B1-bbox + R6 — c1_sigmoid2_t4_v4

Completed on 2026-09-22 with CUDA 12.6 / PyTorch 2.6.0 on an RTX 2000 Ada.

## Protocol

- Geometry: B1-bbox, 2,000 nodes, K=4, direct node position mode, original
  KNN-99 loss 0.01, depth weight 0.25, and unchanged coarse/fine optimizer,
  densification, and pruning settings.
- Appearance: node feature 8D, Gaussian embedding 16D, 13D continuous time
  encoding, MLP 37-64-64-3, `0.3*tanh`, multiplicative composition, and
  `lambda_res=0`.
- Schedule: 100 coarse + 30,000 fine iterations.

## Final quality

| Metric | B1-bbox + HexPlane | B1-bbox + joint R6 | R6 - baseline |
|---|---:|---:|---:|
| PSNR (dB) | 27.228401 | 27.441599 | +0.213198 |
| SSIM | 0.704199 | 0.708378 | +0.004179 |
| LPIPS-Alex | 0.542648 | 0.528195 | -0.014453 |
| Depth MSE | 0.00274184 | 0.00210943 | -0.00063241 (-23.07%) |

## Efficiency

| Measurement | HexPlane | Joint R6 | Change |
|---|---:|---:|---:|
| Color forward, median | 32.2727 ms | 7.1864 ms | 4.49x faster |
| Full deformation forward, median | 34.2502 ms | 9.0988 ms | 3.76x faster |
| Full deformation inference, median | 34.1795 ms | 9.0429 ms | 3.78x faster |
| Render FPS | 6.0947 | 6.3527 | +4.23% |
| Training wall time | 2:42:25 | 1:27:23 | -46.20% / 1.86x faster |
| Peak PyTorch VRAM | 5,730.97 MB | 3,415.91 MB | -2,315.06 MB / -40.40% |

The baseline VRAM value is from a matched-topology 100-coarse + 10-fine
memory probe. The graph is fixed and optimizer state is materialized after the
first fine update, so this captures the baseline training allocation without
re-running its full 30k optimization. The R6 value is from the complete 30k
run. Historical wall times and FPS were recorded on the same workstation but
under different background CPU load; the paired CUDA-event branch timings are
the cleaner speed comparison.

## Audits

- Nine CPU unit/regression tests and the real-render integration test passed.
- The 100 + 1,000 smoke run passed loss, gradient, bound, graph,
  geometry/depth stability, and exact checkpoint-reload checks.
- The full audit sampled 301 training points. Fixed K4 graph and bbox buffers
  remained exact; the unused HexPlane checkpoint stayed bitwise unchanged.
- Checkpoint reload reproduced RGB, depth, alpha, transformed XYZ, and color
  residual exactly (maximum absolute difference 0 for every tensor).
- Final sampled residual: mean absolute value 0.23728, maximum 0.30000001,
  saturation ratio 44.39%. Mean saturation across sampled training points was
  33.74%, with a maximum of 79.10%. The branch frequently uses the configured
  bound, but no eta/loss tuning was performed in this controlled run.
- All stages (`train`, `audit`, `render`, `metrics`, `profile`) exited 0.

## Conclusion

On this scene, joint R6 is not only more efficient than HexPlane; it also
improves every requested final metric. The strongest gain is Depth MSE
(-23.07%), while RGB improves modestly (+0.213 dB PSNR, higher SSIM, lower
LPIPS). The small end-to-end FPS change despite a 3.76x deformation speedup
shows that rasterization and the remaining render pipeline dominate inference.
