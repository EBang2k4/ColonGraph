# Weighted FPS after coarse — joint B1-bbox + R6

Run: `bash run_weighted_fps_r6.sh`.

The queued pipeline checks resources, runs regression tests, runs a 100-coarse /
20-fine integration smoke and both audits, and only then runs 100-coarse /
30,000-fine training, render, metrics, profiling, and the comparison report.
Existing outputs are never overwritten. A process lock prevents duplicate runs.
Resource threshold is 7 GiB MemAvailable and an idle GPU with 8,000 MiB free;
the historical full run peaked near 5.5 GiB host RSS. No other job is stopped.

Only the initial deformation-node sampler changes. Every coarse backward pass
records the total-loss XYZ gradient before clipping for Gaussians with
`radii > 0`. Invisible Gaussians do not add observations even when the KNN loss
produces a gradient. Float64 moments store grad_sum, grad_sq_sum and grad_count.
Variance is clamped to zero before square root; confidence is min(1, count/5).
P5/P95 is computed across all canonical Gaussians, including unseen zero scores.
A constant score vector falls back to zero importance; the experiment audit
rejects this degenerate case instead of presenting it as an effective ablation.

Selection starts at Gaussian index 0, as standard FPS. The score is Euclidean
nearest distance times (1 + importance), with alpha=1. Internally the code
compares d²(1 + importance)², an equivalent ranking which avoids square roots.
This is deliberately different from d²(1 + importance). Selected indices are
excluded explicitly, so duplicate coordinates cannot produce duplicate IDs.

The prior standard-FPS R6 experiment restores an untrained post-coarse reference
before fine. This experiment retains that paired initialization. Gradient
statistics come from the new 100 coarse iterations with unchanged Gaussian
identity/topology; observed post-coarse XYZ is checked against reference XYZ
before applying its importance. Only the graph is rebuilt after reference
restore. Initial canonical attributes, embedding, node MLP, normalization, and
R6 initialization are audited against standard R6 for exact equality.

The original cuVS KNN-99 loss stays unchanged. Historical KNN-99 indices were
not saved, so equal algorithm and seed do not establish bitwise identical
approximate-neighbor indices across runs. The fixed Gaussian-to-node K4 graph
is a separate structure: it necessarily changes with selected nodes and is
then checked to remain bitwise fixed through fine training.

Artifacts under output/joint_b1_bbox_r6_weighted_fps/c1_sigmoid2_t4_v4:

- weighted_fps_sampling.pth: raw moments, counts, importance, canonical XYZ,
  observed post-coarse XYZ, and selected Gaussian IDs.
- weighted_fps_audit.json: paired initialization and selection/graph checks.
- node_distribution.csv: per-node position, importance, K4 association count,
  sum of weights and effective Gaussian support.
- node_spatial_distribution.png and node_influence_distribution.png.
- joint_r6_audit.json: fine gradients, geometry stability and reload checks.
- weighted_fps_comparison.json / .md: final comparison to standard-FPS R6.

Influence statistics describe canonical graph weights, not renderer pixel
contributions. Neither inference nor checkpoint loading resamples nodes.

## Controlled alpha sweep

`bash run_weighted_fps_alpha_sweep.sh` evaluates `alpha=0.25` and `0.5`
sequentially. Both branches reuse the audited coarse-gradient moments and
importance tensor from the completed `alpha=1` run, while still executing the
same 100-coarse/30k-fine schedule. The canonical snapshot, seed, losses,
optimizer, fixed 2,000-node K4 graph construction, and R6 appearance settings
are unchanged. Adaptive nodes are not enabled. Each smoke run must pass the
sampling, graph, geometry, gradient, and checkpoint-reload audits before its
full run begins.
