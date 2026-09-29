# Geometry-regularizer DOF audit

This audit records the exact behavior that must be preserved or deliberately
changed by the revised ARAP experiment.

## Existing Gaussian KNN-99 term

`GaussianModel.rerun_knn(..., k=100)` queries canonical Gaussian XYZ and drops
the self match.  `closest_point_indices` therefore contains 99 neighbors for
each Gaussian.  Matched experiments reload the same saved index tensor from
`output/adaptive_deformation_matched_knn99.pth`.

For every sampled frame, the loss computes the mean transformed position of
those 99 neighbors and penalizes the squared distance from the transformed
center Gaussian to that mean:

```text
mu_i(t) = mean_{j in KNN99(i)} x_j(t)
L_KNN = 0.01 * mean_i ||x_i(t) - mu_i(t)||^2
```

The weight is constant at `0.01`; there is no warm-up or decay.  Historically
the term runs during both 100 coarse and 30k fine iterations.

In direct B1 deformation,

```text
x_i(t) = canonical_xyz_i + sum_k w_ik * node_translation_k(t).
```

Consequently KNN-99 has direct gradients to:

- canonical Gaussian XYZ;
- the translation output rows of `NodeDeformationMLP`;
- the shared hidden layers of that MLP.

It has no direct gradient to Gaussian/node rotation, scale, opacity, or color.
Because the node MLP hidden trunk is shared, changing its hidden weights can
indirectly change all output branches on later forwards, but the instantaneous
ARAP/KNN derivative of the rotation/scale output rows is zero.

KNN-99 is a centroid/Laplacian smoothing term, not a rigidity constraint.  It
also regularizes a Gaussian residual degree of freedom that node ARAP cannot
touch: canonical per-Gaussian XYZ.

## Revised node ARAP term

The v1 ARAP graph is the symmetric union of canonical node 6-NN edges.  Its
topology, Gaussian-distance weights, and median canonical local scales are
fixed.  At a current training timestamp and an independently sampled global
timestamp, it fits an SO(3) rotation per node by weighted SVD, applies the
determinant correction, detaches the rotation, and uses the fixed-scale
normalized residual from the experiment specification.

Node position is `canonical_node_xyz + predicted_node_translation`.  Canonical
node XYZ is a buffer.  Therefore node ARAP directly regularizes only the node
translation trajectory and shared MLP trunk.  It does **not** replace the
canonical-Gaussian-XYZ smoothing effect of KNN-99.

## Canonical XYZ component retained in the revised replacement

The revised A0 branch explicitly keeps

```text
L_canonical = 0.01 * mean_i ||x_i^0 - mean_{j in KNN99(i)} x_j^0||^2.
```

It uses the same frozen KNN-99 index tensor.  This term has gradients only to
canonical per-Gaussian XYZ and no path to `NodeDeformationMLP`.  Node ARAP then
replaces only the time-varying deformation smoothing role.  At zero
deformation, `L_canonical` is exactly the historical KNN-99 loss.

## Controlled replacement

To guarantee identical K0/A0 starting states, the 100 coarse iterations retain
the matched KNN-99 term for every branch.  At the start of fine training all
branches reload the same untrained iteration-1 state.  The experimental switch
then applies only during fine training:

- K0: KNN-99 on, ARAP off;
- A0: KNN-99 off, ARAP on.

More precisely, revised A0 uses canonical-XYZ Laplacian + node ARAP; it does
not leave canonical XYZ unregularized.

Thus “KNN-99 off” in the result table means off for the full 30k fine stage;
the common coarse initializer is not an experimental variable.  This detail is
required when interpreting canonical XYZ drift in A0.

## Revised gradient calibration

The first single-frame calibration was rejected: the sampled K0 checkpoint had
zero accumulated alpha and therefore a genuinely zero render gradient to the
deformation MLP.  Dividing by that zero produced a meaningless lambda near
`2e-21`.

The revised audit loads checkpoints at 2k/5k/10k and uses 16 deterministic
global temporal pairs at each checkpoint.  For every `t1`, it renders the
training frame and calls `torch.autograd.grad` directly on all
`NodeDeformationMLP` parameters for the RGB/depth/TV reconstruction loss.  It
separately refits ARAP over `(t1,t2)`, calls `autograd.grad` on the same
parameters, averages parameter gradients over the 16 samples, and takes the
ratio of the two aggregate norms.  The audit hard-fails if coverage or at least
75% of per-frame reconstruction gradients are not meaningful.
