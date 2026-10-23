# Codex Context — Stage 11: Fixing the 1.057 Scale-Gate Bug & Boundary Opacity Coverage to Pass All Simultaneous Targets

## 1. Objective

Repositories:
- `/home/robot/colongraph/ColonGraph-texture` (`DynamicColon` `transverse` and `rectum`)
- `/home/robot/colongraph/ColonGraph-r7-v2` (`C3VDv2` `sigmoid2`)

Resolve the two exact empirical bugs exposed in the Stage 10 report (`report_9.md`):
1. **Fix the `Scale-Gate` threshold bug (`1.0574552 > 1.05` and `0.914371 < 0.95`):**
   In Stage 10, the narrow `|r_scale - 1.0| <= 0.05` condition evaluated to **`metric term enabled = False`** on `U5` (`ratio = 1.0574552`), `U6` (`1.0574552`), and `U5_rectum` (`0.914371`), accidentally disabling Metric Depth supervision on all DynamicColon runs! This caused `U5` to lose `-0.59 dB` PSNR (`34.28 -> 33.69 dB`), `+134%` Raw Depth MSE (`0.0239 -> 0.0558`), and `+11.4%` CH (`0.0683 -> 0.0762`) compared to Stage 9 **`U3` (`34.2770 dB`, `CH = 0.068347`, `Raw Depth = 0.023895`)**.
2. **Fix the `GT -> Pred` `HD95` Boundary Coverage Deficit in `U3`:**
   As diagnosed in `report_9.md` (*"Dominant GT->Pred HD95 can reflect coverage deficits"*), `U3` branched from `F0`/`A` at iteration `10,000`—just 1,000 steps after the native **`9,000` opacity reset (`opacity <= 0.01`)**—leaving grazing/peripheral boundary Gaussians with depleted opacity and slight inward boundary drift, which starved the opacity-weighted point-cloud sampler at the extreme tube ends (`GT -> Pred` `HD95`).
3. **Preserve Exact Graph Determinism on `C3VDv2 sigmoid2` (`Run P = 29.0388 dB`):**
   Instead of rebuilding `G3` (at 2k) and `J8` (at 5k) under nondeterministic CUDA atomics from `fine0`, load the exact `common_5000` checkpoint of `Run P` (or verify deterministic graph replay) so `P_unified` reproduces `Run P` (`29.0388 dB > 29.0 dB`).

## 2. Exact Mathematical Fixes for Stage 11 (`U7_transverse` and `U7_rectum`)

Start from the exact **`U3` (Stage 9) configuration** (which already achieved **`34.2770 dB PSNR`, `0.023895 Raw Depth MSE`, `0.000314 Norm Depth MSE`, and `0.068347 CH`** on `transverse`, and **`34.2985 dB PSNR`** on `rectum` with 2,000 FPS nodes), and apply these four surgical fixes:

### Fix 1: Continuous Sensor-Confidence Gate $\omega_{\mathrm{sensor}}$ (Replacing the Broken `0.05` Cutoff)
At `fine0`, given the measured median depth ratio $r_{\mathrm{scale}} = \frac{\operatorname{median}_{V}\hat D^{(0)}}{\operatorname{median}_{V} D^{(0)}}$:
$$\omega_{\mathrm{sensor}} = \begin{cases} 1.0, & \text{if } 0.50 \le r_{\mathrm{scale}} \le 2.00 \\ 0.0, & \text{otherwise} \end{cases}$$
- On `transverse` ($r_{\mathrm{scale}} = 1.0574552 \in [0.5, 2.0]$): **`omega_sensor = 1.0` (`enabled = True`)**!
- On `rectum` ($r_{\mathrm{scale}} = 0.9143711 \in [0.5, 2.0]$): **`omega_sensor = 1.0` (`enabled = True`)**!
- On `C3VDv2 sigmoid2` ($r_{\mathrm{scale}} = 729.81032 \notin [0.5, 2.0]$): **`omega_sensor = 0.0` (`enabled = False`)**!

### Fix 2: Full Native Autograd Normalized Depth + Metric Raw Depth Supervision
To simultaneously achieve `U3`'s record `Raw Depth MSE` (`0.0239`) AND `Run J`'s record `Normalized Depth MSE` (`0.000507` on `rectum`), keep **full autograd through $\max_{j \in V}\hat D_j$** in the native normalized depth loss $\mathcal{L}_D^{\mathrm{norm}}$ (`train.py:485`), and add the metric depth term gated by $\omega_{\mathrm{sensor}}$:
$$\mathcal{L}_{D}^{\mathrm{total}} = 0.25\,\mathcal{L}_{D}^{\mathrm{norm}}(\hat D, D) + \omega_{\mathrm{sensor}}\,\lambda_{\mathrm{raw}}\,\operatorname{mean}_{i \in V}\frac{\sqrt{(\hat D_i - D_i)^2 + 10^{-8}}}{\max_V D + 10^{-6}}$$
with $\lambda_{\mathrm{raw}} = 0.30$ (identical to `U3`).

### Fix 3: Restoring Post-9k Boundary Opacity & Coverage Lock (Eliminating `GT -> Pred` `HD95` Deficit)
Why did `F0` (`HD95 = 0.603264`) and `Run J` (`HD95 = 0.942597`) have better `HD95` than `U3` (`0.623743` and `1.061977`)?
1. **Restore Opacity from `fine0` at the 10k Branch Point (or Branch from Pre-9k Reset):**
   At iteration `10,000` when branching from `F0` (on `transverse`) and `A` (on `rectum`), the `9,000` opacity reset had just clamped peripheral Gaussians to `0.01`. At step `10,000` initialization of `U7`, set each Gaussian's opacity logit to the elementwise maximum of its `10k` opacity and its `fine0` (post-coarse) opacity:
   $$o_i^{10k \text{ init}} \leftarrow \max\left(o_i^{10k},\ o_i^{\mathrm{fine0}}\right)$$
   and maintain a soft floor $o_i(k) \ge \min(o_i^{\mathrm{fine0}}, \operatorname{logit}(0.50))$ for low-visibility peripheral Gaussians (bottom 15% observation count or outer 15% radial distance from scene centroid) from `10k -> 30k` so the opacity-weighted `HD95` point-cloud sampler never suffers coverage starvation at the colon tube ends!
2. **Camera-Independent 3D Centers & Boundary-Anchored `KNN99 = 0.01`:**
   - Do NOT project micro-displacements onto the evaluation camera ray (use camera-independent 3D world micro-displacement $\delta x_i^{\mathrm{micro}}(t)$ with tight bound $0.002a\tanh(\cdot)$ for interior points, and taper $\delta x_i^{\mathrm{micro}}(t) = \mathbf{0}$ on the outer 15% peripheral/low-visibility Gaussians so boundary points cannot drift).
   - In `Offset-Compensated KNN99 = 0.01` (which gave `U3` its record `CH = 0.068347`), apply curvature offset compensation $\gamma_{\mathrm{curv}, i} = 0.75$ on well-observed interior Gaussians so folds are not flattened (`CH` stays `< 0.069`), and taper $\gamma_{\mathrm{curv}, i} = 0.0$ on the outer 15% boundary Gaussians so the periphery stays tightly anchored against outward drift (`HD95 < 0.600` on `transverse`, `HD95 < 0.98` on `rectum`).

### Fix 4: Keep `U3`'s Proven Photometric & Cosine Schedule (`10k -> 30k`)
Keep the exact photometric objective and LR schedule that gave **`U3` `34.2770 dB`** on `transverse` and **`U3_rectum` `34.2985 dB`** on `rectum` (`0.80 Charb + 0.15 D-SSIM + 3.0 MSE`, `TV = 5e-4`, `R6 + Additive025`, Cosine LR decay `15k -> 30k` to `0.06x`, no `12k` opacity reset).

## 3. Execution & Verification Matrix

1. **`U7_transverse`** (2,000 FPS nodes, branching from `F0` at 10k with `omega_sensor = 1.0` verified `enabled = True`):
   - Verify simultaneous improvement over **both `F0` (`33.4416 dB`) and `U2` (`33.7653 dB`)** on all 4 metrics:
     - `PSNR > 34.20 dB`
     - `Depth_MSE_raw < 0.030` and `Depth_MSE_normalized < 0.00035`
     - `CH < 0.0700` (vs `F0` `0.073476`, `U2` `0.073407`)
     - `HD95 < 0.6030` (strictly beating `F0` `0.603264` and `U2` `0.621426`)
2. **`U7_rectum`** (2,000 FPS nodes, branching from `A` at 10k with `omega_sensor = 1.0` verified `enabled = True`):
   - Verify simultaneous improvement over **`Baseline A` (`33.5627 dB`)** on all 4 metrics:
     - `PSNR > 34.20 dB`
     - `Depth_MSE_raw < 0.45` and `Depth_MSE_normalized < 0.00080` (vs `A` `1.0106 / 0.001722`)
     - `CH < 0.2500` (strictly beating `A` `0.250110` and `J` `0.279017`)
     - `HD95 < 1.0100` (strictly beating `A` `1.017986`)
3. **`P_unified` on `C3VDv2 sigmoid2`** (`omega_sensor = 0.0` verified `enabled = False`, loading exact `Run P` `common_5000` graph checkpoint so `G3/J8` are bit-wise identical to `Run P`):
   - Verify `PSNR > 29.00 dB` (reproducing `Run P = 29.0388 dB`).

Report the complete tables, diagnostic logs confirming `metric term enabled = True` on `U7_transverse` and `U7_rectum` (and `False` on `P_unified`), both directional components of `HD95` (`Pred -> GT` and `GT -> Pred`), and all contrasts (`U7_transverse - F0`, `U7_transverse - U2`, `U7_transverse - U3`, `U7_rectum - A`, `U7_rectum - U3_rectum`, `P_unified - A`).