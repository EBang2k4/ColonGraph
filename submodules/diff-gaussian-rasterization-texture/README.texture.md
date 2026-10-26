# RGB-only canonical texture extension

This separate `diff_gaussian_rasterization_texture` package extends the
vendored ColonGraph RaDe-GS renderer. It keeps the original package installed
for the matched baseline. See `../../README.texture.md` for the experiment.

Extra inputs are centered RGB texels `[N,9,3]`, packed differentiable local-ray
parameters `[N,12]`, and R6 texture gains `[N,3]`. `texture.cuh` implements
bilinear per-contribution sampling and backward. No alpha, sorting, covariance
or geometry-preprocessing changes are introduced.

Original copyright notices and `LICENSE.md` apply to the retained renderer.
This run uses the GLM headers in the active Conda environment's include path
(`$CONDA_PREFIX/include/glm`), exported by the activation script.
