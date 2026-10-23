# Packaging record

Source: ColonGraph-stage11, U7/R completed fine30k, original Stage11 protocol and fixed TRAIN masks. Original raw model/Adam/RNG state files are byte copies; cfg_args and deformation_nodes.pth config strings were normalized to @bundle/@dataset paths. Only load/save metadata code changed in nativeGaussianModel. U7 engine keeps exact math and removes the unrelated P route.

No experiment outputs are symlinked. No source/training/evaluation path needs the original Stage experiment folders; u7.py installs an IO guard rejecting those accesses. Dataset and installed Conda/CUDA extensions remain external dependencies.

assets/provenance.json records original state/config source hashes. MANIFEST.json checks final packaged bytes. verification/core_equivalence.json checks native equation equality; validation/smoke/metric-equivalence/relocation reports cover two scenes. Probe checkpoint copies were removed after successful checks.

Simple-KNN source was populated from ColonGraph-GSurf/submodules/simple-knn; GLM headers from the tested Conda include tree. Native extension binaries and environment were not copied. Licenses retained in source submodules.

The existing 30k results are preserved; no full training/evaluation was repeated as part of packaging. Only3TRAIN updates per scene and2TEST metric probes were used to check the package.
