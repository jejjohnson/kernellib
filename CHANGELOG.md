# Changelog

## [0.0.17](https://github.com/jejjohnson/kernellib/compare/v0.0.16...v0.0.17) (2026-10-07)


### Features

* **decomposition:** eigen_solver="lobpcg", a JAX-native sparse eigensolver for graph embeddings ([#174](https://github.com/jejjohnson/kernellib/issues/174)) ([d667fd1](https://github.com/jejjohnson/kernellib/commit/d667fd13d5620432953c85d9e3bebe2607cb845d)), closes [#91](https://github.com/jejjohnson/kernellib/issues/91)
* **regression:** CG iteration stats and non-throwing solves on KRR, jit select_landmarks ([#168](https://github.com/jejjohnson/kernellib/issues/168)) ([17cf82e](https://github.com/jejjohnson/kernellib/commit/17cf82e7b7fc53a668e594b9bbb359b7f65150ad))
* **regression:** laplacian_penalty accepts an AbstractGraph ([#165](https://github.com/jejjohnson/kernellib/issues/165)) ([1faccea](https://github.com/jejjohnson/kernellib/commit/1faccea1d63412a03efd707b6b7693eb534db61c)), closes [#153](https://github.com/jejjohnson/kernellib/issues/153)


### Bug Fixes

* **decomposition:** centre the KernelPCA pre-image targets ([#167](https://github.com/jejjohnson/kernellib/issues/167)) ([3a83a09](https://github.com/jejjohnson/kernellib/commit/3a83a090e1baf531f07718f018fa432d2fa58af6))
* **decomposition:** check Lanczos eigenpair residuals and grow the Krylov space until they converge ([#171](https://github.com/jejjohnson/kernellib/issues/171)) ([acce208](https://github.com/jejjohnson/kernellib/commit/acce20866823f1ba8e8f3b9f95e5d261e5ab6245))
* **graph:** neighbour backends return distances in X's dtype ([#170](https://github.com/jejjohnson/kernellib/issues/170)) ([35a7002](https://github.com/jejjohnson/kernellib/commit/35a7002e6dee130ed71a4d064555d3b5abbc0a7a))
* **graph:** tell obtuse hull edges from non-Delaunay edges in mesh_graph ([#164](https://github.com/jejjohnson/kernellib/issues/164)) ([7b7759a](https://github.com/jejjohnson/kernellib/commit/7b7759a112f974f488c55730ee1ba5fc588b584b))
* **regression:** float32 / small-lambda robustness for KRR, Falkon and EigenPro ([#172](https://github.com/jejjohnson/kernellib/issues/172)) ([961ee41](https://github.com/jejjohnson/kernellib/commit/961ee4146265c6e516edcfc5afb7898212a6f15d)), closes [#162](https://github.com/jejjohnson/kernellib/issues/162)


### Performance Improvements

* **graph:** closed-form path/cycle eigenpairs for Kronecker laplacian_eigpairs ([#166](https://github.com/jejjohnson/kernellib/issues/166)) ([c8912f7](https://github.com/jejjohnson/kernellib/commit/c8912f79c0b2371edf0cad6ae74ec9c833e46fed))

## [0.0.16](https://github.com/jejjohnson/kernellib/compare/v0.0.15...v0.0.16) (2026-10-05)


### Features

* **decomposition:** graph-input eigenmaps, combine_potentials, SEP and kernel LPP / SEP (K5) ([#146](https://github.com/jejjohnson/kernellib/issues/146)) ([cc29fd6](https://github.com/jejjohnson/kernellib/commit/cc29fd651bdbdba91b7d9a98daa63c454c370a69))
* **graph:** add Delaunay, Gabriel and relative-neighbourhood proximity graphs (K15) ([#145](https://github.com/jejjohnson/kernellib/issues/145)) ([fb96a39](https://github.com/jejjohnson/kernellib/commit/fb96a39f2fe24eff2062c4ee586370e24afaea2d))

## [0.0.15](https://github.com/jejjohnson/kernellib/compare/v0.0.14...v0.0.15) (2026-10-03)


### Features

* **graph:** add GMRF null spaces, scaled structure matrices and mesh graphs (K6) ([#139](https://github.com/jejjohnson/kernellib/issues/139)) ([60dcaa5](https://github.com/jejjohnson/kernellib/commit/60dcaa5f96009c44f425ce9004078e614b7178c8))
* **graph:** laplacian_eigpairs and n_components_graph (K3) ([#137](https://github.com/jejjohnson/kernellib/issues/137)) ([74888ac](https://github.com/jejjohnson/kernellib/commit/74888acc9ad5c96bc8648de6ae75e9ba570a439c))

## [0.0.14](https://github.com/jejjohnson/kernellib/compare/v0.0.13...v0.0.14) (2026-10-03)


### Features

* **decomposition:** randomized KernelPCA via gx.randomized_eigh (K10) ([#134](https://github.com/jejjohnson/kernellib/issues/134)) ([60d098c](https://github.com/jejjohnson/kernellib/commit/60d098caffd3b3a4d7712020082eaa0ea2b41e08)), closes [#118](https://github.com/jejjohnson/kernellib/issues/118)
* **graph:** graph builders and radius neighbours (K2, part 2) ([#130](https://github.com/jejjohnson/kernellib/issues/130)) ([a8e57fe](https://github.com/jejjohnson/kernellib/commit/a8e57fe5a335c3b341f42bd67431ed6426ab98b6)), closes [#110](https://github.com/jejjohnson/kernellib/issues/110)
* **graph:** sparse graph types and their gaussx operators (K2, part 1) ([#127](https://github.com/jejjohnson/kernellib/issues/127)) ([5755e7a](https://github.com/jejjohnson/kernellib/commit/5755e7ae2799f9f437d5a1b3015837f0a80fc8e9))
* **regression:** preconditioned KRR with Nystrom and RPCholesky (K9) ([#133](https://github.com/jejjohnson/kernellib/issues/133)) ([ca4e98a](https://github.com/jejjohnson/kernellib/commit/ca4e98ad55a6ab7a0d5b7a4d7a7290f7bf4bca1a)), closes [#117](https://github.com/jejjohnson/kernellib/issues/117)
* **spectral:** select_landmarks with RPCholesky and greedy pivoting (K8) ([#131](https://github.com/jejjohnson/kernellib/issues/131)) ([2b975ce](https://github.com/jejjohnson/kernellib/commit/2b975ce90a4df36ad436f907174980300b4624cb)), closes [#116](https://github.com/jejjohnson/kernellib/issues/116)

## [0.0.13](https://github.com/jejjohnson/kernellib/compare/v0.0.12...v0.0.13) (2026-10-01)


### Bug Fixes

* wave-1 review follow-ups (jit-safe checks, symmetric penalties, jitted plain KPCA) ([#108](https://github.com/jejjohnson/kernellib/issues/108)) ([96ba637](https://github.com/jejjohnson/kernellib/commit/96ba63738391ba287dcd98e0a44effb590d9f776))

## [0.0.12](https://github.com/jejjohnson/kernellib/compare/v0.0.11...v0.0.12) (2026-09-30)


### Features

* **decomposition:** supervised and fair KernelPCA, pre-images and center_cross_kernel (K14) ([#104](https://github.com/jejjohnson/kernellib/issues/104)) ([0bc6fa1](https://github.com/jejjohnson/kernellib/commit/0bc6fa11f2deddcd3ca8d4c2c094b40f78f31085))
* **dependence:** gradient-safe CKA, U-centred unbiased HSIC, CKAAccumulator and a Gaussian bandwidth (K12) ([#101](https://github.com/jejjohnson/kernellib/issues/101)) ([6af61dc](https://github.com/jejjohnson/kernellib/commit/6af61dc96231b208f725a1ee910c6cac08a8ab99))
* **regression:** quadratic-penalty KRR for fair KRR and LapRLS (K13) ([#103](https://github.com/jejjohnson/kernellib/issues/103)) ([97399d8](https://github.com/jejjohnson/kernellib/commit/97399d898bd7822352093779768dbc247a8699a3))

## [0.0.11](https://github.com/jejjohnson/kernellib/compare/v0.0.10...v0.0.11) (2026-09-29)


### Features

* **dependence:** distance kernel, distance correlation and energy distance ([#74](https://github.com/jejjohnson/kernellib/issues/74)) ([0477606](https://github.com/jejjohnson/kernellib/commit/0477606c3d64336b51c8d74f09ffd7daae78a67b))
* **dependence:** taylor_statistics for kernel taylor diagrams ([#75](https://github.com/jejjohnson/kernellib/issues/75)) ([df9578f](https://github.com/jejjohnson/kernellib/commit/df9578f07d93f969d748777ba0ee9fa1f839f2c1))

## [0.0.10](https://github.com/jejjohnson/kernellib/compare/v0.0.9...v0.0.10) (2026-09-29)


### Features

* **kernels:** Derivative and DerivativeIndexed, derivative kernels as objects ([#66](https://github.com/jejjohnson/kernellib/issues/66)) ([97ef44f](https://github.com/jejjohnson/kernellib/commit/97ef44f634e140891b2d449ffa42e2934d88d042))
* **kernels:** kernels from functions of the inputs, FeatureKernel and Modulated ([#63](https://github.com/jejjohnson/kernellib/issues/63)) ([7ecd52f](https://github.com/jejjohnson/kernellib/commit/7ecd52f0984172fc2c79e9fc0c0e40b5cbae7b73))
* **kernels:** nystrom_kernel and Residual, approximations as kernels ([#64](https://github.com/jejjohnson/kernellib/issues/64)) ([2ebe5c0](https://github.com/jejjohnson/kernellib/commit/2ebe5c04293e6a85d73e2dccaabe97a3881ff4a3))
* **kernels:** transform methods, elwise and is_stationary ([#65](https://github.com/jejjohnson/kernellib/issues/65)) ([b6c45d1](https://github.com/jejjohnson/kernellib/commit/b6c45d12029d7e00f3710cb6a1eeff18dd6812a1))
* **operators:** keep diagonal and low-rank Gram structure in to_operator ([#61](https://github.com/jejjohnson/kernellib/issues/61)) ([9b558ad](https://github.com/jejjohnson/kernellib/commit/9b558ad822047807746cf101b6b097557402b187))

## [0.0.9](https://github.com/jejjohnson/kernellib/compare/v0.0.8...v0.0.9) (2026-09-29)


### Features

* **functional:** keep low-rank operators low-rank through center_kernel and HSIC/CKA ([#49](https://github.com/jejjohnson/kernellib/issues/49)) ([027f4de](https://github.com/jejjohnson/kernellib/commit/027f4de518f97bd3565b21af150691886376bcbc))
* **kernels:** closed-form spectral density for RationalQuadratic ([#50](https://github.com/jejjohnson/kernellib/issues/50)) ([e69f158](https://github.com/jejjohnson/kernellib/commit/e69f158d3802b94db75ffbc7e60e090eff7a682f))
* **kernels:** spectral densities and frequency samplers for Scaled and Sum ([#51](https://github.com/jejjohnson/kernellib/issues/51)) ([16167ed](https://github.com/jejjohnson/kernellib/commit/16167ed9f99b2878709b278acb279d12fdf2bee4))
* **regression:** report the CG iterations falkon_solve used, and make tol work ([#48](https://github.com/jejjohnson/kernellib/issues/48)) ([ea18b76](https://github.com/jejjohnson/kernellib/commit/ea18b769c792331a84421228bb90fac501ad19ed))
* **spectral:** leverage-score landmark selection for NystromFeatures ([#53](https://github.com/jejjohnson/kernellib/issues/53)) ([4e3eb44](https://github.com/jejjohnson/kernellib/commit/4e3eb44c6f6a8167dc7ec8136da5f18702f3af84))


### Bug Fixes

* **regression:** stream EigenPro's beta estimate in a lax.scan ([#47](https://github.com/jejjohnson/kernellib/issues/47)) ([6d9eabc](https://github.com/jejjohnson/kernellib/commit/6d9eabc5d84f9d6c102a1e2fca87f0f59db750e1))

## [0.0.8](https://github.com/jejjohnson/kernellib/compare/v0.0.7...v0.0.8) (2026-09-26)


### Features

* **decomposition:** kernel PCA, graph kernels, Laplacian and Schrödinger eigenmaps, LPP ([#43](https://github.com/jejjohnson/kernellib/issues/43)) ([b4078d5](https://github.com/jejjohnson/kernellib/commit/b4078d5b24a079864c3c055d24ed44e80122b2ed))

## [0.0.7](https://github.com/jejjohnson/kernellib/compare/v0.0.6...v0.0.7) (2026-09-26)


### Features

* phase 5 algorithms — heuristics, KRR, dependence measures, Falkon and EigenPro ([#30](https://github.com/jejjohnson/kernellib/issues/30)) ([9999b4d](https://github.com/jejjohnson/kernellib/commit/9999b4d479ba64ff983de8278e44ad6e785cd6d8))
* **sklearn:** opt-in scikit-learn adapters for estimators, feature maps, HSIC and MMD ([#31](https://github.com/jejjohnson/kernellib/issues/31)) ([4095ba5](https://github.com/jejjohnson/kernellib/commit/4095ba546d9d41d29410458969c325dac84f98ec))

## [0.0.6](https://github.com/jejjohnson/kernellib/compare/v0.0.5...v0.0.6) (2026-09-26)


### Features

* **spectral:** feature maps (RFF, ORF, FastFood, Nyström) and Laplace-eigenfunction features ([#28](https://github.com/jejjohnson/kernellib/issues/28)) ([d458da1](https://github.com/jejjohnson/kernellib/commit/d458da1e52a6878198f98cea9cd157c686b2c711))

## [0.0.5](https://github.com/jejjohnson/kernellib/compare/v0.0.4...v0.0.5) (2026-09-25)


### Features

* **spectral:** spectral densities, frequency samplers and RFF prior draws ([#22](https://github.com/jejjohnson/kernellib/issues/22)) ([d318219](https://github.com/jejjohnson/kernellib/commit/d3182199427cf216ab8af95ca60517ff5aa42468))

## [0.0.4](https://github.com/jejjohnson/kernellib/compare/v0.0.3...v0.0.4) (2026-09-25)


### Features

* **operators:** add FastFood random features ([#19](https://github.com/jejjohnson/kernellib/issues/19)) ([864f536](https://github.com/jejjohnson/kernellib/commit/864f536dbce7ed4480c2831da02d41fbecf8a53c))


### Bug Fixes

* **operators:** register lineax.diagonal; port gaussx kernel notebooks and guidance ([#18](https://github.com/jejjohnson/kernellib/issues/18)) ([0dca689](https://github.com/jejjohnson/kernellib/commit/0dca689bdca09fef28b94ed313595505b6493d89))

## [0.0.3](https://github.com/jejjohnson/kernellib/compare/v0.0.2...v0.0.3) (2026-09-25)


### Features

* **regression:** move Falkon, EigenPro and stable_rbf_kernel from gaussx ([#16](https://github.com/jejjohnson/kernellib/issues/16)) ([2864cac](https://github.com/jejjohnson/kernellib/commit/2864cac15211a04267ccab8a7fb252e0056d483b))

## [0.0.2](https://github.com/jejjohnson/kernellib/compare/v0.0.1...v0.0.2) (2026-09-25)


### Features

* **functional:** move kernel statistics from gaussx; add unbiased HSIC and CKA ([#12](https://github.com/jejjohnson/kernellib/issues/12)) ([99123b2](https://github.com/jejjohnson/kernellib/commit/99123b2fbe3ddcc36a58f55af3a3ae02f6306bf6))
* **functional:** port pure kernel functions from pyrox-gp ([#9](https://github.com/jejjohnson/kernellib/issues/9)) ([095c905](https://github.com/jejjohnson/kernellib/commit/095c905bdbd25f4940d68f21b8085002e3135ce2))
* **kernels:** kernel classes, composition, and the to_operator bridge ([#14](https://github.com/jejjohnson/kernellib/issues/14)) ([ba00616](https://github.com/jejjohnson/kernellib/commit/ba00616e17ad247bc5d81240db447da4c219b4b9))
* **operators:** move kernel operators from gaussx ([#11](https://github.com/jejjohnson/kernellib/issues/11)) ([1231826](https://github.com/jejjohnson/kernellib/commit/12318260c7dec6e9aac0278f12588097d4487c20))

## Changelog

## Changelog

All notable changes to this project will be documented in this file.

See [Conventional Commits](https://www.conventionalcommits.org/) for commit guidelines.
