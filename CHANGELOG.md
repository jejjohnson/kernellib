# Changelog

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
