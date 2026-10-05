# API Reference

Everything listed in `kernellib.__all__` is importable from the top level:

```python
import kernellib as kl

kl.__version__
```

The submodules are an implementation detail you are welcome to reach into,
but never have to. Modules appear here as each phase of the
[architecture](../architecture/) lands.

## Modules

| Module | Contents |
|---|---|
| [Kernels](kernels.md) | `AbstractKernel`, `AbstractPointwiseKernel`, `AbstractStationaryKernel`, `RBF`, `Matern`, `RationalQuadratic`, `Periodic`, `Cosine`, `White`, `Constant`, `Linear`, `Polynomial`, `Distance`, `Sum`, `Product`, `Scaled`, `ActiveDims`, `Warped`, `Stretch`, `Shift`, `Periodised`, `FeatureKernel`, `Modulated`, `nystrom_kernel`, `Residual`, `Derivative`, `DerivativeIndexed`, `derivative_inputs`, `to_operator`, `to_cross_operator` |
| [Functional](functional.md) | `rbf_kernel`, `matern_kernel`, `rational_quadratic_kernel`, `periodic_kernel`, `cosine_kernel`, `linear_kernel`, `polynomial_kernel`, `distance_kernel`, `white_kernel`, `constant_kernel`, `kernel_add`, `kernel_mul`, `stable_rbf_kernel`; `centering_operator`, `center_kernel`, `hsic`, `cka`, `mmd_squared` |
| [Operators](operators.md) | `KernelOperator`, `ImplicitKernelOperator`, `ImplicitCrossKernelOperator`, `implicit_cross_kernel`, `batched_kernel_matvec`, `batched_kernel_rmatvec`, `nystrom_operator`, `rff_operator`, `fastfood_params`, `fastfood_features`, `fastfood_operator`, `FastFoodParams`, `fastfood_frequencies`, `hadamard_transform` |
| [Spectral](spectral.md) | `AbstractStationaryKernel.spectral_density` / `sample_frequencies`, `AbstractFeatureMap`, `RandomFourierFeatures`, `OrthogonalRandomFeatures`, `FastFoodFeatures`, `NystromFeatures`, `select_landmarks`, `LaplaceEigenfunctionFeatures`, `draw_rff_cosine_basis`, `evaluate_rff_cosine_paths` |
| [Heuristics](heuristics.md) | `estimate_lengthscale`, `lengthscale_to_gamma`, `gamma_to_lengthscale`, `lengthscale_grid` |
| [Dependence](dependence.md) | `hsic`, `cka`, `kernel_alignment`, `mmd_squared`, `distance_covariance_squared`, `distance_correlation_squared`, `energy_distance`, `taylor_statistics`, `TaylorStatistics`, `permutation_test`, `PermutationTestResult` |
| [Regression](regression.md) | `AbstractEstimator`, `KRR`, `Falkon`, `EigenPro`, `falkon_preconditioner`, `falkon_solve`, `falkon_predict`, `FalkonPreconditioner`, `FalkonInfo`, `eigenpro_preconditioner`, `eigenpro_step_size`, `eigenpro_correction`, `EigenProPreconditioner` |
| [Graphs](graph.md) | `AbstractGraph`, `Graph`, `GraphTopology`, `GridGraph`; `knn_graph`, `graph_from_neighbors`, `radius_graph`, `radius_neighbors`, `grid_graph`, `graph_from_adjacency`, `graph_from_edges`, `edge_weights`; `delaunay_graph`, `gabriel_graph`, `relative_neighborhood_graph`; `nearest_neighbors`, `KNNGraph`, `adjacency_matrix`, `graph_laplacian`; `laplacian_eigpairs`, `n_components_graph`; `matern_graph_kernel`, `diffusion_kernel`, `regularized_laplacian_kernel`, `random_walk_kernel`, `cosine_graph_kernel`, `commute_time_kernel`; `structure_matrix`, `graph_null_space`, `mesh_graph` |
| [Decomposition](decomposition.md) | `KernelPCA` (dense, randomized and feature-map paths; supervised / fair; pre-images), `LaplacianEigenmaps`, `SchrodingerEigenmaps`, `LocalityPreservingProjections`, `SchrodingerEigenmapProjections`, `KernelLocalityPreservingProjections`, `KernelSchrodingerProjections`, `laplacian_eigenmap`, `schrodinger_eigenmap`, `combine_potentials`, `barrier_potential`, `label_potential`, `spatial_spectral_graph`, `spatial_spectral_potential` |
| [scikit-learn](sklearn.md) | Optional `kernellib.sklearn`: `KernelRidge`, `FalkonRegressor`, `EigenProRegressor`, feature-map transformers, `HSIC`, `MMD` |

## Package overview

::: kernellib
    options:
      members: false
      show_root_heading: false
      show_source: false
