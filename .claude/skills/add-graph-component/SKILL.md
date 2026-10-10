---
name: add-graph-component
description: Add a graph component to kernellib — a graph builder (kNN / radius / grid / mesh / from edges), an edge weigher, a Laplacian normalisation, a graph kernel (a spectral function of the Laplacian), or a graph embedding (Laplacian / Schrödinger eigenmaps, locality-preserving projections) — on the sparse Graph / GridGraph types. Use when asked to add or change anything in src/kernellib/_graph or the eigenmap / projection estimators in src/kernellib/_decomposition.
---

# Add a graph component

## 1. Make sure it does not exist yet

Builders `knn_graph`, `radius_graph`, `grid_graph`, `mesh_graph`,
`graph_from_neighbors` / `_edges` / `_adjacency`; neighbour search
`nearest_neighbors`, `radius_neighbors` (backends `"exact"`,
`"pynndescent"`, `"sklearn"`); the private weighers in `_graph/_weights.py`
(`heat_weights`, `local_heat_weights`, `cosine_weights`, `edge_weigher`);
`graph_laplacian`; `laplacian_eigpairs`; graph kernels
`diffusion_kernel`, `matern_graph_kernel`, `regularized_laplacian_kernel`,
`random_walk_kernel`, `cosine_graph_kernel`, `commute_time_kernel`;
embeddings `LaplacianEigenmaps`, `SchrodingerEigenmaps`, the projections,
`spatial_spectral_graph` and the potentials. Sparse Laplacians are gaussx
operators (`gaussx.KroneckerSum` on a lattice); GMRF precisions from a
graph belong to gaussx / pyrox-lgm.

## 2. Write it (`src/kernellib/_graph/`)

- **Graphs** are `AbstractGraph` modules (`_types.py`): a static
  `GraphTopology` (edge list) and traced `weights`, so a weight's gradient
  flows and the structure stays static under `jit`. Build them with the
  helpers in `_construct.py`; a builder that needs concrete values (an
  edge list) says so and calls `_concrete`.
- **Graph kernels** (`_kernels.py`): a function `name(W | graph, params,
  *, normalization=...)` that calls `_spectral(W, fn, normalization)` with
  `fn` applied to the clipped Laplacian eigenvalues; PSD by construction
  (state the condition on `fn`), dense N × N, so document the size limit.
- **Edge weighers** go through `edge_weigher` (`_weights.py`).
- **Embeddings** (`src/kernellib/_decomposition/`): subclass
  `_GraphEmbedding`, keep the `eigen_solver` options, take eigenpairs from
  `laplacian_eigpairs` (it fixes eigenvector signs, so embeddings are
  deterministic), and add the scikit-learn adapter to
  `kernellib.sklearn._decomposition` and `DECOMPOSITION` in
  `tests/sklearn/test_estimator_checks.py`.
- Optional backends (pynndescent, scikit-learn) are imported only when the
  backend is chosen, with an install hint naming the extra
  (`kernellib[neighbors]`, `kernellib[sklearn]`); `tests/test_imports.py`
  checks the core stays free of them.

## 3. Export, docs, tests

- `_graph/__init__.py` (or `_decomposition/__init__.py`),
  `src/kernellib/__init__.py`, `:::` entries on `docs/api/graph.md` /
  `decomposition.md`, the name in its module's row of `docs/api/index.md`, `make capabilities`.
- Tests (`tests/graph/`, `tests/decomposition/`): against a hand-built
  small graph (path, cycle, lattice with known spectrum), symmetry and PSD
  of kernels, sparse vs dense agreement, the gradient of the weights, and
  the eigen solvers agreeing with each other.

## 4. Verify

`make test`, `uv run pytest -n auto -m "slow and not integration" tests/graph
tests/decomposition`, then `pre-pr-check`.
