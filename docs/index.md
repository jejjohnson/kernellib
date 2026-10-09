# kernellib

> Kernels and scalable kernel methods for JAX, built on gaussx.

`kernellib` owns kernel functions and their composition, spectral densities
and feature maps, kernel ridge regression, dependence measures (HSIC, CKA,
MMD), kernel embeddings, and kernel derivatives. It does **not** own Gaussian
processes, priors, or inference; those live in
[pyrox-gp](https://github.com/jejjohnson/pyrox).

Scale is inherited rather than reimplemented. Every kernel can be turned into
a [gaussx](https://github.com/jejjohnson/gaussx) linear operator, and every
algorithm here solves through gaussx's solver strategies.

## Installation

kernellib is not on PyPI yet. Until it is, install from the repository:

::::{tab-set}
:::{tab-item} uv

```bash
uv add "kernellib @ git+https://github.com/jejjohnson/kernellib.git"
```
:::
:::{tab-item} From source

```bash
git clone https://github.com/jejjohnson/kernellib.git
cd kernellib
make install
```
:::
::::

The scikit-learn adapters in `kernellib.sklearn` need the optional extra,
`kernellib[sklearn]`; see [scikit-learn workflows](notebooks/sklearn_workflows.ipynb).

## The stack

```
lineax · matfree · equinox · einx          geonnax
          │                                   │
          ▼                                   │
        gaussx   ◄────────────────────────────┤   structured operators, solvers
          │                                   │
          ▼                                   │
       kernellib ◄────────────────────────────┘   kernels, kernel methods
          │
          ▼
       pyrox-gp                                   GP models with NumPyro priors
```

Two rules hold the chain together: gaussx never imports kernellib, and
kernellib never imports NumPyro.

## Where to go next

| I want to… | Start here |
|---|---|
| Understand what goes where | [Architecture](guide/architecture.md) |
| Approximate a kernel, or test independence | [Kernel approximations](notebooks/kernel_approximations.ipynb) |
| Measure similarity: RV, CKA, distance correlation, Taylor diagrams | [Similarity measures](notebooks/similarity_measures.ipynb) |
| Run a GP without forming the kernel matrix | [Matrix-free GP](notebooks/matrix_free_gp.ipynb) |
| Read the API | [API reference](xref:api#kernellib) |
| Contribute | [Contributing](contributing.md) |
