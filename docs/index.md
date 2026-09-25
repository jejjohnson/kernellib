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

:::{warning} Status: scaffold
This repository was reset from a 2018 numpy / scikit-learn package to a fresh
JAX package. The public API lands in phases described in the
[architecture guide](guide/architecture.md); nothing beyond `__version__` is
exported yet.
:::

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
| Run a GP without forming the kernel matrix | [Matrix-free GP](notebooks/matrix_free_gp.ipynb) |
| Read the API | [API reference](xref:api#kernellib) |
| Contribute | [Contributing](contributing.md) |
