# Building with agents

kernellib exists so that a kernel is one object whose hyperparameters can
be differentiated, and whose Gram matrix can be solved at scale by gaussx
without being formed. Coding agents tend to re-implement what they cannot
see — a distance matrix and an `exp`, a Cholesky of `K + σ²I`, `cos(XW + b)`
random features, an HSIC estimator — and each one loses gradients,
structure or scale. kernellib ships three things that let them find it.

## The capability index

The [capability index](https://jejjohnson.github.io/kernellib/reference/capabilities/)
lists every public name in `kernellib`, `kernellib.functional` and
`kernellib.sklearn`, grouped by the module that defines it, with a one-line
summary; then the public API of gaussx and of geonnax's random-feature and
basis modules, which kernellib builds on. It is regenerated from the code
and checked in the test suite, so it never drifts.

## The Claude Code plugin

The repository is a Claude Code plugin marketplace. In any project:

```text
/plugin marketplace add jejjohnson/kernellib
/plugin install kernellib@kernellib
```

The plugin adds:

- **`kernel-methods-with-kernellib`** (skill) — loads whenever a task
  computes a kernel or Gram matrix, solves a kernel system, approximates a
  kernel with features, fits a kernel regressor or measures dependence:
  what lives where, the rules (hyperparameters are leaves, solves through
  gaussx, fits return new objects, explicit keys), a worked example, and a
  "don't write it — use kernellib" table.
- **`kernellib-reuse-reviewer`** (subagent) — a read-only check of a diff
  for kernel-method code kernellib already provides, and for misuse.

## llms.txt

For other agents and tools, the docs site serves
[`llms.txt`](https://jejjohnson.github.io/kernellib/llms.txt): a curated map
of kernellib and its key pages.

## Rules for your project's `AGENTS.md`

Paste this into the agent instructions of a project that builds on
kernellib:

```markdown
## Kernel methods: build on kernellib

This project uses kernellib (kernels, kernel operators, feature maps, kernel
regression, dependence measures) on top of gaussx (structured linear
algebra). Before writing a kernel, a distance matrix, a solve with a Gram
matrix, random features, a kernel regressor or an HSIC / MMD statistic,
search the capability index
(https://jejjohnson.github.io/kernellib/reference/capabilities/) or
`kernellib.__all__`, and compose what exists:

- Kernels are `kernellib` classes (`kl.RBF(lengthscale=...)`, `kl.Matern`,
  composites with `+` / `*`); their hyperparameters are array leaves, so
  rebuild a kernel to change them and let `jax.grad` reach them.
- Solve with `gaussx.solve(kl.to_operator(kernel, X, noise=s2), y)`, never
  `jnp.linalg.solve` / `cho_solve` on a Gram; for large N use
  `implicit=True` with an iterative gaussx strategy, or
  `kl.KRR(preconditioner="nystrom")`, `kl.Falkon`, `kl.EigenPro`.
- Approximate kernels with `kl.RandomFourierFeatures` /
  `kl.NystromFeatures` (`.fit(kernel, X)` returns a fitted copy).
- Dependence: `kl.hsic`, `kl.cka`, `kl.mmd_squared`, with
  `kl.permutation_test` for p-values.
- Pass an explicit, freshly split PRNG key to every random routine.
```

## Working on kernellib itself

Contributors (and their agents) follow
[`AGENTS.md`](https://github.com/jejjohnson/kernellib/blob/main/AGENTS.md) in
the repository: the layer map, the boundaries with gaussx, geonnax and
pyrox, "reuse before you write", the contracts and the tests that enforce
them, and recipe skills for adding kernels, feature maps, operators,
estimators, dependence measures and graph components.
