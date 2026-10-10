---
name: kernellib-review
description: Review a change or pull request in kernellib against the repo's own rules — CODE_REVIEW.md, the kernel / operator / feature-map / estimator / numerics contracts in AGENTS.md, the import boundaries, and reuse of gaussx and geonnax. Use when asked to review a diff, branch or PR in this repo.
---

# Review a kernellib change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   which layers it touches (kernels, operators, spectral, regression,
   dependence, graph, decomposition, sklearn).
2. **Reuse** — run the `reuse-reviewer` subagent on the diff:
   hand-written solves, Cholesky, CG, Woodbury, random-feature arithmetic or
   distance code are the main way kernellib drifts from gaussx and geonnax
   (and from its own `functional` layer).
3. **Numerics** — run the `numerics-reviewer` subagent in parallel with
   step 2: r = 0 gradients, traced control flow, dtype promotion,
   densified matrix-free paths, false operator tags, PRNG misuse,
   unconverged solves, tolerances without provenance.
4. **Boundaries** — `import kernellib` loads no scikit-learn, pynndescent,
   numba, numpyro, pipekit or optax (optional backends import inside the
   function that needs them, with an install hint); `kernellib.sklearn` imports the core, never
   the reverse; priors belong in pyrox, structured linear algebra in
   gaussx.
5. **Contracts** — for each layer touched, the rules in `AGENTS.md`'s
   contracts: array-leaf hyperparameters and static code-path fields, the
   ZOO entry, the `functional` twin (variance before lengthscale),
   `_KERNEL_OPERATORS` + `lx.diagonal`, `fit` returning a new module,
   required keys, the scikit-learn adapter in `ESTIMATORS`.
6. **Checklist** — the rest of `CODE_REVIEW.md` (public API, docs entry,
   capability index, executable examples, tests against dense references),
   skipping what ruff, ty or the tests already enforce.
7. **Verify claims** — for anything you flag as a bug, run it: the
   computation against a dense reference, `jax.grad` at coincident points,
   the float32 case, `jax.jit` around the call.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
