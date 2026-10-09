---
name: code-review
description: Review a change or pull request in kernellib against CODE_REVIEW.md, the contracts in AGENTS.md (kernels, operators and feature maps, estimators, JAX numerics and einx), the boundaries with gaussx / geonnax / pyrox, and reuse of existing primitives.
---

# Code review

Read "Boundaries", "Reuse before you write" and "The contracts" in
[`AGENTS.md`](../../../AGENTS.md); for every function, class or module the
diff adds, search [`docs/api/capabilities.md`](../../../docs/api/capabilities.md)
for an existing equivalent; then apply [`CODE_REVIEW.md`](../../../CODE_REVIEW.md)
and report in its format.
