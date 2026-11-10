---
title: "Kernels to attention: kriging, Nadaraya–Watson and one attention head"
subtitle: "Guide · explainer · three gap-fillers for a cloudy methane scene, written as operators, from the GP posterior mean to softmax attention"
short_title: Kernels to attention
description: "Kriging, Nadaraya–Watson and an attention head each fill a cloud gap in a TROPOMI XCH₄ scene with a weighted sum of the clear cells. Written as operators with fixed signatures, they show that attention is a Nadaraya–Watson smoother on learned features: it keeps the kernel and the normalisation, and drops the solve that gives kriging its declustering and its variance."
date: 2026-11-10
keywords:
  - attention
  - kriging
  - Nadaraya–Watson
  - kernel smoother
  - Gaussian process
---

+++ {"part": "abstract"}

A cloud mask has punched holes in a day of TROPOMI methane. Kriging, Nadaraya–Watson and one attention head each fill a hole with a weighted sum of the cells they can see. Written as operators with fixed signatures, with shapes and units on every input and output, they show exactly what attention keeps from kriging and what it drops: it keeps a kernel and a row normalisation, and it drops the linear solve.

+++

{button}`Open the full page <./pages/kernels-to-attention.html>`

**Where it sits.** This is a teaching page, not API documentation: it uses kernellib for the kernel and the operator, and otherwise writes the maths out by hand. It continues from [Kernels on the Earth](./geo.md), which chooses the kernel, and asks what happens when the solve is replaced by a softmax. The full page is a self-contained document with operator cards, units checks, code sketches and an interactive lab (tap a cell and compare the three weight maps), which opens on its own and links back here.

## TL;DR: Four operators, one gap

Here $\mathbf{s}_n$ is the centre of clear cell $n$ (km), $y_n$ its XCH₄ (ppb), and $\mathbf{s}_*$ the cell under the cloud.

- **Kriging** gets its weights from a solve, $\mathbf{w}_* = (\mathbf{K}_{SS} + \sigma_\varepsilon^2 \mathbf{I})^{-1} \mathbf{K}_{*S}^\top$. The solve declusters redundant cells, so weights can be negative and need not sum to one, and it returns a posterior variance.
- **Nadaraya–Watson** drops the solve and normalises the kernel row: weights are non-negative and sum to one. With an RBF kernel on equal-norm inputs it is exactly a softmax of dot products.
- **An attention head** is Nadaraya–Watson on learned features, $P_{mn} = \mathrm{softmax}_n(\mathbf{q}_m^\top \mathbf{k}_n / \sqrt{D_q})$. Its kernel is asymmetric and indefinite: a smoothing kernel, not a Mercer kernel.
- **Tile tokens** (a ViT's "patches") make the kernel act between regions, which is learned non-local means. **Multi-scale** behaviour comes from heads with different lengthscales, from depth and from hierarchy, not from the softmax.
- **The test.** Under the plume cloud the truth is 1,930.0 ppb: kriging gives 1,924.0 ± 7.8 ppb, Nadaraya–Watson 1,915.9 ppb and a 20 km head 1,922.3 ppb. Only kriging says how far to trust its number.

## On the page

1. Every gap-filler is an operator: the four boxes with their inputs, parameters and outputs
2. The running example: a cloud over a Permian methane plume
3. Symbols
4. The lab: tap a cell and compare the kriging, Nadaraya–Watson and attention weights
5. Kriging solves for its weights: the GP posterior mean under four assumptions
6. Nadaraya–Watson normalises instead: a conditional mean under a kernel density estimate; an RBF is already a softmax
7. An attention head is Nadaraya–Watson on learned features; code sketches, including the same smoothers in kernellib
8. Tiles make it a kernel between regions; heads make it multi-scale
9. Worked numbers at the plume query
10. Next: random features, and the GP approximation behind each efficient attention
