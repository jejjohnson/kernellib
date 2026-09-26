# Dependence

Dependence measures on kernels and data. The matrix-level versions (Gram
matrices in, scalar out) are in [Functional](functional.md); these take
kernels and samples.

| Function | Measures | Estimators |
|---|---|---|
| `hsic` | dependence between paired samples | biased, unbiased (Song et al., 2012) |
| `cka` | HSIC normalised to ``[0, 1]`` | biased, unbiased |
| `kernel_alignment` | uncentred alignment of two Gram matrices | — |
| `mmd_squared` | difference between two distributions | biased, unbiased, linear-time |
| `permutation_test` | a p-value for any of the above | exact permutation test |

```python
import jax
import kernellib as kl

kx = kl.RBF(lengthscale=kl.estimate_lengthscale(X))
ky = kl.RBF(lengthscale=kl.estimate_lengthscale(Y))

h = kl.hsic(kx, ky, X, Y)
h = kl.hsic(kx, ky, X, Y, estimator="unbiased")
c = kl.cka(kx, ky, X, Y)
m = kl.mmd_squared(kx, X, X_other, estimator="linear")  # O(N)

result = kl.permutation_test(
    lambda X, Y: kl.hsic(kx, ky, X, Y), X, Y, key=key, n_permutations=500
)
result.p_value
```

## Randomised estimates

Pass an unfitted feature map as ``approx`` and each Gram matrix is replaced by
$\Phi\Phi^\top$. The statistic is computed from the features, in
``O(N R_x R_y)`` time and ``O(N (R_x + R_y))`` memory, never forming an
``N x N`` matrix. For HSIC and CKA the map is fitted once per kernel with
independent keys; for MMD it is fitted once on the pooled sample.

```python
h = kl.hsic(kx, ky, X, Y, approx=kl.NystromFeatures(300, key))
h = kl.hsic(kx, ky, X, Y, approx=kl.RandomFourierFeatures(1024, key))
m = kl.mmd_squared(kx, X, X_other, approx=kl.FastFoodFeatures(1024, key))
```

Everything is differentiable, so ``jax.grad`` of HSIC with respect to a
lengthscale (bandwidth selection) or the inputs (sensitivity) works directly.

::: kernellib.hsic

::: kernellib.cka

::: kernellib.kernel_alignment

::: kernellib.mmd_squared

::: kernellib.permutation_test

::: kernellib.PermutationTestResult
