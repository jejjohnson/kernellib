# Dependence

Dependence measures on kernels and data. The matrix-level versions (Gram
matrices in, scalar out) are in [Functional](functional.md); these take
kernels and samples.

| Function | Measures | Estimators |
|---|---|---|
| `hsic` | dependence between paired samples | biased, unbiased (Song et al., 2012) |
| `cka` | HSIC normalised to ``[0, 1]`` | biased, unbiased |
| `CKAAccumulator` | CKA over a dataset, batch by batch | unbiased per batch (Nguyen et al., 2021) |
| `kernel_alignment` | uncentred alignment of two Gram matrices | — |
| `mmd_squared` | difference between two distributions | biased, unbiased, linear-time |
| `distance_covariance_squared` | dependence (Székely et al., 2007) | biased (V), unbiased (U-centred) |
| `distance_correlation_squared` | dCov normalised to ``[0, 1]`` | biased, unbiased |
| `energy_distance` | difference between two distributions | biased, unbiased, linear-time |
| `taylor_statistics` | radii, CKA and distance for a kernel Taylor diagram | biased, unbiased |
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

## Training with CKA, and CKA over a dataset

`cka` is safe as a training penalty. When either input is constant (a
network at initialisation, a collapsed representation), CKA is defined as
``0`` with a zero gradient, instead of ``0 / 0``. The unbiased HSIC is
computed from U-centred matrices, so it stays accurate in float32 even when
a Gram matrix is nearly constant.

To compare two representations over a dataset too large for one Gram
matrix, accumulate the unbiased HSIC terms batch by batch. Each term is
unbiased, so the result does not depend on the batch size (Nguyen, Raghu &
Kornblith, 2021):

```python
acc = kl.CKAAccumulator(kl.Linear(), kl.Linear())
for xb in batches:
    acc = acc.update(layer_a(xb), layer_b(xb))  # O(B^2) memory per batch
similarity = acc.result()
```

`CKAAccumulator` is a pytree, so it also works as a ``jax.lax.scan`` carry.

The [Comparing representations with CKA](../../representation-similarity/)
notebook compares the layers of neural networks with linear, RBF and
debiased CKA, and uses `CKAAccumulator` on a large evaluation set.

Dependence measures also work as penalties: the
[Dependence penalties](../../dependence-penalties/) notebook fits fair
kernel ridge regression with `hsic_penalty`, trains a network with a `cka`
penalty, and covers LapRLS and supervised / fair kernel PCA.

## Distance-based statistics

For a walk-through from correlation to these measures, see the
[Similarity measures](../../similarity-measures/) notebook.

Distance covariance, distance correlation and energy distance are HSIC, CKA
and MMD under the distance-induced kernel
$k(x, x') = \tfrac12(\|x\|^a + \|x'\|^a - \|x - x'\|^a)$ (Sejdinovic et
al., 2013), so they share every estimator and the ``approx`` path:

| Distance-based | Kernel |
|---|---|
| `distance_covariance_squared(X, Y)` | `4 * hsic(Distance(), Distance(), X, Y)` |
| `distance_correlation_squared(X, Y)` | `cka(Distance(), Distance(), X, Y)` |
| `energy_distance(X, Y)` | `2 * mmd_squared(Distance(), X, Y)` |

Use Nyström for ``approx`` here: `Distance` is not stationary and has no
spectral density.

## Taylor diagrams

Centred Gram matrices are vectors with inner product HSIC, so their norms,
their CKA and their distance obey the law of cosines, the same triangle a
Taylor diagram (Taylor, 2001) draws for standard deviation, correlation and
centred RMSE. `taylor_statistics` returns the four numbers from one set of
HSIC values, so the triangle closes exactly, including under ``approx``:

```python
k = kl.RBF(lengthscale=kl.estimate_lengthscale(X_ref))
for name, Y in models.items():
    s = kl.taylor_statistics(k, k, X_ref, Y)
    ax.plot(jnp.arccos(s.correlation), s.norm_y, "o", label=name)  # polar axes
```

With `Linear` kernels it is the RV-coefficient diagram; on 1-D data, the
classic diagram on a squared scale (variance and $\rho^2$).

::: kernellib.hsic

::: kernellib.cka

::: kernellib.CKAAccumulator

::: kernellib.kernel_alignment

::: kernellib.mmd_squared

::: kernellib.distance_covariance_squared

::: kernellib.distance_correlation_squared

::: kernellib.energy_distance

::: kernellib.taylor_statistics

::: kernellib.TaylorStatistics

::: kernellib.permutation_test

::: kernellib.PermutationTestResult
