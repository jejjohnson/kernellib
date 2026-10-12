r"""Matrix-valued kernels for vector fields $F : \mathbb{R}^D \to \mathbb{R}^D$.

- `CurlFree(k)`: $F = \nabla\Phi$, the gradient of a scalar potential
  $\Phi \sim \mathcal{GP}(0, k)$.
- `DivergenceFree(k)`: $F = \operatorname{rot}\nabla\Psi$ in 2-D (the
  stream-function construction), and its $D$-dimensional analogue.
- `Helmholtz(k_Φ, k_Ψ)`: their sum, the Helmholtz-decomposition prior of
  Berlinghieri et al. (2023).

**Output layout.** A matrix-valued kernel $K(x, x') \in \mathbb{R}^{D\times D}$
is exposed as a *scalar* kernel on component-indexed rows ``[x, i]``,

$$
k([x, i], [x', j]) = K(x, x')_{ij} = \operatorname{cov}(F_i(x), F_j(x')),
$$

the convention of `DerivativeIndexed` (whose row ``[x, i]``, ``i >= 0``, is
$\partial_i f(x)$: `CurlFree` is exactly `DerivativeIndexed` restricted to
gradient rows). So these kernels keep the ``(N1, N2)`` Gram contract and
work unchanged with `to_operator` (dense or ``implicit=True``), noise,
``+`` / ``*`` and the solvers; a component missing at some location is a
dropped row. `vector_inputs` builds the canonical **point-major** rows
``[x_n, i]`` at position ``n * D + i``, so the Gram is the
``(n i), (m j)`` flattening of the 4-D block array `matrix_gram` returns.
`derivative_inputs(X, values=False)` gives the same rows component-major.
"""

from __future__ import annotations

from abc import abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from kernellib._einx import einsum, rearrange
from kernellib._kernels._base import AbstractKernel, AbstractPointwiseKernel
from kernellib._kernels._derivative import _check_differentiable, derivative_inputs
from kernellib._kernels._stationary import RBF


__all__ = ["CurlFree", "DivergenceFree", "Helmholtz", "vector_inputs"]


def _cross_hessian(
    kernel: AbstractKernel, x: Float[Array, " D"], y: Float[Array, " D"]
) -> Float[Array, "D D"]:
    r"""$H_{ab} = \partial_{x_a}\partial_{x'_b} k(x, x')$.

    Closed form for an exact `RBF` (isotropic or ARD), with
    $s = 1/\ell^2$, $h = x - x'$, $u = s \odot h$:
    $H = \sigma^2 e^{-\frac12 \sum_a s_a h_a^2}\,(\operatorname{diag}(s) - u u^\top)$.
    Otherwise forward-over-reverse autodiff of ``kernel.pairwise``.
    """
    if type(kernel) is RBF:
        s = jnp.broadcast_to(1.0 / kernel.lengthscale**2, x.shape)
        h = x - y
        u = s * h
        e = kernel.variance * jnp.exp(-0.5 * jnp.sum(u * h))
        return e * (jnp.diag(s) - einsum(u, u, "a, b -> a b"))
    return jax.jacfwd(jax.grad(kernel.pairwise, argnums=0), argnums=1)(x, y)


def _divergence_free(H: Float[Array, "D D"]) -> Float[Array, "D D"]:
    r"""$\operatorname{tr}(H)\,I - H^\top$; equals $R H R^\top$ when $D = 2$."""
    d = H.shape[0]
    if d < 2:
        raise ValueError(
            f"DivergenceFree needs inputs of dimension >= 2; got {d} (in 1-D the "
            "only divergence-free field is constant)."
        )
    return jnp.trace(H) * jnp.eye(d, dtype=H.dtype) - rearrange(H, "a b -> b a")


class AbstractVectorFieldKernel(AbstractPointwiseKernel):
    r"""A matrix-valued kernel $K(x, x') \in \mathbb{R}^{D \times D}$.

    Subclasses implement `matrix`. ``pairwise`` reads the component index
    from the last coordinate of each row ``[x, i]`` (see the module
    docstring) and returns ``matrix(x, x')[i, j]``. Kept private to
    ``_geo``; the public contract is the scalar kernel on indexed rows.
    """

    @abstractmethod
    def matrix(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, "D D"]:
        """The ``(D, D)`` block ``cov(F(x), F(y))``."""
        raise NotImplementedError

    def matrix_gram(
        self, X1: Float[Array, "N1 D"], X2: Float[Array, "N2 D"]
    ) -> Float[Array, "N1 N2 D D"]:
        """All ``(D, D)`` blocks between two sets of locations.

        Args:
            X1: Locations, shape ``(N1, D)``.
            X2: Locations, shape ``(N2, D)``.

        Returns:
            ``(N1, N2, D, D)`` blocks. ``rearrange(B, "n m i j -> (n i) (m j)")``
            is the Gram on `vector_inputs` rows, computed ``D**2`` times
            cheaper than evaluating that Gram entry by entry.
        """
        return jax.vmap(lambda x: jax.vmap(lambda y: self.matrix(x, y))(X2))(X1)

    def pairwise(
        self, x: Float[Array, " D1"], y: Float[Array, " D1"]
    ) -> Float[Array, ""]:
        i = jnp.round(x[-1]).astype(jnp.int32)
        j = jnp.round(y[-1]).astype(jnp.int32)
        return self.matrix(x[:-1], y[:-1])[i, j]


class CurlFree(AbstractVectorFieldKernel):
    r"""Curl-free kernel: the covariance of a gradient field $F = \nabla\Phi$.

    $$
    K_{\mathrm{cf}}(x, x') = \nabla_x \nabla_{x'}^\top k(x, x'),
    $$

    which for a stationary $k(h)$, $h = x - x'$, is $-\nabla_h^2 k(h)$ (minus
    the Hessian). For the RBF $k = \sigma^2 e^{-r^2/2\ell^2}$, $r = \|h\|$:

    $$
    K_{\mathrm{cf}} = \frac{\sigma^2}{\ell^2} e^{-r^2/2\ell^2}
    \left(I - \frac{h h^\top}{\ell^2}\right).
    $$

    **Why it is PSD.** $\nabla$ is a linear operator, so $F = \nabla\Phi$ is
    a Gaussian process and $K_{\mathrm{cf}}$ is its covariance: for any
    points and weights, $\sum c_{ni} c_{mj} K_{ij}(x_n, x_m) =
    \operatorname{var}(\sum c_{ni} F_i(x_n)) \ge 0$. In 2-D every sample
    has zero scalar curl, $\partial_0 F_1 - \partial_1 F_0 = 0$.

    Pseudocode::

        H[a, b] = d/dx_a d/dy_b k(x, y)   # closed form for RBF, else autodiff
        K(x, y) = H
        k([x, i], [y, j]) = K(x, y)[i, j]

    Numerics: the base kernel must be pointwise and twice differentiable
    (mean-square differentiable GP: `RBF`, `RationalQuadratic`,
    `Matern` with ``nu >= 1.5``); `Matern(nu=0.5)`, `White` and rough
    `Distance` kernels raise. The block is evaluated in the inputs' dtype;
    the RBF path is closed form (any lengthscale shape, including ARD), the
    rest is forward-over-reverse autodiff of ``pairwise``, exact at
    coincident points for kernels smooth in $r^2$.

    Attributes:
        kernel: Kernel of the scalar potential $\Phi$.

    References:
        Narcowich, F. J. & Ward, J. D. (1994). Generalized Hermite
        interpolation via matrix-valued conditionally positive definite
        functions. *Mathematics of Computation* 63(208), 661-687.

        Fuselier, E. J. (2008). Sobolev-type approximation rates for
        divergence-free and curl-free RBF interpolants. *Mathematics of
        Computation* 77(263), 1407-1423.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.CurlFree(kl.RBF(lengthscale=0.5))
        >>> k.matrix(jnp.zeros(2), jnp.zeros(2)).tolist()  # sigma^2 / l^2 * I
        [[4.0, 0.0], [0.0, 4.0]]
        >>> X = jnp.array([[0.0, 0.0], [0.3, -0.2]])
        >>> k(kl.vector_inputs(X), kl.vector_inputs(X)).shape  # (N D, N D)
        (4, 4)
    """

    kernel: AbstractKernel

    def __check_init__(self) -> None:
        _check_differentiable(self.kernel, "CurlFree")

    def matrix(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, "D D"]:
        return _cross_hessian(self.kernel, x, y)


class DivergenceFree(AbstractVectorFieldKernel):
    r"""Divergence-free kernel: the covariance of a rotated gradient field.

    In 2-D, $F = R\nabla\Psi = (\partial_1\Psi, -\partial_0\Psi)$ for a
    stream function $\Psi \sim \mathcal{GP}(0, k)$ and
    $R = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$, so with
    $H = \nabla_x\nabla_{x'}^\top k$,

    $$
    K_{\mathrm{df}} = R\,H\,R^\top = \operatorname{tr}(H)\,I - H^\top .
    $$

    The right-hand side is the kernel in any $D \ge 2$: it is the
    covariance of $F = \sum_{a<b} (e_a\partial_b - e_b\partial_a)\Psi_{ab}$
    with independent $\Psi_{ab} \sim \mathcal{GP}(0, k)$ (in 3-D, the curl of
    a vector potential with i.i.d. components), and for stationary $k$ it is
    Narcowich & Ward's $(\Delta I - \nabla\nabla^\top)\,k(h)$. For the RBF:

    $$
    K_{\mathrm{df}} = \frac{\sigma^2}{\ell^2} e^{-r^2/2\ell^2}
    \left(\frac{h h^\top}{\ell^2} + \Big(D - 1 - \frac{r^2}{\ell^2}\Big) I\right).
    $$

    **Why it is PSD.** It is the covariance of a linear differential
    operator applied to independent GPs, so every Gram is a covariance
    matrix. Each term $\partial_b\partial_a\Psi - \partial_a\partial_b\Psi$
    of $\nabla\cdot F$ vanishes, so every sample, and every column
    $K_{\cdot j}(\cdot, x')$, is divergence-free.

    Pseudocode::

        H[a, b] = d/dx_a d/dy_b k(x, y)   # closed form for RBF, else autodiff
        K(x, y) = trace(H) * I - H.T      # = R H R.T in 2-D
        k([x, i], [y, j]) = K(x, y)[i, j]

    Numerics: as `CurlFree` (twice-differentiable base; rough kernels raise;
    inputs' dtype; closed-form RBF path). Inputs must have ``D >= 2``.

    Attributes:
        kernel: Kernel of the stream function(s) $\Psi$.

    References:
        Narcowich, F. J. & Ward, J. D. (1994). *Mathematics of Computation*
        63(208), 661-687.

        Fuselier, E. J. (2008). *Mathematics of Computation* 77(263),
        1407-1423.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.DivergenceFree(kl.RBF(lengthscale=0.5))
        >>> k.matrix(jnp.zeros(3), jnp.zeros(3)).tolist()  # (D - 1) sigma^2 / l^2 I
        [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    """

    kernel: AbstractKernel

    def __check_init__(self) -> None:
        _check_differentiable(self.kernel, "DivergenceFree")

    def matrix(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, "D D"]:
        return _divergence_free(_cross_hessian(self.kernel, x, y))


class Helmholtz(AbstractVectorFieldKernel):
    r"""Helmholtz-decomposition kernel: curl-free plus divergence-free parts.

    $F = \nabla\Phi + R\nabla\Psi$ with independent
    $\Phi \sim \mathcal{GP}(0, k_\Phi)$ and $\Psi \sim \mathcal{GP}(0, k_\Psi)$
    (Berlinghieri et al., 2023), so

    $$
    K = K_{\mathrm{cf}}[k_\Phi] + K_{\mathrm{df}}[k_\Psi],
    $$

    PSD as a sum of covariances. The two parts have separate
    hyperparameters, so a fit can say how much of an ocean current is
    divergent (upwelling, fronts) and how much rotational (eddies). It is
    `CurlFree` ``+`` `DivergenceFree` sharing the cross-Hessian machinery.

    Pseudocode::

        K(x, y) = CurlFree(potential).matrix(x, y)
                + DivergenceFree(stream).matrix(x, y)

    Numerics: as `CurlFree`, for both kernels.

    Attributes:
        potential: Kernel of the scalar potential $\Phi$ (curl-free part).
        stream: Kernel of the stream function $\Psi$ (divergence-free part).

    References:
        Berlinghieri, R., Trippe, B. L., Burt, D. R., Giordano, R.,
        Srinivasan, K., Özgökmen, T., Xia, J. & Broderick, T. (2023).
        Gaussian processes at the Helm(holtz): a more fluid model for ocean
        currents. *ICML*.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> k = kl.Helmholtz(kl.RBF(lengthscale=0.5), kl.RBF(variance=2.0))
        >>> k.matrix(jnp.zeros(2), jnp.zeros(2)).tolist()  # 4 I + 2 I
        [[6.0, 0.0], [0.0, 6.0]]
    """

    potential: AbstractKernel
    stream: AbstractKernel

    def __check_init__(self) -> None:
        _check_differentiable(self.potential, "Helmholtz")
        _check_differentiable(self.stream, "Helmholtz")

    def matrix(
        self, x: Float[Array, " D"], y: Float[Array, " D"]
    ) -> Float[Array, "D D"]:
        return _cross_hessian(self.potential, x, y) + _divergence_free(
            _cross_hessian(self.stream, x, y)
        )


def vector_inputs(X: Float[Array, "N D"]) -> Float[Array, "ND D1"]:
    """Point-major rows ``[x_n, i]`` for the vector-field kernels.

    Row ``n * D + i`` is component ``i`` at location ``n``, so stack
    observations the same way, ``rearrange(F, "n i -> (n i)")`` for an
    ``(N, D)`` array of vectors. The rows are those of
    `derivative_inputs` ``(X, values=False)``, reordered point-major.

    Args:
        X: Locations, shape ``(N, D)``.

    Returns:
        ``(N * D, D + 1)`` rows.

    Examples:
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> kl.vector_inputs(jnp.array([[1.0, 2.0], [3.0, 4.0]])).tolist()
        [[1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [3.0, 4.0, 0.0], [3.0, 4.0, 1.0]]
    """
    return rearrange(
        derivative_inputs(X, values=False), "(i n) c -> (n i) c", n=X.shape[0]
    )
