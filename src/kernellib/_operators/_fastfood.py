r"""FastFood random features: RBF random features in loglinear time.

Random Fourier features approximate an RBF kernel with a dense Gaussian
frequency matrix $\Omega \in \mathbb{R}^{D \times d}$, costing $O(Dd)$ storage
and $O(Dd)$ time per input. FastFood (Le, Sarlós & Smola, 2013) replaces each
$d \times d$ block of $\Omega$ with the structured product

$$
V = \frac{1}{\ell \sqrt{d}}\, S H G \Pi H B,
$$

where $H$ is the Walsh-Hadamard matrix (applied by a fast transform in
$O(d \log d)$), $B$ is a diagonal of random signs, $\Pi$ a random permutation,
$G$ a Gaussian diagonal, and $S$ a diagonal that gives each row the length of a
$d$-dimensional Gaussian vector. Storage drops to $O(D)$ and evaluation to
$O(D \log d)$ per input. Inputs whose dimension is not a power of two are
zero-padded, which leaves pairwise distances unchanged.

The kernel approximation is returned as a `gaussx.LowRankUpdate`, exactly like
`rff_operator`, so solves and log-determinants go through Woodbury.

This implements the proposal in gaussx#62, which moved here with the rest of
the kernel layer.
"""

from __future__ import annotations

import math

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from jaxtyping import Array, Float, Int

from kernellib._einx import rearrange


__all__ = [
    "FastFoodParams",
    "fastfood_features",
    "fastfood_frequencies",
    "fastfood_operator",
    "fastfood_params",
    "hadamard_transform",
]


def _is_power_of_two(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def hadamard_transform(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    r"""Unnormalized Walsh-Hadamard transform along the last axis.

    Computes $H_d x$ for the Sylvester-ordered Hadamard matrix
    $H_{2m} = \begin{pmatrix} H_m & H_m \\ H_m & -H_m \end{pmatrix}$,
    $H_1 = 1$, with $\log_2 d$ butterfly passes: $O(d \log d)$ work, no
    $d \times d$ matrix. Applying it twice returns $d\,x$.

    Args:
        x: Array whose last axis has a power-of-two length ``d``. Leading axes
            are batch axes.

    Returns:
        $H_d x$, same shape as ``x``.

    Raises:
        ValueError: If the last axis is not a power of two.

    Examples:
        >>> import jax.numpy as jnp
        >>> from kernellib import hadamard_transform
        >>> hadamard_transform(jnp.array([1.0, 0.0, 0.0, 0.0])).tolist()
        [1.0, 1.0, 1.0, 1.0]
        >>> hadamard_transform(jnp.array([1.0, 2.0])).tolist()
        [3.0, -1.0]
    """
    d = x.shape[-1]
    if not _is_power_of_two(d):
        raise ValueError(f"hadamard_transform needs a power-of-two last axis, got {d}.")
    h = 1
    while h < d:
        # Pair entry i with entry i + h inside each block of 2h.
        y = rearrange(x, "... (m two h) -> ... m two h", two=2, h=h)
        a, b = y[..., 0, :], y[..., 1, :]
        x = rearrange(
            jnp.stack([a + b, a - b], axis=-2), "... m two h -> ... (m two h)"
        )
        h *= 2
    return x


class FastFoodParams(eqx.Module):
    r"""The random structure of a FastFood feature map.

    Holds one set of diagonals and one permutation per stack of $d$
    frequencies, plus the lengthscale. Draw it with `fastfood_params`.

    Attributes:
        B: Random signs, shape ``(n_stacks, d_padded)``.
        G: Standard Gaussian diagonals, shape ``(n_stacks, d_padded)``.
        P: Permutation indices, shape ``(n_stacks, d_padded)``.
        S: Row lengths, $\chi_{d}$-distributed, shape ``(n_stacks, d_padded)``.
        lengthscale: Scalar or ``(d,)`` (ARD) RBF lengthscale. A
            differentiable leaf.
        d: Input dimension.
        n_components: Number of frequencies; the feature map has
            ``2 * n_components`` columns.
    """

    B: Float[Array, "n_stacks d_padded"]
    G: Float[Array, "n_stacks d_padded"]
    P: Int[Array, "n_stacks d_padded"]
    S: Float[Array, "n_stacks d_padded"]
    lengthscale: Float[Array, ""] | Float[Array, " d"] = eqx.field(
        converter=jnp.asarray
    )
    d: int = eqx.field(static=True)
    n_components: int = eqx.field(static=True)

    @property
    def d_padded(self) -> int:
        """Input dimension rounded up to a power of two."""
        return self.B.shape[-1]

    @property
    def n_stacks(self) -> int:
        """Number of independent ``d_padded x d_padded`` blocks."""
        return self.B.shape[0]


def fastfood_params(
    d: int,
    n_components: int,
    lengthscale: float | Float[Array, ""] | Float[Array, " d"],
    key: jax.Array,
) -> FastFoodParams:
    r"""Draw the random structure for a FastFood RBF feature map.

    ``n_components`` frequencies are built from
    ``ceil(n_components / d_padded)`` independent stacks, each with fresh
    $B$, $G$, $\Pi$ and $S$; surplus frequencies from the last stack are
    dropped.

    Args:
        d: Input dimension.
        n_components: Number of frequencies. The feature map returned by
            `fastfood_features` has ``2 * n_components`` columns (a cosine
            and a sine per frequency).
        lengthscale: RBF lengthscale, scalar or ``(d,)`` for ARD.
        key: PRNG key.

    Returns:
        The FastFood parameters.

    Raises:
        ValueError: If ``d`` or ``n_components`` is not positive.

    Examples:
        >>> import jax.random as jr
        >>> from kernellib import fastfood_params
        >>> p = fastfood_params(
        ...     d=5, n_components=20, lengthscale=1.0, key=jr.key(0)
        ... )
        >>> p.d_padded, p.n_stacks
        (8, 3)
    """
    if d < 1 or n_components < 1:
        raise ValueError(
            f"d and n_components must be positive, got d={d}, "
            f"n_components={n_components}."
        )
    d_padded = 1 << (d - 1).bit_length()
    n_stacks = math.ceil(n_components / d_padded)
    kb, kg, kp, ks = jr.split(key, 4)
    B = jr.rademacher(kb, (n_stacks, d_padded)).astype(jnp.result_type(float))
    G = jr.normal(kg, (n_stacks, d_padded))
    P = jax.vmap(lambda k: jr.permutation(k, d_padded))(jr.split(kp, n_stacks))
    # chi_{d_padded}: the length of a d_padded-dimensional standard Gaussian.
    S = jnp.sqrt(2.0 * jr.gamma(ks, d_padded / 2.0, (n_stacks, d_padded)))
    return FastFoodParams(
        B=B, G=G, P=P, S=S, lengthscale=lengthscale, d=d, n_components=n_components
    )


def _pad(X: Float[Array, "N d"], params: FastFoodParams) -> Float[Array, "N dp"]:
    if X.shape[-1] != params.d:
        raise ValueError(
            f"FastFood params were drawn for d={params.d}, got inputs with "
            f"{X.shape[-1]} features."
        )
    X = X / params.lengthscale
    return jnp.pad(X, ((0, 0), (0, params.d_padded - params.d)))


def _project(X: Float[Array, "N dp"], params: FastFoodParams) -> Float[Array, "N D"]:
    """``X V^T`` for all stacks, truncated to ``n_components`` columns."""
    dp = params.d_padded

    def one_stack(B, G, P, S):
        y = hadamard_transform(X * B)
        y = y[:, P] * G
        y = hadamard_transform(y)
        # S gives each row the length of a Gaussian vector: rows of H G Pi H B
        # have length sqrt(dp) * ||G||, so divide that out.
        return y * (S / (jnp.sqrt(dp) * jnp.linalg.norm(G)))

    Z = jax.vmap(one_stack)(params.B, params.G, params.P, params.S)
    Z = rearrange(Z, "s n d -> n (s d)")
    return Z[:, : params.n_components]


def fastfood_frequencies(params: FastFoodParams) -> Float[Array, "n_components d"]:
    r"""The dense frequency matrix $V$ that FastFood applies implicitly.

    Row $j$ is the frequency $v_j$ with $\langle v_j, x \rangle$ the $j$-th
    projection of an input. Materializing it costs $O(D d)$, which is what
    FastFood exists to avoid; it is here for inspection and testing, and to
    compare with random Fourier features drawn with the same frequencies.

    Args:
        params: FastFood parameters.

    Returns:
        $V$ with the lengthscale folded in, shape ``(n_components, d)``.

    Examples:
        >>> import jax.random as jr
        >>> from kernellib import fastfood_frequencies, fastfood_params
        >>> p = fastfood_params(d=3, n_components=6, lengthscale=1.0, key=jr.key(0))
        >>> fastfood_frequencies(p).shape
        (6, 3)
    """
    # Projecting the identity gives V with the 1 / lengthscale scaling that
    # _pad applies to real inputs.
    return _project(_pad(jnp.eye(params.d), params), params).T


def fastfood_features(
    X: Float[Array, "N d"], params: FastFoodParams
) -> Float[Array, "N two_D"]:
    r"""FastFood feature map, $\Phi(x) = D^{-1/2}\,[\cos(Vx), \sin(Vx)]$.

    $\Phi(x)^\top \Phi(x') = \frac{1}{D}\sum_j \cos\big(v_j^\top (x - x')\big)$
    approximates the unit-variance RBF kernel
    $\exp(-\|x - x'\|^2 / 2\ell^2)$. Cost per input is $O(D \log d)$.

    Args:
        X: Inputs, shape ``(N, d)``.
        params: FastFood parameters from `fastfood_params`.

    Returns:
        Features, shape ``(N, 2 * n_components)``.

    Raises:
        ValueError: If ``X`` does not have ``params.d`` features.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from kernellib import fastfood_features, fastfood_params
        >>> p = fastfood_params(d=2, n_components=8, lengthscale=1.0, key=jr.key(0))
        >>> Phi = fastfood_features(jnp.zeros((3, 2)), p)
        >>> Phi.shape
        (3, 16)
        >>> bool(jnp.allclose(jnp.sum(Phi**2, axis=1), 1.0))
        True
    """
    Z = _project(_pad(X, params), params)
    scale = 1.0 / jnp.sqrt(params.n_components)
    return scale * jnp.concatenate([jnp.cos(Z), jnp.sin(Z)], axis=-1)


def fastfood_operator(
    X: Float[Array, "N d"], params: FastFoodParams
) -> gx.LowRankUpdate:
    r"""FastFood approximation of the RBF Gram matrix, $K \approx \Phi\Phi^\top$.

    Same contract as `rff_operator`: a zero-base, symmetric PSD
    `gaussx.LowRankUpdate` of rank ``2 * n_components`` that never forms the
    ``N x N`` matrix. Add noise by swapping in a diagonal base.

    Args:
        X: Inputs, shape ``(N, d)``.
        params: FastFood parameters from `fastfood_params`.

    Returns:
        `LowRankUpdate` of shape ``(N, N)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> import jax.random as jr
        >>> from kernellib import fastfood_operator, fastfood_params
        >>> p = fastfood_params(
        ...     d=2, n_components=64, lengthscale=1.0, key=jr.key(0)
        ... )
        >>> K = fastfood_operator(jnp.zeros((4, 2)), p)
        >>> K.as_matrix().shape
        (4, 4)
    """
    Phi = fastfood_features(X, params)
    N = X.shape[0]
    return gx.LowRankUpdate(
        base=lx.DiagonalLinearOperator(jnp.zeros(N, dtype=Phi.dtype)),
        U=Phi,
        d=jnp.ones(Phi.shape[1], dtype=Phi.dtype),
        V=Phi,
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
