import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from kernellib.kernels import rbf_kernel


class KernelRidge(BaseEstimator, RegressorMixin):
    """Kernel ridge regression.
    Kernel ridge regression (KRR) combines ridge regression (linear least
    squares with l2-norm regularization) with the kernel trick. It thus
    learns a linear function in the space induced by the respective kernel and
    the data. For non-linear kernels, this corresponds to a non-linear
    function in the original space.
    The form of the model learned by KRR is identical to support vector
    regression (SVR). However, different loss functions are used: KRR uses
    squared error loss while support vector regression uses epsilon-insensitive
    loss, both combined with l2 regularization. In contrast to SVR, fitting a
    KRR model can be done in closed-form and is typically faster for
    medium-sized datasets. On the other  hand, the learned model is non-sparse
    and thus slower than SVR, which learns a sparse model for epsilon > 0, at
    prediction-time.
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape [n_samples, n_targets]).
    Read more in the :ref:`User Guide <kernel_ridge>`.
    Parameters
    ----------
    alpha : {float, array-like}, shape = [n_targets]
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.  Alpha corresponds to
        ``(2*C)^-1`` in other linear models such as LogisticRegression or
        LinearSVC. If an array is passed, penalties are assumed to be specific
        to the targets. Hence they must correspond in number.
    kernel : string or callable, default="linear"
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.
    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    Attributes
    ----------
    dual_coef_ : array, shape = [n_samples] or [n_samples, n_targets]
        Representation of weight vector(s) in kernel space
    X_fit_ : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training data, which is also required for prediction
    References
    ----------
    * Kevin P. Murphy
      "Machine Learning: A Probabilistic Perspective", The MIT Press
      chapter 14.4.3, pp. 492-493
    See also
    --------
    Ridge
        Linear ridge regression.
    SVR
        Support Vector Regression implemented using libsvm.
    Examples
    --------
    >>> from sklearn.kernel_ridge import KernelRidge
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples)
    >>> X = rng.randn(n_samples, n_features)
    >>> clf = KernelRidge(alpha=1.0)
    >>> clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
    KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='linear',
                kernel_params=None)
    """
    def __init__(self, alpha=0.01, gamma=1.0, scale=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale


    def fit(self, X, y=None):
        """Fit Kernel Ridge regression model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data
        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values
        sample_weight : float or array-like of shape [n_samples]
            Individual weights for each sample, ignored if None is passed.
        Returns
        -------
        self : returns an instance of self.
        """
        # Convert data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"), multi_output=True,
                         y_numeric=True)

        K = rbf_kernel(X, y, gamma=self.gamma, scale=self.scale)
        L = np.linalg.cholesky(K + (self.alpha + 1e-10) * np.eye(X.shape[0]))


        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        self.dual_coef_ = np.linalg.solve(L.T, np.linalg.solve(L, y))
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X

        return self

    def predict(self, X):
        """Predict using the kernel ridge model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.
        Returns
        -------
        C : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self, ["X_fit_", "dual_coef_"])
        K = rbf_kernel(X, self.X_fit_, gamma=self.gamma, scale=self.scale)
        return np.dot(K, self.dual_coef_)
