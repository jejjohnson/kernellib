import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from kernellib.kernels import rbf_kernel
import numba
from numba import float64

class KernelRidge(BaseEstimator, RegressorMixin):
    """Kernel ridge regression.
    My custom kernel ridge regression algorithm. It strictly utilizes the 
    RBF kernel 
    Parameters
    ----------
    alpha : {float, array-like}, shape = [n_targets]
        Small positive values of alpha improve the conditioning of the problem
        and reduce the variance of the estimates.  Alpha corresponds to
        ``(2*C)^-1`` in other linear models such as LogisticRegression or
        LinearSVC. If an array is passed, penalties are assumed to be specific
        to the targets. Hence they must correspond in number.
    length_scale : float, default=None
        length_scale parameter for the RBF kernel.
    signal_variance : float, default=1.0
        signal_variance parameter for the RBF kernel.

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
    def __init__(self, alpha=1, length_scale=1.0, signal_variance=1.0):
        self.alpha = alpha
        self.length_scale = length_scale
        self.signal_variance = signal_variance

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

        K = rbf_kernel(X, length_scale=self.length_scale, signal_variance=self.signal_variance)
        alpha = np.atleast_1d(self.alpha)

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        self.dual_coef_ = _solve_cholesky_kernel(K, y, alpha)
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
        K = rbf_kernel(X, self.X_fit_, length_scale=self.length_scale, 
                       signal_variance=self.signal_variance)
        return np.dot(K, self.dual_coef_)

    def derivative(self, X):
        
        X = check_array(X)

        K = rbf_kernel(X, self.X_fit_, length_scale=self.length_scale, 
                       signal_variance=self.signal_variance)
        
        return self.rbf_derivative(self.X_fit_, X, K, self.dual_coef_, self.length_scale)

    def sensitivity(self, x_test, sample='point', method='squared'):
        derivative = self.derivative(x_test)

        # Define the method of stopping term cancellations
        if method == 'squared':
            derivative **= 2
        else:
            np.abs(derivative, derivative)

        # Point Sensitivity or Dimension Sensitivity
        if sample == 'dim':
            return np.mean(derivative, axis=0)
        elif sample == 'point':
            return np.mean(derivative, axis=1)
        else:
            raise ValueError('Unrecognized sample type.')

    @staticmethod
    @numba.njit('float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64)',fastmath=True, nogil=True)
    def rbf_derivative(x_train, x_function, K, weights, length_scale):
        #     # check the sizes of x_train and x_test
        #     err_msg = "xtrain and xtest d dimensions are not equivalent."
        #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

        #     # check the n_samples for x_train and weights are equal
        #     err_msg = "Number of training samples for xtrain and weights are not equal."
        #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

        n_test, n_dims = x_function.shape

        derivative = np.zeros(shape=x_function.shape)

        for itest in range(n_test):
            derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T,
                                          (np.expand_dims(K[itest, :], axis=1) * weights))

        derivative *= - 1 / length_scale**2

        return derivative 

