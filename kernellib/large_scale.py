import numpy as np
from kernellib.kernel_approximation import RandomizedNystrom, RandomFourierFeatures, FastFood
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.linear_model.ridge import _solve_cholesky_kernel


class RKSKernelRidge(BaseEstimator, RegressorMixin):
    """Random Kitchen Sinks Kernel Approximation.


    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : 3rd - August, 2018
    """
    def __init__(self, n_components=10, alpha=1e-3, sigma=1.0,
                 random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X, y):
        """Fits the Random Kitchen Sinks Kernel Ridge Regression Model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target Values

        sample_weight : float or array-like of shape [n_samples]
            Individual weights for each sample, ignored if None is passed.

        Returns
        -------
        self : returns an instance of self

        """
        # Convert the data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"), multi_output=True,
                         y_numeric=True)

        # iniate randomization
        rng = check_random_state(self.random_state)

        # Generate n_components iid samples (Random Projection Matrix)
        self.w = np.sqrt(1 / (self.sigma**2)) * rng.randn(self.n_components, X.shape[1])

        # Explicitly project the features
        self.L = np.exp(1j * np.dot(X, self.w.T))

        # Calculate the Kernel Matrix
        K = np.dot(self.L.T, self.L) + self.alpha * np.eye(self.n_components)

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True
        #
        # self.dual_coef_ = _solve_cholesky_kernel(K, np.dot(self.L.T, y), alpha)
        #
        # if ravel:
        #     self.dual_coef_ = self.dual_coef_.ravel()
        self.dual_coef_ = np.linalg.solve(K , np.dot(self.L.T, y))

        if ravel:
            self.dual_coef_  = self.dual_coef_.ravel()


        self.X_fit_ = X

        return self


    def predict(self, X, return_real=True):
        """Predict using the RKS Kernel Model


        """
        check_is_fitted(self, ["X_fit_", "dual_coef_"])

        X = check_array(X)

        K = np.exp(1j * np.dot(X, self.w.T))

        if return_real:
            return np.real(np.dot(K, self.dual_coef_))
        else:
            return np.dot(K, self.dual_coef_)


class KernelRidge(BaseEstimator, RegressorMixin):
    """Kernel Ridge Regression with kernel Approximations.


    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : 3rd - August, 2018
    """
    def __init__(self, n_components=10, alpha=1e-3, sigma=None,
                 random_state=None, approximation='nystrom',
                 k_rank=10, kernel='rbf', trade_off='acc'):
        self.n_components = n_components
        self.alpha = alpha
        self.sigma = sigma
        self.random_state = random_state
        self.approximation = approximation
        self.k_rank = k_rank
        self.n_components = n_components
        self.kernel = kernel
        self.trade_off = trade_off

    def fit(self, X, y):

        # Convert the data
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"), multi_output=True,
                         y_numeric=True)

        # iniate randomization
        rng = check_random_state(self.random_state)
        
        # Sigma
        if self.sigma is None:
            self.sigma = 1.0

        # Kernel Approximation Step
        self.L = self._kernel_approximation(X)

        # Solve for weights
        K = np.dot(self.L.T, self.L)
        alpha = np.atleast_1d(self.alpha)

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        if self.approximation == 'rnystrom':
            self.dual_coef_ = solve(K + alpha * np.eye(K.shape[0]), np.dot(self.L.T, y))
        else:
            self.dual_coef_ = _solve_cholesky_kernel(K, np.dot(self.L.T, y), alpha)

        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X


        return self

    def _kernel_approximation(self, X):

        # Random Fourier Features
        if self.approximation == 'rff':
            self.trans = RandomFourierFeatures(
                n_components=self.n_components,
                gamma=1 / np.sqrt(2 * self.sigma**2)
            )

        # RBF Sampler (Variant of Random Kitchen Sinks)
        elif self.approximation == 'rks':
            self.trans = RBFSampler(
                gamma=1 / np.sqrt(2 * self.sigma**2),
                n_components=self.n_components,
                random_state=self.random_state)

        # Nystrom Approximation
        elif self.approximation == 'nystrom':
            self.trans = Nystroem(
                kernel=self.kernel,
                gamma=1 / np.sqrt(2 * self.sigma**2),
                n_components=self.n_components,
                random_state=self.random_state
            )
        # Fast Food Approximation
        elif self.approximation == 'fastfood':
            self.trans = FastFood(
                 sigma=self.sigma,
                 n_components=self.n_components,
                 tradeoff_mem_accuracy=self.trade_off,
                 random_state=self.random_state
            )
        # Randomized Nystrom Approximation
        elif self.approximation == 'rnystrom':
            self.trans = RandomizedNystrom(
                kernel=self.kernel,
                sigma=self.sigma,
                n_components=self.n_components,
                k_rank=self.k_rank,
                random_state=self.random_state
            )
        else:
            raise ValueError('Unrecognized algorithm.')

        self.trans.fit(X)

        return self.trans.transform(X)

    def predict(self, X):
        """Predict using the kernel ridge model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.
        Returns
        -------
        Predictions : array, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self, ["X_fit_", "dual_coef_"])

        X = check_array(X)

        K = self.trans.transform(X)

        return np.real(np.dot(K, self.dual_coef_)) 

