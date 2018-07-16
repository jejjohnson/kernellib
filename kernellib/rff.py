import numpy as np
from time import time
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, pairwise_kernels
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.utils import check_array, check_random_state
from sklearn.exceptions import NotFittedError
from kernellib.nystrom import KRRNystrom

# TODO - More Complete Documentation
# TODO - Speed Experiment
# TODO - Merge RFF and RBFSampler into 1 class
# TODO - Add other kernels
# TODO - Add not fitted exception (sklearn.exceptions, NotFittedError)
# TODO - Complete RFF Class


class KRRRFF(BaseEstimator, RegressorMixin):
    def __init__(self, sigma_y=1e-3, length_scale=None, kernel='rbf', n_components=100, 
                 projection='cos', random_state=None):
        self.sigma_y = sigma_y
        self.length_scale = length_scale
        self.kernel = kernel
        self.n_components = n_components
        self.projection = projection
        self.random_state = random_state

    def fit(self, X, y=None):


        # check x array
        X = check_array(X)

        # kernel length scale parameter
        if self.length_scale is None:

            # common heuristic for finding the sigma value
            self.length_scale = estimate_sigma(X)

        rnd = check_random_state(self.random_state)
        n_dimensions = X.shape[1]

        # Generate n_component iid samples from p(w)
        # self.rand_mat_ = (1/self.sigma) * rnd.randn(self.n_components, n_dimensions)
        # L = np.exp(1j * X.dot(self.rand_mat_.T))

        # Get the RFF transformation
        self.rff = RFF(n_components=self.n_components, length_scale=self.length_scale,
                       projection=self.projection, random_state=self.random_state)

        L = self.rff.fit_transform(X)

        # Solve the kernel matrix
        rhs = L.T.dot(y)
        lhs = self.sigma_y  * np.eye(L.shape[1]) + L.T.dot(L)
        self.weights_ = _solve_cholesky_kernel(lhs, rhs, self.sigma_y)

        self.x_train = X

        return self

    def predict(self, X):

        # L = np.exp(1j * X.dot(self.rand_mat_.T))
        # return np.real(np.dot(L, self.weights_))

        L = self.rff.transform(X)

        return np.dot(L, self.weights_)

class RFF(BaseEstimator, TransformerMixin):
    """Randomized Fourier Feature Algorithm.


    Resources:
    https://github.com/hichamjanati/srf
    """
    def __init__(self, length_scale=None, n_components=50, random_state=None, projection='exp'):
        self.length_scale = length_scale
        self.n_components = n_components
        self.random_state = random_state
        self.projection = projection

    def fit(self, X, y=None):
        """Generates MonteCarlo Random Samples"""
        n_dims = X.shape[1]

        rnd = check_random_state(self.random_state)

        # Generate n_components iid samples from p(w)
        if self.projection in ['cos', 'cosine']:
            self.W = np.sqrt(self.length_scale) * rnd.normal(size=(self.n_components, n_dims))
            # generate n_components from Uniform (0, 2*pi)
            self.u = 2*np.pi*rnd.rand(self.n_components)
        elif self.projection in ['exp']:
            self.W = (self.length_scale) * rnd.randn(self.n_components, n_dims)

        self.fitted = True

        return self

    def transform(self, X):
        """Transforms the data to the new map space.

        Parameters
        ----------
        X : array, (nsamples x n features)

        Returns
        -------
        Z(X) : array, (n samples x n components)
        """

        # check if fitted
        if not self.fitted:
            raise NotFittedError('Need to be fitted before computing the feature map.')

        # Compute feature map Z(X):
        if self.projection in ['cos', 'cosine']:
            Z = np.sqrt(2 / self.n_components) * np.cos((X.dot(self.W.T) + self.u[np.newaxis, :]))
        elif self.projection in ['exp']:
            Z = np.real(np.exp(1j * np.dot(X, self.W.T)))
        return Z

    def compute_kernel(self, X):

        if not self.fitted:
            raise NotFittedError('Need to be fitted before computing the kernel.')

        Z = self.transform(X)

        return np.dot(Z, Z.T)


class KRRRBFSampler(BaseEstimator, RegressorMixin):
    def __init__(self, lam=1e-3, kernel='rbf', sigma=None, n_components=100,
                 random_state=None):
        self.lam = lam
        self.kernel = kernel
        self.sigma = sigma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):


        # check x array
        X = check_array(X)

        # kernel length scale parameter
        if self.sigma is None:

            # common heuristic for finding the sigma value
            self.sigma = np.mean(pdist(X, metric='euclidean'))

        # gamma parameter for the kernel matrix
        self.gamma = 1 / (2 * self.sigma ** 2)

        rnd = check_random_state(self.random_state)

        # perform Nystrom method
        rbf_sampler = RBFSampler(n_components=self.n_components,
                                 gamma=self.gamma,
                                 random_state=self.random_state)

        L = rbf_sampler.fit_transform(X)

        # Solve the kernel matrix
        rhs = L.T.dot(y)
        lhs = self.lam  * np.eye(L.shape[1]) + L.T.dot(L)
        self.weights_ = y - L.dot(_solve_cholesky_kernel(lhs, rhs, self.lam))
        self.weights_ /= self.lam

        self.X_fit_ = X

        return self

    def predict(self, X):

        K = pairwise_kernels(X, self.X_fit_,
                             metric=self.kernel,
                             gamma=self.gamma)

        return np.dot(K, self.weights_)


def generate_data(n_train_samples=1e4, n_test_samples=1e4, random_state=None):

    rnd = check_random_state(random_state)

    x_train = rnd.randn(int(n_train_samples))
    y_train = np.sin(x_train) * 0.1 * rnd.randn(int(n_train_samples))

    x_test = rnd.randn(int(n_test_samples))
    y_test = np.sin(x_test) * 0.1 * rnd.randn(int(n_test_samples))

    x_train, x_test = x_train[:, np.newaxis], x_test[:, np.newaxis]
    y_train, y_test = y_train[:, np.newaxis], y_test[:, np.newaxis]

    return x_train, x_test, y_train, y_test


def main():

    print('Starting Demo...')
    random_state = 123      # reproducibility

    x_train, x_test, y_train, y_test = generate_data(random_state=random_state)

    # Experimental Parameters
    n_components = 100          # number of sample components to keep
    k_rank = 50                 # rank of the matrix for rsvd
    sigma_y = 1e-3                  # regularization parameter
    kernel = 'rbf'              # rbf kernel matrix
    length_scale = np.mean(pdist(x_train, metric='euclidean'))
    gamma = 1 / (length_scale**2)  # length scale for rbf kernel

    # -----------------------------
    # KRR RRF Approximation
    # -----------------------------
    print('\nRunning KRR with Random Fourier Features Approximation ...\n')
    t0 = time()

    krr_rff = KRRRFF(length_scale=length_scale, sigma_y=sigma_y,
                     n_components=n_components,
                     random_state=random_state,
                     projection='cos')

    krr_rff.fit(x_train, y_train)

    y_pred = krr_rff.predict(x_test)

    t1_rff = time() - t0
    print('RFF (time): {:.4f} secs'.format(t1_rff))

    error_nystrom = mean_squared_error(y_pred.squeeze(), y_test.squeeze())
    print('RFF (MSE): {:5f}'.format(error_nystrom))

    # -----------------------------
    # RBF Sampler Approximation
    # -----------------------------
    print('\nRunning KRR with RBF Sampler Approximation ...\n')
    t0 = time()

    krr_nystrom = KRRRBFSampler(lam=sigma_y, kernel=kernel, sigma=length_scale,
                             n_components=2000,
                             random_state=random_state)

    krr_nystrom.fit(x_train, y_train)

    y_pred = krr_nystrom.predict(x_test)

    t1_rbf = time() - t0
    print('RBF Sampler (time): {:.4f} secs'.format(t1_rbf))

    error_rbf = mean_squared_error(y_pred.squeeze(), y_test.squeeze())
    print('RBF Sampler (MSE): {:5f}'.format(error_rbf))

    # -----------------------------
    # Nystrom Approximation
    # -----------------------------
    print('\nRunning KRR with Nystrom Approximation ...\n')
    t0 = time()

    krr_nystrom = KRRNystrom(lam=sigma_y, kernel=kernel, sigma=length_scale,
                             n_components=n_components, k_rank=k_rank,
                             random_state=random_state)

    krr_nystrom.fit(x_train, y_train)

    y_pred = krr_nystrom.predict(x_test)

    t1_nystrom = time() - t0
    print('Nystrom (time): {:.4f} secs'.format(t1_nystrom))

    error_nystrom = mean_squared_error(y_pred.squeeze(), y_test.squeeze())
    print('Nystrom (MSE): {:5f}'.format(error_nystrom))

    # -----------------------------
    # Sklearn KRR
    # -----------------------------
    print('\nRunning KRR without Approximation ...\n')
    t0 = time()

    krr_model = KernelRidge(alpha=sigma_y, kernel=kernel, gamma=gamma)
    krr_model.fit(x_train, y_train)

    y_pred = krr_model.predict(x_test)

    t1_normal = time() - t0
    print('Sklearn KRR (Time): {:2f} secs'.format(t1_normal))

    error_normal = mean_squared_error(y_pred.squeeze(),
                                    y_test.squeeze())
    print('Sklearn KRR (MSE): {:5f}'.format(error_normal))

    print('\nSpeedup (RFF): x{:.2f}'.format(t1_normal / t1_rff))
    print('Speedup (Nystrom): x{:.2f}'.format(t1_normal / t1_nystrom))
    print('Speedup (RBF Sampler): x{:.2f}\n'.format(t1_normal / t1_rbf))

    return None


if __name__ == "__main__":

    main()
