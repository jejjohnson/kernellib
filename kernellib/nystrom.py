import numpy as np
from time import time
from scipy.spatial.distance import pdist
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import Nystroem
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from sklearn.metrics import mean_squared_error, pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from kernellib.utils import estimate_sigma

# TODO - take care of other kernel methods
# TODO - fix bug, see notebook
# TODO - rewrite solve cholesky kernel function to remove protected member status


class KRRNystrom(BaseEstimator, RegressorMixin):
    def __init__(self, lam=1e-3, kernel='rbf', sigma=None, n_components=100,
                 svd='randomized', k_rank=1, random_state=None):
        self.lam = lam
        self.kernel = kernel
        self.sigma = sigma
        self.n_components = n_components
        self.svd = svd
        self.k_rank = k_rank
        self.random_state = random_state

    def fit(self, X, y=None):


        # check x array
        X = check_array(X)

        # kernel length scale parameter
        if self.sigma is None:

            # common heuristic for finding the sigma value
            self.sigma = estimate_sigma(X, method='mean')

        # gamma parameter for the kernel matrix
        self.gamma = 1 / (2 * self.sigma ** 2)

        rnd = check_random_state(self.random_state)

        # perform Nystrom method
        if self.svd is 'randomized':
            nystrom = RandomizedNystrom(kernel=self.kernel, n_components=self.n_components,
                            sigma=self.sigma, random_state=self.random_state,
                            k_rank=self.k_rank)

            L = nystrom.fit_transform(X)
        elif self.svd is 'arpack':
            nystrom = Nystroem(n_components=self.n_components, kernel=self.kernel,
                               gamma=self.gamma, random_state=self.random_state)

            L = nystrom.fit_transform(X)

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


class RandomizedNystrom(BaseEstimator, TransformerMixin):
    """Approximation of a kernel map using a subset of
    training data. Utilizes the randomized svd for the
    kernel decomposition to speed up the computations."""
    def __init__(self, kernel='rbf', sigma=1.0, n_components=100,
                 k_rank=1, random_state=None):
        self.kernel = kernel
        self.sigma = sigma
        self.n_components = n_components
        self.k_rank = k_rank
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit estimator to the data"""
        X = check_array(X)
        rnd = check_random_state(self.random_state)

        # gamma parameter for the kernel matrix
        self.gamma = 1 / (2 * self.sigma ** 2)

        n_samples = X.shape[0]
        if self.n_components > n_samples:
            n_components = n_samples
        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)

        indices = rnd.permutation(n_samples)
        basis_indices = indices[:n_components]
        basis = X[basis_indices]

        basis_kernel = pairwise_kernels(basis, metric=self.kernel,
                                        gamma=self.gamma)

        # Randomized SVD
        U, S, V = randomized_svd(basis_kernel, self.k_rank, self.random_state)

        S = np.maximum(S, 1e-12)

        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = indices

        return self

    def transform(self, X):
        """Apply the feature map to X."""
        X = check_array(X)

        embedded = pairwise_kernels(X, self.components_,
                                    metric=self.kernel,
                                    gamma=self.gamma)

        return np.dot(embedded, self.normalization_.T)

    def compute_kernel(self, X):

        L = self.transform(X)

        return np.dot(L, L.T)


def nystrom_kernel(K, n_col_indices, n_components=None,
                   random_state=None,
                   svd='randomized'):
    """The nystrom approximation for a kernel matrix.

    Parameters
    ----------

    K : array, (n x n)
        The kernel matrix to perform the nystrom
        approximation

    n_col_indices : int,
        The number of column indices to be used.

    n_components : int,
        The number of k-components to be extracted from
        the svd.

    random_state : int, default = None
        for reproducibility

    svd : string, {'randomized', 'arpack'}
        (default = 'randomized)

        The svd method to use for find the k components

    Returns
    -------
    U, D, V :
        The number of components

    """

    n_samples = K.shape[0]

    if n_components is None:
        n_components = n_samples

    # -------------
    # Sampling
    # -------------
    generator = check_random_state(random_state)
    random_indices = generator.permutation(n_samples)

    # choose 200 samples
    column_indices = random_indices[:n_col_indices]

    # choose the columns randomly from the matrix
    C = K[:, column_indices]

    # get the other sampled columns
    W = C[column_indices, :]

    # Perform SVD
    if svd in ['randomized']:
        U, D, V = randomized_svd(W, n_components=n_components,
                                 random_state=random_state)

        U_approx = np.sqrt(n_col_indices / n_samples) * C.dot(U)
        D_approx = (n_samples / n_col_indices) * np.diag(D**(-1))

    elif svd in ['arpack']:

        U, D, V = np.linalg.svd(W, full_matrices=False)

        U = U[:, :n_components]
        V = V[:, :n_components]
        D = D[:n_components]

        U_approx = np.sqrt(n_col_indices / n_samples) * C.dot(U).dot(np.diag(D**(-1)))
        D_approx = (n_samples / n_col_indices) * np.diag(D)

    else:
        raise ValueError('Unrecognized svd function.')


    W_approx = U.dot(np.diag(D)).dot(U.T)

    return U_approx, D_approx, W_approx, C


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
    lam = 1e-3                  # regularization parameter
    kernel = 'rbf'              # rbf kernel matrix
    sigma = np.mean(pdist(x_train, metric='euclidean'))
    gamma = 1 / (2 * sigma**2)  # length scale for rbf kernel

    # -----------------------------
    # Nystrom Approximation
    # -----------------------------
    print('\nRunning KRR with Nystrom Approximation ...\n')
    t0 = time()

    krr_nystrom = KRRNystrom(lam=lam, kernel=kernel, sigma=sigma,
                             n_components=n_components, k_rank=k_rank,
                             random_state=random_state)

    krr_nystrom.fit(x_train, y_train)

    y_pred = krr_nystrom.predict(x_test)

    t1_nystrom = time() - t0
    print('Nystrom (time): {:.2f} secs'.format(t1_nystrom))

    error_nystrom = mean_squared_error(y_pred.squeeze(), y_test.squeeze())
    print('Nystrom (MSE): {:5f}'.format(error_nystrom))

    # -----------------------------------
    # Nystrom Approximation (Randomized)
    # -----------------------------------
    print('\nRunning KRR with Randomized Nystrom Approximation ...\n')
    t0 = time()

    krr_rnystrom = KRRNystrom(lam=lam, kernel=kernel, sigma=sigma,
                             n_components=n_components, k_rank=k_rank,
                             random_state=random_state, svd='randomized')

    krr_rnystrom.fit(x_train, y_train)

    y_pred = krr_rnystrom.predict(x_test)

    t1_rnystrom = time() - t0
    print('Randomized Nystrom (time): {:.2f} secs'.format(t1_rnystrom))

    error_rnystrom = mean_squared_error(y_pred.squeeze(), y_test.squeeze())
    print('Randomized Nystrom (MSE): {:5f}'.format(error_rnystrom))

    

    # --------------------------------
    # Scikit-Learn KRR Implementation
    # --------------------------------
    print('\nRunning KRR without Approximation ...\n')
    t0 = time()

    krr_model = KernelRidge(alpha=lam, kernel=kernel, gamma=gamma)
    krr_model.fit(x_train, y_train)

    y_pred = krr_model.predict(x_test)

    t1_normal = time() - t0
    print('Sklearn KRR (Time): {:2f} secs'.format(t1_normal))

    error_normal = mean_squared_error(y_pred.squeeze(),
                                      y_test.squeeze())
    print('Sklearn KRR (MSE): {:5f}'.format(error_normal))

    print('\nSpeedup: x{:.2f}\n'.format(t1_normal / t1_nystrom))
    print('\nSpeedup: x{:.2f}\n'.format(t1_normal / t1_rnystrom))

    return None


if __name__ == "__main__":

    main()
