import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_kernels


class RandomFourierFeatures(BaseEstimator, TransformerMixin):
    """Random Fourier Features Kernel Matrix Approximation


    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : 3rd - August, 2018
    """

    def __init__(self, n_components=50, gamma=1.0,
                 random_state=None):
        self.gamma = gamma
        # Dimensionality D (number of MonteCarlo samples)
        self.n_components = n_components
        self.random_state = random_state
        self.fitted = False

    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        X = check_array(X, accept_sparse='csr')

        n_features = X.shape[1]

        rng = np.random.RandomState(self.random_state)
        # Generate D iid samples from p(w)
        self.w = np.sqrt(2 * self.gamma) * \
                 np.random.normal(size=(n_features, self.n_components))

        # Generate D iid samples from Uniform(0,2*pi)
        self.u = 2 * np.pi * np.random.rand(self.n_components)
        self.fitted = True
        return self

    def transform(self, X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the feature map Z")
        # Compute feature map Z(x):
        Z = np.sqrt(2 / self.n_components) * \
            np.cos((np.dot(X, self.w) + self.u[np.newaxis, :]))
        return Z

    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the kernel matrix")
        Z = self.transform(X)
        return np.dot(Z, Z.T)


class RandomizedNystrom(BaseEstimator, TransformerMixin):
    """Approximation of a kernel map using a subset of
    training data. Utilizes the randomized svd for the
    kernel decomposition to speed up the computations.


    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : December, 2017
    """
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

    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : December, 2017
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


