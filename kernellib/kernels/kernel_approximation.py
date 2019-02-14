import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import pairwise_kernels
from scipy.stats import chi
from numba import jit
from ..kernels import estimate_length_scale


class RFF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=100, 
        length_scale=None,
        method='mean', 
        center=None,
        random_state=None):
        self.n_components = n_components 
        self.length_scale = length_scale 
        self.method = method
        self.center = center 
        self.rng = check_random_state(random_state)

    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=True)

        n_features = X.shape[1]

        if self.length_scale is None:
            self.length_scale = estimate_length_scale(
                X, method=self.method, random_state=self.rng)

        # Generate n_components iid samples
        self.W = ( 1 / self.length_scale) * self.rng.randn(n_features, self.n_components)


        return self

    def transform(self, X, return_real=False):

        # Explicitly Project features
        Z = (1 / np.sqrt(self.n_components)) * np.exp(1j * X @ self.W)

        if self.center:
            Z -= np.mean(Z, axis=0)

        if return_real:
            return np.real(Z)
        else:
            return Z

    def compute_kernel(self, X, return_real=False):
        Z = self.transform(X, return_real=False)

        K = np.dot(Z, np.matrix.getH(Z))

        if return_real:
            return np.real(K)
        else:
            return K

class RandomFourierFeatures(BaseEstimator, TransformerMixin):
    """Random Fourier Features Kernel Matrix Approximation


    Author: J. Emmanuel Johnson
    Email : jemanjohnson34@gmail.com
            emanjohnson91@gmail.com
    Date  : 3rd - August, 2018
    """

    def __init__(self, n_components=50, length_scale=None,
                 random_state=None):
        self.length_scale = length_scale
        # Dimensionality D (number of MonteCarlo samples)
        self.n_components = n_components
        self.rng = check_random_state(random_state)
        self.fitted = False

    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        X = check_array(X, ensure_2d=True, accept_sparse='csr')

        n_features = X.shape[1]

        rng = np.random.RandomState(self.random_state)
        # Generate D iid samples from p(w)
        self.w = 1 / np.sqrt(self.length_scale) * \
                 np.random.normal(size=(n_features, self.n_components))

        # Generate D iid samples from Uniform(0,2*pi)
        self.u = 2 * np.pi * rng.rand(self.n_components)
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
                 k_rank=1, random_state=None, **kwargs):
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
        U, S, V = randomized_svd(basis_kernel, self.k_rank, random_state=self.random_state)

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


'''
Implementation of Fastfood (Le, Sarlos, and Smola, ICML 2013).
Primarily by @esc (Valentin Haenel) and felixmaximilian
from https://github.com/scikit-learn/scikit-learn/pull/3665.
Modified by @dougalsutherland.
FHT implementation was "inspired by" https://github.com/nbarbey/fht.
'''


@jit(nopython=True)
def fht(array_):
    """ Pure Python implementation for educational purposes. """
    bit = length = len(array_)
    for _ in range(int(np.log2(length))):
        bit >>= 1
        for i in range(length):
            if i & bit == 0:
                j = i | bit
                temp = array_[i]
                array_[i] += array_[j]
                array_[j] = temp - array_[j]


@jit(nopython=True)
def is_power_of_two(input_integer):
    """ Test if an integer is a power of two. """
    if input_integer == 1:
        return False
    return input_integer != 0 and ((input_integer & (input_integer - 1)) == 0)


@jit(nopython=True)
def fht2(array_):
    """ Two dimensional row-wise FHT. """
    if not is_power_of_two(array_.shape[1]):
        raise ValueError('Length of rows for fht2 must be a power of two')

    for x in range(array_.shape[0]):
        fht(array_[x])


class FastFood(BaseEstimator, TransformerMixin):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.
    Fastfood replaces the random matrix of Random Kitchen Sinks (RBFSampler)
    with an approximation that uses the Walsh-Hadamard transformation to gain
    significant speed and storage advantages.  The computational complexity for
    mapping a single example is O(n_components log d).  The space complexity is
    O(n_components).  Hint: n_components should be a power of two. If this is
    not the case, the next higher number that fulfills this constraint is
    chosen automatically.
    Parameters
    ----------
    sigma : float
        Parameter of RBF kernel: exp(-(1/(2*sigma^2)) * x^2)
    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    tradeoff_mem_accuracy : "accuracy" or "mem", default: 'accuracy'
        mem:        This version is not as accurate as the option "accuracy",
                    but is consuming less memory.
        accuracy:   The final feature space is of dimension 2*n_components,
                    while being more accurate and consuming more memory.
    random_state : {int, RandomState}, optional
        If int, random_state is the seed used by the random number generator;
        if RandomState instance, random_state is the random number generator.
    Notes
    -----
    See "Fastfood | Approximating Kernel Expansions in Loglinear Time" by
    Quoc Le, Tamas Sarl and Alex Smola.
    Examples
    ----
    See scikit-learn-fastfood/examples/plot_digits_classification_fastfood.py
    for an example how to use fastfood with a primal classifier in comparison
    to an usual rbf-kernel with a dual classifier.
    """

    def __init__(self,
                 sigma=np.sqrt(1/2),
                 n_components=100,
                 tradeoff_mem_accuracy='acc',
                 random_state=None):
        self.sigma = sigma
        self.n_components = n_components
        self.random_state = random_state
        self.rng = check_random_state(self.random_state)
        # map to 2*n_components features or to n_components features with less
        # accuracy
        self.tradeoff_mem_accuracy = \
            tradeoff_mem_accuracy

    @staticmethod
    def enforce_dimensionality_constraints(d, n):
        if not is_power_of_two(d):
            # find d that fulfills 2^l
            d = np.power(2, np.floor(np.log2(d)) + 1)
        divisor, remainder = divmod(n, d)
        times_to_stack_v = int(divisor)
        if remainder != 0:
            # output info, that we increase n so that d is a divider of n
            n = (divisor + 1) * d
            times_to_stack_v = int(divisor+1)
        return int(d), int(n), times_to_stack_v

    def pad_with_zeros(self, X):
        try:
            X_padded = np.pad(X,
                              ((0, 0),
                               (0, self.number_of_features_to_pad_with_zeros)),
                              'constant')
        except AttributeError:
            zeros = np.zeros((X.shape[0],
                              self.number_of_features_to_pad_with_zeros))
            X_padded = np.concatenate((X, zeros), axis=1)

        return X_padded

    @staticmethod
    def approx_fourier_transformation_multi_dim(result):
        fht2(result)

    @staticmethod
    def l2norm_along_axis1(X):
        return np.sqrt(np.einsum('ij,ij->i', X, X))

    def uniform_vector(self):
        if self.tradeoff_mem_accuracy != 'acc':
            return self.rng.uniform(0, 2 * np.pi, size=self.n)
        else:
            return None

    def apply_approximate_gaussian_matrix(self, B, G, P, X):
        """ Create mapping of all x_i by applying B, G and P step-wise """
        num_examples = X.shape[0]

        result = np.multiply(B, X.reshape((1, num_examples, 1, self.d)))
        result = result.reshape((num_examples*self.times_to_stack_v, self.d))
        FastFood.approx_fourier_transformation_multi_dim(result)
        result = result.reshape((num_examples, -1))
        np.take(result, P, axis=1, mode='wrap', out=result)
        np.multiply(np.ravel(G), result.reshape(num_examples, self.n),
                    out=result)
        result = result.reshape(num_examples*self.times_to_stack_v, self.d)
        FastFood.approx_fourier_transformation_multi_dim(result)
        return result

    def scale_transformed_data(self, S, VX):
        """ Scale mapped data VX to match kernel(e.g. RBF-Kernel) """
        VX = VX.reshape(-1, self.times_to_stack_v*self.d)

        return (1 / (self.sigma * np.sqrt(self.d)) *
                np.multiply(np.ravel(S), VX))

    def phi(self, X):
        if self.tradeoff_mem_accuracy == 'acc':
            m, n = X.shape
            out = np.empty((m, 2 * n), dtype=X.dtype)
            np.cos(X, out=out[:, :n])
            np.sin(X, out=out[:, n:])
            out /= np.sqrt(X.shape[1])
            return out
        else:
            np.cos(X+self.U, X)
            return X * np.sqrt(2. / X.shape[1])

    def fit(self, X, y=None):
        """Fit the model with X.
        Samples a couple of random based vectors to approximate a Gaussian
        random projection matrix to generate n_components features.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the transformer.
        """
        X = check_array(X)

        d_orig = X.shape[1]

        self.d, self.n, self.times_to_stack_v = \
            FastFood.enforce_dimensionality_constraints(d_orig,
                                                        self.n_components)
        self.number_of_features_to_pad_with_zeros = self.d - d_orig

        self.G = self.rng.normal(size=(self.times_to_stack_v, self.d))
        self.B = self.rng.choice([-1, 1],
                        size=(self.times_to_stack_v, self.d),
                        replace=True)
        self.P = np.hstack([(i*self.d)+self.rng.permutation(self.d)
                            for i in range(self.times_to_stack_v)])
        self.S = np.multiply(1 / self.l2norm_along_axis1(self.G)
                             .reshape((-1, 1)),
                             chi.rvs(self.d,
                                     size=(self.times_to_stack_v, self.d)))

        self.U = self.uniform_vector()

        return self

    def transform(self, X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X = check_array(X)
        X_padded = self.pad_with_zeros(X)
        HGPHBX = self.apply_approximate_gaussian_matrix(self.B,
                                                        self.G,
                                                        self.P,
                                                        X_padded)
        VX = self.scale_transformed_data(self.S, HGPHBX)
        return self.phi(VX)
