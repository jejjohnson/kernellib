import numpy as np
import numba
from numba import prange
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances
from sklearn.gaussian_process.kernels import (_check_length_scale)


def ard_kernel(X, Y=None, length_scale=None, scale=1.0):
    """The Automatic Relevance Determination Kernel.

    Parameters
    ----------
    x : array-like, (n_samples x n_dimensions)
        An array for the left argument for the returned kernel K(X,Y)

    y : array-like, optional (n_samples x n_dimensions), default = None
        The right argument for the returned kernel K(X, Y). If none,
        K(X, X) is evaulated instead.

    length_scale : array (n_dimensions), default: 1.0
        The length scale of the kernel.

    scale : float, default: 1.0
        The vertical scale relative to the zero mean of the process in the
        output space.

    Returns
    -------
    K : array, (n_samples x n_samples)
        The kernel matrix for K(X,Y) or K(X,X)

    Information
    -----------
    Author : Juan Emmanuel Johnson

    References
    ----------
    Scikit-Learn (RBF Kernel): https://goo.gl/Sz5icv
    """
    X = np.atleast_1d(X)
    length_scale = _check_length_scale(X, length_scale)

    # compute the euclidean distances
    if Y is None:
        dists = pdist(X / length_scale, metric='sqeuclidean')

        K = np.exp(-.5 * dists)

        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)

    else:
        dists = cdist(X / length_scale, Y / length_scale, metric='sqeuclidean')

        # exponentiate the distances
        K = np.exp(-0.5 * dists)

    return scale * K


def ard_kernel_weighted(x, y=None, x_cov=None, length_scale=None, scale=None):
    
    # check if x, y have the same shape
    if y is not None:
        x, y = check_pairwise_arrays(x, y)
        
    # grab samples and dimensions
    n_samples, n_dimensions = x.shape
    
    # get the default sigma values
    if length_scale is None:
        length_scale = np.ones(shape=n_dimensions)
        
    else:
        length_scale = _check_length_scale(x, length_scale)
        
    # check covariance values
    if x_cov is None:
        x_cov = 0.0
    else:
        x_cov = _check_length_scale(x, x_cov)
        
    # Add dimensions to lengthscale and x_cov
    if np.ndim(length_scale) == 0:
        length_scale = np.array([length_scale])
        
    if np.ndim(x_cov) == 0:
        x_cov = np.array([x_cov])
        
    # get default scale values
    if scale is None:
        scale = 1.0
        

    exp_scale = np.sqrt(x_cov + length_scale**2)
    
    scale_term = np.diag(x_cov * (length_scale**2)**(-1)) + np.eye(N=n_dimensions)
    scale_term = np.linalg.det(scale_term)
    scale_term = scale * np.power(scale_term, -1/2) 
    
    if y is None:
        dists = pdist(x / exp_scale, metric='sqeuclidean')
        
        K = np.exp(- 0.5 * dists)
        
        K = squareform(K)
        
        np.fill_diagonal(K, 1)
        
        K *= scale_term
        
    else:
        
        dists = cdist(x / exp_scale, y / exp_scale, metric='sqeuclidean')
        
        K = np.exp(- 0.5 * dists)

        K *= scale_term
    
    return K


@numba.jit(nopython=True, nogil=True)
def calculate_q_numba(x_train, x_test, K, det_term, exp_scale):
    """Calculates the Q matrix used to compute the variance of the
    inputs with a noise covariance matrix. This uses numba to 
    speed up the calculations.
    
    Parameters
    ----------
    x_train : array, (n_samples x d_dimensions)
        The data used to train the weights.
    
    x_test : array, (d_dimensions)
        A vector of test points.
        
    K : array, (n_samples)
        The portion of the kernel matrix of the training points at 
        test point i, e.g. K = full_kernel_mat[:, i_test]
        
    det_term : float
        The determinant term that's in from of the exponent
        term.
        
    exp_scale : array, (d_dimensions)
        The length_scale that's used within the exponential term.
        
    Returns
    -------
    Q : array, (n_samples x n_samples)
        The Q matrix used to calculate the variance of the samples.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 13 - 06 - 2018
    
    References
    ----------
    McHutchen et al. - Gaussian Process Training with Input Noise
    http://mlg.eng.cam.ac.uk/pub/pdf/MchRas11.pdf
    """
    n_train, d_dimensions = x_train.shape
    
    Q = np.zeros(shape=(n_train, n_train), dtype=np.float64)
    
    # Loop through the row terms
    for iterrow in range(n_train):
        
        # Calculate the row terms
        x_train_row = 0.5 * x_train[iterrow, :]  - x_test
        
        K_row = K[iterrow] * det_term
        
        # Loop through column terms
        for itercol in range(n_train):
            
            # Z Term
            z_term = x_train_row + 0.5 * x_train[itercol, :]
            
            # EXPONENTIAL TERM
            exp_term = np.exp( np.sum( z_term**2 * exp_scale) )
            
            # CONSTANT TERM
            constant_term = K_row * K[itercol] 
            
            # Q Matrix (Corrective Gaussian Kernel)
            Q[iterrow, itercol] = constant_term * exp_term
            
    return Q

@numba.njit(parallel=True)
def calculate_Q(xtrain, xtest, K, det_term, exp_scale):
    
    n_train, d_dimensions = xtrain.shape
    m_test, d_dimensions = xtest.shape
    
    Q = np.zeros(shape=(m_test, n_train, n_train), dtype=np.float64)
    
    # Loop through test points
    for itest in prange(m_test):
        for irow in range(n_train):
            x_train_row = 0.5 * xtrain[irow, :] - xtest[itest, :]
            K_row = K[irow, itest] * det_term
            for icol in range(n_train):
                
                # Calculate Z = .5 (xi - xj)
                z_term = x_train_row + 0.5 * xtrain[icol, :]
                
                # Exponential
                exp_term = np.exp( np.sum(z_term**2 * exp_scale))
                # constant term
                constant_term = K_row * K[icol, itest]
                
                # Q matrix
                Q[itest, irow, icol] = constant_term * exp_term

    
    return Q

def calculate_q(x_train, x_test, K, det_term, exp_scale):
    """Calculates the Q matrix used to compute the variance of the
    inputs with a noise covariance matrix. This is the pure python
    version. ( Note: there is a working Numba version which is 
    significantly faster, a x35 speedup)
    
    Parameters
    ----------
    x_train : array, (n_samples x d_dimensions)
        The data used to train the weights.
    
    x_test : array, (d_dimensions)
        A vector of test points.
        
    K : array, (n_samples)
        The portion of the kernel matrix of the training points at 
        test point i, e.g. K = full_kernel_mat[:, i_test]
        
    det_term : float
        The determinant term that's in from of the exponent
        term.
        
    exp_scale : array, (d_dimensions)
        The length_scale that's used within the exponential term.
        
    Returns
    -------
    Q : array, (n_samples x n_samples)
        The Q matrix used to calculate the variance of the samples.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 13 - 06 - 2018
    
    References
    ----------
    McHutchen et al. - Gaussian Process Training with Input Noise
    http://mlg.eng.cam.ac.uk/pub/pdf/MchRas11.pdf
    """
    n_train, d_dimensions = x_train.shape
    
    Q = np.zeros(shape=(n_train, n_train), dtype=np.float64)
    
    # Loop through the row terms
    for iterrow in range(n_train):
        
        # Calculate the row terms
        x_train_row = 0.5 * x_train[iterrow, :]  - x_test
        
        K_row = K[iterrow] * det_term
        
        # Loop through column terms
        for itercol in range(n_train):
            
            # Z Term
            z_term = x_train_row + 0.5 * x_train[itercol, :]
            
            # EXPONENTIAL TERM
            exp_term = np.exp( np.sum( z_term**2 * exp_scale) )
            
            # CONSTANT TERM
            constant_term = K_row * K[itercol] 
            
            # Q Matrix (Corrective Gaussian Kernel)
            Q[iterrow, itercol] = constant_term * exp_term
            
    return Q


