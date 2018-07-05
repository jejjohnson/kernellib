import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances


def rbf_kernel(X, Y=None, length_scale=1.0, scale=1.0):
    """The Radial Basis Function (RBF) Kernel with two
    parameters: the length scale and the signal variance.
    
    Parameters
    ----------
    x : array-like, (n_samples x n_dimensions)
        An array for the left argument for the returned kernel K(X,Y)
        
    y : array-like, optional (n_samples x n_dimensions), default = None
        The right argument for the returned kernel K(X, Y). If none, 
        K(X, X) is evaulated instead.
        
    length_scale : float, default: 1.0
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
    
    X, Y = check_pairwise_arrays(X, Y)
    
    scale_term = - 0.5 / np.power(length_scale, 2)
    
    if Y is None:
        
        dists = pdist(X, metric='sqeuclidean')
        
        K = np.exp(scale_term * dists)
        
        K = squareform(K)
        
        np.fill_diagonal(K, 1)
        
    else:
        
        dists = cdist(X, Y, metric='sqeuclidean')

        K = np.exp(scale_term  * dists)

    return K


def ard_kernel(x, y=None, length_scale=None, scale=None, weighted=None, x_cov=None):
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
    # check if x, y have the same shape
    if y is not None:
        x, y = check_pairwise_arrays(x, y)
    
    # grab samples and dimensions
    n_samples, n_dimensions = x.shape
    
    # get default sigma values
    if length_scale is None:
        length_scale = np.ones(shape=n_dimensions)
        
    else:
        err_msg = 'Number of sigma values do not match number of x dimensions.'
        np.testing.assert_equal(np.shape(length_scale)[0], n_dimensions, err_msg=err_msg)
        
    # get default scale value
    if scale is None:
        scale = 1.0
    
    # compute the euclidean distances
    if y is None:
        dists = pdist(x / length_scale, metric='sqeuclidean')
        
        K = np.exp(-.5 * dists)
        
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)
        
    else:
        dists = cdist(x / length_scale, y / length_scale, metric='sqeuclidean')
        
        # exponentiate the distances
        K = scale * np.exp( -0.5 * dists)
        
    return K

