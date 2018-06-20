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