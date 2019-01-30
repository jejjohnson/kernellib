"""
Created: 4th February, 2017
Author  : J. Emmanuel Johnson
Date    : 4th February, 2017
Email   : emanjohnson91@gmail.com
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import (csr_matrix, spdiags)
from manilearn.utils.graph import (maximum, adjacency, create_laplacian)
from utils.embeddings import graph_embedding


class SchroedingerEigenmaps(BaseEstimator):
    """Scikit-Learn compatible class for Schroedinger Eigenmaps.

    This algorithm is a semisupervised Kernel Eigenmap (Manifold Learning)
    algorithm which seeks to use spatial information, patial labels and label
    propogation within the Laplacian Eigenmaps (spectral embedding)
    algorithm.

    Parameters
    ----------
    n_components : integer, default = 2
        number of coordinates for the manifold embedding

    eigen_solver : string, default = 'lobpcg'
        ['auto' | 'dense' | 'arpack' | 'lobpcg' | 'amg']
        eigenvalue decomposition solver

    operator : string, default = 'ssse'
        ['laplacian' | 'ssse' | 'sepl' | 'sssepl' | 'ssseplp']
        the operator to use for the Schroedinger Eigenmaps algorithm

    constraint : string, default = 'degree'
        ['degree'|'identity'|'similarity'|'dissimilarity']
        the constraint matrix for the eigenvalue decomposition problem

    n_neighbors : integer, default = 10
        number of neighbors for constructing the spectral adjacency matrix

    alpha : float, default = 17.73
        trade-off parameter for the spatial-spectral potential

    beta : float, default = 13
        trade-off parameter for the partial labels potential

    adjacency_kwargs : dict,
        dictionary of key word values for the adjacency matrix construction

    embedding_kwargs : dict,
        dictionary of key word values for the embedding problem construction

    eigensolver_kwargs : dict,
        dictionary of key word arguments for the eigenvalue decomposition solver

    potential_kwargs : dict,
        dictionary of key word arguments for constructing the potential matrix

    References
    ----------
    * Schroedinger Eigenmaps for the Analysis of Bio-Medical Data
        http://arxiv.org/pdf/1102.4086.pdf

    * Schroedinger Eigenmaps w/ Nondiagonal Potentials for Spatial-Spectral
      Clustering of Hyperspectral Imagery
        https://people.rit.edu/ndcsma/pubs/SPIE_May_2014.pdf

    * TODO: references (ncahill code, papers, czaja)
    * TODO: Examples using the script

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    : 4th February, 2017
    Email   : emanjohnson91@gmail.com
    """
    def __init__(self, n_components=2, n_neighbors=10,
                 operator='ssse', constraint='degree', alpha=17.73, beta=13,
                 adjacency_kwargs=None, embedding_kwargs=None,
                 eigensolver_kwargs=None, potential_kwargs=None):
        self.n_components = n_components
        self.eigen_solver = eigen_solver
        self.n_neighbors = n_neighbors
        self.operator = operator
        self.constraint = constraint
        self.alpha = alpha
        self.beta = beta
        self.adjacency_kwargs = adjacency_kwargs
        self.embedding_kwargs = embedding_kwargs
        self.eigensolver_kwargs = eigensolver_kwargs
        self.potential_kwargs = potential_kwargs
        self.eigenvalues = None
        self.embedding_ = None

    def fit(self, X, y=None):
        # check X array
        X = check_array(X)

        # compute weighted adjacency matrix for X
        adjacency_matrix = adjacency(X, n_neighbors=self.n_neighbors,
                                     adjacency_kwargs=self.adjacency_kwargs)

        # create potential matrix
        if self.operator not in ['laplacian']:

            if self.potential_kwargs is None:
                potential_matrix = self.potential_(X)
            else:
                potential_matrix = self.potential_(X, **self.potential_kwargs)
        else:
            potential_matrix = None

        # transfer keywords
        if self.potential_kwargs is None:
            self.potential_kwargs = {'alpha': self.alpha, 'beta': self.beta}

        # compute embedding
        self.eigenvalues, self.embedding_ = \
            graph_embedding(adjacency_matrix, n_components=self.n_components,
                            operator=self.operator, constraint=self.constraint,
                            regularize_mat=potential_matrix,
                            regularize_kwargs=self.potential_kwargs,
                            eigensolver_kwargs=self.eigensolver_kwargs)

        return self

    def fit_transform(self, data):

        # check X array
        data = check_array(data)

        # fit model on X data
        self.fit(data)

        return self.embedding_

    def potential_(self, data, image=None, potential='ssse', spa_neighbors=4,
                   weight='heat', sigma=1.0, eta=1.0):
        """
        Parameters
        ----------
        data : array, shape (N*M samples, D features)
            typically an image vector

        image : array, (N samples, M samples, D features)

        potential : str, default = 'ssse'
            ['ssse'|'pl'|'sssepl'|'ssseplp']

        spa_neighbors : int, default = 4
            spatial neighbors along the image

        weight : str, default = 'heat'
            ['heat'|'angle']
            weight parameters between the spatial and spectral case

        sigma : float, default = 1.0
            spread parameter for the spectral case

        eta : float, default = 1.0
            spread parameter for the spatial case

        Return
        ------
        potential_matrix : array, shape (N*M samples, N*M samples)


        """

        # check for image
        if image is None:
            raise ValueError('Need an image to create a potential matrix.')

        # spatial-spectral potential matriix
        if potential in ['ssse']:

            # get spatial coordinates for dataset
            spatial_coordinates = get_spatial_coordinates(image)

            # find k-nearest spatial neighbors
            nbrs_model = NearestNeighbors(n_neighbors=spa_neighbors).fit(data)
            distances, indices = nbrs_model.kneighbors()

            # spatial-spectral potential
            potential_matrix = spatial_spectral_potential(data, spatial_coordinates,
                                                          indices, weight=weight,
                                                          sigma=sigma, eta=eta)

        elif potential in ['sim']:
            raise NotImplementedError('Similarity potential not available.')

        elif potential in ['dis']:
            raise NotImplementedError('Dissimilarity potential not available.')

        elif potential in ['sepl']:
            raise NotImplementedError('Partial labels potential not available.')

        elif potential in ['sepls']:
            raise NotImplementedError('Labels propagation potential not available.')

        elif potential in ['sssepl']:
            raise NotImplementedError('Spatial-Spectral with Partial labels.')

        elif potential in ['ssseplp']:
            raise NotImplementedError('not available.')

        else:
            raise ValueError('Unrecognized potential matrix argument.')

        return potential_matrix


# extract spatial dimensions of the data
def get_spatial_coordinates(data):
    """Extracts the spatial dimensions of the data (e.g. an image).
    It accepts two types of data inputs: 2D and 3D data.
    - For 2D data, this will simulate x neighbors
    - For 3D data, this will simulate x-y neighbors

    Parameters
    ----------
    data : array, (N x D) or (N x M x D)
        a 2D or 3D dense numpy array

    Returns
    -------
    coordinates : array, (N x D)
        a 2D dense array

    References
    ----------
    Nathan Cahill
    TODO: get link to schroedinger eigenmaps paper

    """
    # get the dimensions of data (3D and 2D case)
    try:
        nrows = data.shape[0]
        ncols = data.shape[1]
        ndims = data.shape[2]

    except:
        nrows = data.shape[0]
        ncols = 1

    # create a meshgrid for the spatial locations
    xv, yv = np.meshgrid(np.arange(0, ncols, 1),
                         np.arange(0, nrows, 1),
                         sparse=False)

    # ravel the data using the FORTRAN order system
    # for the x and y coordinates
    xv = np.ravel(xv, order='F')
    yv = np.ravel(yv, order='F')

    return np.vstack((xv, yv)).T


# Construct the Schroedinger Spatial-Spectral Potential Matrix
def spatial_spectral_potential(data, clusterdata, indices, weight='heat',
                               sigma=1.0, eta=1.0):
    """Constructs the Schroedinger spatial-spectral cluster potential

    Parameters
    ----------
    data : array, MxD
        M is the number of data points and D is the dimension of
        the data.

    clusterdata : array, Mx2
        spatial data array for the clustering points where M is the
        number of data points.

    indices : (M, N) array_like
        an MxN array where M are the number of data points and N
        are the N-1 nearest neighbors connected to that data point M.

    weight : str ['heat'|'angle'] (optional)
        The weight parameter as the kernel for the spatial-spectral
        difference.

    sigma : float, default = 1.0
        The parameter for the heat kernel for the spatial values.
        Default: 1.0

    eta: float, default = 1.0
        The parameter for the heat kernel.
        Default: 1.0

    Returns
    -------
    * Potential Matrix     - a sparse MxM potential matrix

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    :
    Email   : emanjohnson91@gmail.com

    References
    -----------

    Original Author: Nathan D. Cahill

    Code    : https://people.rit.edu/ndcsma/code.html
    Website : https://people.rit.edu/ndcsma/

    Papers
    ------
    N. D. Cahill, W. Czaja, and D. W. Messinger, "Schroedinger Eigenmaps
    with Nondiagonal Potentials for Spatial-Spectral Clustering of
    Hyperspectral Imagery," Proc. SPIE Defense & Security: Algorithms and
    Technologies for Multispectral, Hyperspectral, and Ultraspectral
    Imagery XX, May 2014

    N. D. Cahill, W. Czaja, and D. W. Messinger, "Spatial-Spectral
    Schroedinger Eigenmaps for Dimensionality Reduction and Classification
    of Hyperspectral Imagery," submitted.

    """
    # Number of data points and number of cluster potentials
    N = data.shape[0];
    K = indices.shape[1]-1

    # Compute the weights for the Data Vector EData
    x1 = np.repeat(
            np.transpose(
            data[:, :, np.newaxis], axes=[0, 2, 1]), K, axis=1)

    x2 = data[indices[:,1:]].reshape((N, K, data.shape[1]))

    if weight == 'heat':
        WE = np.exp( - np.sum ( ( x1-x2 )**2, axis=2 ) / sigma**2)

    elif weight == 'angle':
        WE = np.exp( - np.arccos(1-np.sum( (x1-x2), axis=2 ) ) )

    else:
        raise ValueError('Unrecognized Schroedinger Potential weight.')

    # Compute the weights for the Clustering Data Vector CData
    x1 = np.repeat(
            np.transpose(
            clusterdata[:, :, np.newaxis], axes=[0, 2, 1]), K, axis=1)

    x2 = clusterdata[indices[:,1:]].reshape(( N, K, clusterdata.shape[1] ))

    if weight == 'heat':
        WC = np.exp( - np.sum ( ( x1 - x2 )**2, axis=2 ) / eta**2)

    elif weight == 'angle':
        WC = cosine_distances(x1, x2)
        #WC = np.exp( - np.arccos(1-np.sum( (x1-x2), axis=2 ) ) )

    else:
        raise ValueError('Unrecognized SSSE Potential weight.')

    # Create NonDiagonal Elements of Potential matrix, V
    Vrow = np.tile( indices[:, 0], K)
    Vcol = np.ravel( indices[:, 1:], order='F' )
    V_vals = -WE*WC
    Vdata = np.ravel( V_vals, order='F')

    # Create the symmetric Sparse Potential Matrix, V
    V_sparse = csr_matrix( (Vdata, (Vrow, Vcol) ),
                           shape=( N,N ))
    # Make Potential Matrix sparse
    V_sparse = maximum(V_sparse, V_sparse.T)

    # Compute the Diagonal Elements of Potential Matrix, V
    V_diags = spdiags( -V_sparse.sum(axis=1).T, 0, N, N)

    # Return the sparse Potential Matrix V
    return V_diags + V_sparse


# create similarity and dissimilarity potential matrices
# TODO: fix similarity/dissimilarity potential function
def sim_potential(X, potential='sim', norm_lap=None, method='personal',
                  sparse_mat=None):
    """Creates similarity or dissimilarity potential matrix.

    Parameters:
    ----------
    X               - a k list of (n1+n2) x (m) labeled data matrices where
                      n1 is the number of labeled samples and n2 is the
                      number of unlabeled samples. We assume that the
                      labeled entries are positive natural numbers and the
                      unlabeled entries are zero.

    Returns
    -------
    Ws              - a sparse (n1+n2)*k x (n1+n2)*k adjacency matrix
                      showing the connectivity between the corresponding
                      similar labels between the k entries in the list.
    Wd              - a sparse (n1+n2)*k x (n1+n2)*k adjacency matrix
                      showing the connectivity between the corresponding
                      dissimilar labels between the k entries in the list.

    TODO: References
    ----------------

    Tuia et al. - Semisupervised Manifold Alignment of Multimodal Remote
        Sensing Images
        http://isp.uv.es/code/ssma.htm
    Wang - Heterogenous Domain Adaption using Manifold Alignment
        https://goo.gl/QaLTA4

    """
    # create sparse matrix from array X
    new = {}
    new['csr'] = csr_matrix((X.shape[0],X.shape[0]), dtype='int')
    row = []; col = []; datapoint = []

    i = 0
    for (Xrow, Xcol, Xdata) in zip(X.row, X.col, X.data):

        # find all the datapoints in the Y matrix
        # greater than 0
        if Xdata > 0:

            # copy that data point with its row and column entry
            new['csr'][Xrow,:] = Xdata

        i += 1
    # Copy the original matrix transposed
    new['csrt'] = new['csr'].T.copy()

    # convert the matrix to a coordinate matrix
    new['coo'] = new['csr'].tocoo()
    new['coot'] = new['csrt'].tocoo()


    col_labels = ['rows', 'cols', 'data']


    # join the dataframes
    united_data = pd.concat([pd.DataFrame(data=np.array((new['coo'].row,
                                                         new['coo'].col,
                                                         new['coo'].data)).T,
                                          columns=col_labels),
                             pd.DataFrame(data=np.array((new['coot'].row,
                                                         new['coot'].col,
                                                         new['coot'].data)).T,
                                          columns=col_labels)])

    # group the data by the whole row to find the duplicates
    united_data_grouped = united_data.groupby(list(united_data.columns))

    # detect the row indices with similar values
    same_data_idx = [x[0] \
        for x in united_data_grouped.indices.values() if len(x)!=1]

    # extract the unique values
    same_data = united_data.iloc[same_data_idx]

    # create empty sparse matrix
    Ws = csr_matrix((X.shape[0],X.shape[0]), dtype='int')

    # set the unique value indcies to 1
    Ws[same_data['rows'], same_data['cols']] = 1

    if potential in ['sim']:
        return create_laplacian(Ws,
                                norm_lap=norm_lap,
                                method=method,
                                sparse=sparse_mat)

    # group the data by the two columns (row, col)
    united_cols_grouped = united_data.groupby(['rows','cols'])

    # detect the row indices with similar values for the two columns compared with data
    diff_cols_idx = [x[0] for x, z in zip(united_cols_grouped.indices.values(),
                                          united_cols_grouped['data'].indices.values())
                                          if (len(x)!=1 & len(z)!=1)]

    # extract those same values
    diff_data = united_data.iloc[diff_cols_idx]


    # convert back to csr format
    Wd = csr_matrix((X.shape[0],X.shape[0]), dtype='int')
    Wd[diff_data['rows'], diff_data['cols']] = 1

    return create_laplacian(Wd-Ws, norm_lap=norm_lap, method=method,
                            sparse=sparse_mat)


# Determine appropriate trade off parameter between L and D
def potential_tradeoff(L, V, weight=17.78):
    """Gives the suggested value of alpha:

             trace of potential
    weight * ------------------ = suggested weight
             trace of Laplacian

    Parameters
    ----------
    L : array, sparse ( N x N )
        sparse Laplacian matrix

    V : array, sparse ( N x N )
        sparse Schroedinger potential matrix

    weight : float, default = 17.78
        trade-off parameter between the Laplacian matrix and the Schroedinger
        potential matrix

    Returns
    -------
    new_weight : float
        scaled weight parameter to trade off the Laplacian and the Potential matrix

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    : 4th February
    Email   : emanjohnson91@gmail.com
    """
    return weight * (np.trace(L.todense()) / np.trace(V.todense()))
