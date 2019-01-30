"""
Author  : Juan Emmanuel Johnson
Date    : 25th February, 2018
Email   : emanjohnson91@gmail.com
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from utils.graph import (adjacency, Adjacency, create_laplacian)
from utils.embeddings import graph_embedding


class LaplacianEigenmaps(BaseEstimator):
    """Scikit-Learn compatible class for Laplacian Eigenmaps.

    This algorithm implements the Laplacian Eigenmaps algorithm 

    Parameters
    ----------
    n_components : int, (default = 2)
        number of coordinates for the learned embedding
    
    constraint: str, (default='degree')
        the constraint matrix used  
        ['degree', 'identity']
    
    n_neighbors : int, (default=10)
        number of neighbors for constructing the adjacency matrix
    
    adjacency_kwargs : dict, (default=None)
        dictionary of kwargs for the adjacency matrix construction
        see 'graph.py' for more details

    decomposition_kwargs : dict, (default=None)
        dictionary of kwargs used to solve for the eigenvalues

    """
    def __init__(self, n_components=2, n_neighbors=10,
        constraint='degree', adjacency_kwargs=None, neighbors_kwargs=None,
        eigensolver_kwargs=None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.constraint = constraint
        self.adjacency_kwargs = adjacency_kwargs
        self.neighbors_kwargs = neighbors_kwargs
        self.eigensolver_kwargs = eigensolver_kwargs
    
    def fit(self, X, y=None):
        # Check the X array
        X = check_array(X)

        if self.adjacency_kwargs is None:
            self.adjacency_kwargs = {
                'algorithm': 'brute',
                'metric': 'euclidean',
                'mode': 'distance',
                'method': 'knn',
                'weight': 'heat',
                'gamma': 1.0/X.shape[0]
            }

        if self.eigensolver_kwargs is None:
            self.eigensolver_kwargs = {

            }
        
        GetAdjacency = Adjacency(n_neighbors=self.n_neighbors,
                                 **self.adjacency_kwargs)
        
        # Compute the adjacency matrix for X
        adjacency_matrix = GetAdjacency.create_adjacency(X)

        # Compute graph embedding
        self.eigenvalues, self.embedding_ = \
            graph_embedding(
                adjacency_matrix,
                n_components=self.n_components,
                operator = 'lap',
                constraint=self.constraint
            )

        return self

    def fit_transform(self, X, y=None):

        X = check_array(X)

        self.fit(X)

        return self.embedding_


def swiss_roll_test():

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    from time import time

    from sklearn import manifold, datasets
    from sklearn.manifold import SpectralEmbedding

    n_points = 1000
    X, color = datasets.samples_generator.make_s_curve(n_points,
                                                       random_state=0)
    n_neighbors=20
    n_components=2

    # scikit-learn le algorithm
    t0 = time()
    ml_model = SpectralEmbedding(n_neighbors=n_neighbors,
                                 n_components=n_components)
    Y = ml_model.fit_transform(X)
    t1 = time()

    # 2d projection
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,10))
    ax[0].scatter(Y[:,0], Y[:,1], c=color, label='scikit')
    ax[0].set_title('Sklearn-LE: {t:.2g}'.format(t=t1-t0))


    # MY Laplacian Eigenmaps Algorithm

    t0 = time()
    ml_model = LaplacianEigenmaps(n_components=n_components,
                                  n_neighbors=n_neighbors,
                                  constraint='degree')
    Y = ml_model.fit_transform(X)
    t1 = time()

    ax[1].scatter(Y[:,0], Y[:,1], c=color, label='My Algorithm')
    ax[1].set_title('My Algorithm: {t:.2g}'.format(t=t1-t0))

    plt.show()

    return None

if __name__ == "__main__":
    swiss_roll_test()