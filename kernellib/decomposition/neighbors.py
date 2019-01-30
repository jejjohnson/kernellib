"""
Created: Sunday, 5th February, 2017
Author: J. Emmanuel Johnson
Email : emanjohnson91@gmail.com
"""
# TODO: reference all of the nearest neighbor libraries used

import numpy as np
from sklearn.utils import check_array
from sklearn.neighbors import NearestNeighbors, LSHForest
from sklearn.utils.validation import check_random_state
from annoy import AnnoyIndex
from pyflann import FLANN
# import hdidx


class KnnSolver(object):
    """KnnSolver class implements some nearest neighbor algorithms

    Parameters
    ----------
    n_neighbors : int, default = 2
        number of nearest neighbors

    radius : int, default = 1
        length of the radius for the neighbors in distance

    algorithm : str, default = 'annoy'
        ['auto'|'annoy'|'brute'|'kd_tree'|'ball_tree'|'pyflann'|'cyflann']
        algorithm to find the k-nearest or radius-nearest neighbors

    algorithm_kwargs : dict, default = None
        a dictionary of key word values for specific arguments on each algorithm

    References
    ----------
    * sklearn: brute, kd_tree, ball_tree
        https://goo.gl/2noI11
    * annoy
        https://github.com/spotify/annoy
    * nmslib (TODO)
    * pyflann (TODO)
        https://github.com/primetang/pyflann
    * cyflann (TODO)
        https://github.com/dougalsutherland/cyflann

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    : 5th February, 2017
    Email   : emanjohnson91@gmail.com
    """
    def __init__(self, n_neighbors=2, radius=1.5, method='knn', algorithm='brute',
                 random_state=None, algorithm_kwargs=None):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.method = method
        self.algorithm = algorithm
        self.algorithm_kwargs = algorithm_kwargs
        self.random_state = random_state

    def find_knn(self, data):

        # check random state
        self.random_state = check_random_state(self.random_state)

        # check the array
        data = check_array(data)

        # TODO: check kwargs
        self.check_nn_solver_()

        # sklearn (auto, brute, kd_tree, ball_tree)
        if self.algorithm in ['auto', 'brute', 'kd_tree', 'ball_tree']:

            # initialize nearest neighbors model
            nbrs_model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                          radius=self.radius,
                                          algorithm=self.algorithm,
                                          **self.algorithm_kwargs)

            # fit the model to the data
            nbrs_model.fit(data)

            # extract distances and indices
            if self.method in ['knn']:
                distances, indices = \
                    nbrs_model.kneighbors(data, n_neighbors=self.n_neighbors,
                                          return_distance=True)

            elif self.method in ['radius']:
                distances, indices = \
                    nbrs_model.radius_neighbors(data, radius=self.radius,
                                                return_distance=True)

            else:
                raise ValueError('Unrecognized connectivity method.')

            return distances, indices

        elif self.algorithm in ['annoy']:

            if self.algorithm_kwargs is None:
                return ann_annoy(data, n_neighbors=self.n_neighbors)
            else:
                return ann_annoy(data, n_neighbors=self.n_neighbors,
                                 **self.algorithm_kwargs)

        elif self.algorithm in ['pyflann']:
            # TODO: implement pyflann nn method
            raise NotImplementedError('Method has not been completed yet.')

        elif self.algorithm in ['cyflann']:
            # TODO: implement cyflann nn method
            raise NotImplementedError('cyflann has not been completed yet.')

        else:
            raise ValueError('Unrecognized algorithm.')

    def check_nn_solver_(self):

        # check for None type
        if self.algorithm_kwargs is None:
            self.algorithm_kwargs = {}

        # TODO: check sklearn-brute
        # TODO: check sklearn-kd_tree
        # TODO: check sklearn-ball_tree
        # TODO: check sklearn-lshf
        # TODO: check annoy
        # TODO: check pyflann
        # TODO: check cyflann

        return self


def ann_annoy(data, n_neighbors=2, metric='euclidean', trees=10):
    """My approximate nearest neighbor function that uses the ANNOY python
    package

    Parameters
    ----------
    data : array, (N x D)

    n_neighbors : int, default = 2

    metric : str, default = 'euclidean'

    trees : int, default = 10

    Returns
    -------
    distances : array, (N x n_neighbors)

    indices : array, (N x n_neighbors)

    Information
    -----------
    Author  : J. Emmanuel Johnson
    Date    :
    Email   : emanjohnson91@gmail.com
    """
    datapoints = data.shape[0]
    dimension = data.shape[1]

    # initialize the annoy database
    ann = AnnoyIndex(dimension, metric=metric)

    # store the datapoints
    for (i, row) in enumerate(data):
        ann.add_item(i, row.tolist())

    # build the index
    ann.build(trees)

    # find the k-nearest neighbors for all points
    indices = np.zeros((datapoints, n_neighbors), dtype='int')
    distances = indices.copy().astype(np.float)

    # extract the distance values
    for i in range(0, datapoints):
        indices[i, :] = ann.get_nns_by_item(i, n_neighbors)

        for j in range(0, n_neighbors):
            distances[i, j] = ann.get_distance(i, indices[i, j])

    return distances, indices
