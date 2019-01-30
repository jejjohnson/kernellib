"""
General and conceptual tests for graph routines

Date Created    : Tuesday, 7th February, 2017
Author          : J. Emmanuel Johnson
Email           : emanjohnson91@gmail.com

Most of the tests here are based off of the sklearn library.
I decided not to reinvent the wheel and just wanted to use what
they already have. I mainly wanted to ensure that my 'wrapper'
routine outputs the same results as the sklearn library.
"""

from nose.tools import assert_equal
import numpy as np
from sklearn.neighbors import NearestNeighbors
from manilearn.utils.graph import adjacency


# adjacency matrix - k-nearest neighbors test
def test_adjacency_k_brute_connect():
    """
    Adjacency Matrix k-Nearest Neighbors test
    """

    # import data
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    # set default parameters
    n_neighbors = 5
    algorithm = 'brute'
    method = 'knn'

    # sklearn adjacency matrix
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm=algorithm).fit(data)
    sklearn_mat = nbrs.kneighbors_graph(data)
    sklearn_mat = sklearn_mat.toarray()

    # my routine
    my_mat = adjacency(data,
                       n_neighbors=n_neighbors,
                       algorithm=algorithm,
                       method=method)
    my_mat = my_mat.toarray()

    # assert adjacency matrices are equal
    msg = 'Distance values comparison.'
    assert_equal(sklearn_mat.all(), my_mat.all(), msg=msg)


def test_adjacency_k_brute_heat():
    """
    Adjacency Matrix k-Nearest Neighbors test
    """

    # import data
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    # set default parameters
    n_neighbors = 5
    algorithm = 'brute'
    method = 'knn'
    weight = 'heat'
    adjacency_kwargs = {'gamma': 1.0}
    gamma = 1.0

    # sklearn adjacency matrix
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm=algorithm).fit(data)
    sklearn_mat = nbrs.kneighbors_graph(data)
    sklearn_mat.data = np.exp(-sklearn_mat.data**2 / gamma**2)
    sklearn_mat = sklearn_mat.toarray()

    # my routine
    my_mat = adjacency(data,
                       n_neighbors=n_neighbors,
                       algorithm=algorithm,
                       method=method,
                       weight=weight,
                       adjacency_kwargs=adjacency_kwargs)
    my_mat = my_mat.toarray()

    # assert adjacency matrices are equal
    msg = 'Distance values comparison.'
    assert_equal(sklearn_mat.all(), my_mat.all(), msg=msg)


def test_adjacency_k_brute_angle():
    """
    Adjacency Matrix k-Nearest Neighbors test
    """

    # import data
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    # set default parameters
    n_neighbors = 5
    algorithm = 'brute'
    method = 'knn'
    weight = 'angle'

    # sklearn adjacency matrix
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                            algorithm=algorithm).fit(data)
    sklearn_mat = nbrs.kneighbors_graph(data)
    sklearn_mat.data = np.exp(-np.arccos(1-sklearn_mat.data))
    sklearn_mat = sklearn_mat.toarray()

    # my routine
    my_mat = adjacency(data,
                       n_neighbors=n_neighbors,
                       algorithm=algorithm,
                       method=method,
                       weight=weight)
    my_mat = my_mat.toarray()

    # assert adjacency matrices are equal
    msg = 'Distance values comparison.'
    assert_equal(sklearn_mat.all(), my_mat.all(), msg=msg)


# adjacency matrix - radius-nearest neighbors test
def test_adjacency_r_brute_connect():
    """
    Adjacency Matrix radius-Nearest Neighbors test
    """

    # import data
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    # set default parameters
    radius = 1.5
    algorithm = 'brute'
    method = 'radius'

    # sklearn adjacency matrix
    nbrs = NearestNeighbors(radius=radius,
                            algorithm=algorithm).fit(data)
    sklearn_mat = nbrs.radius_neighbors_graph(data)
    sklearn_mat = sklearn_mat.toarray()

    # my routine
    my_mat = adjacency(data,
                       radius=radius,
                       algorithm=algorithm,
                       method=method)
    my_mat = my_mat.toarray()

    # assert adjacency matrices are equal
    msg = 'Distance values comparison.'
    assert_equal(sklearn_mat.all(), my_mat.all(), msg=msg)


def test_adjacency_r_brute_heat():
    """
    Adjacency Matrix k-Nearest Neighbors test
    """

    # import data
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    # set default parameters
    radius = 1.5
    algorithm = 'brute'
    method = 'radius'
    weight = 'heat'
    adjacency_kwargs = {'gamma': 1.0}
    gamma = 1.0

    # sklearn adjacency matrix
    nbrs = NearestNeighbors(radius=radius,
                            algorithm=algorithm).fit(data)
    sklearn_mat = nbrs.radius_neighbors_graph(data)
    sklearn_mat.data = np.exp(-sklearn_mat.data**2 / gamma**2)
    sklearn_mat = sklearn_mat.toarray()

    # my routine
    my_mat = adjacency(data,
                       radius=radius,
                       algorithm=algorithm,
                       method=method,
                       weight=weight,
                       adjacency_kwargs=adjacency_kwargs)
    my_mat = my_mat.toarray()

    # assert adjacency matrices are equal
    msg = 'Distance values comparison.'
    assert_equal(sklearn_mat.all(), my_mat.all(), msg=msg)


def test_adjacency_r_brute_angle():
    """
    Adjacency Matrix k-Nearest Neighbors test
    """

    # import data
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    # set default parameters
    radius = 1.5
    algorithm = 'brute'
    method = 'radius'
    weight = 'angle'

    # sklearn adjacency matrix
    nbrs = NearestNeighbors(radius=radius,
                            algorithm=algorithm).fit(data)
    sklearn_mat = nbrs.radius_neighbors_graph(data)
    sklearn_mat.data = np.exp(-np.arccos(1-sklearn_mat.data))
    sklearn_mat = sklearn_mat.toarray()

    # my routine
    my_mat = adjacency(data,
                       radius=radius,
                       algorithm=algorithm,
                       method=method,
                       weight=weight)
    my_mat = my_mat.toarray()

    # assert adjacency matrices are equal
    msg = 'Distance values comparison.'
    assert_equal(sklearn_mat.all(), my_mat.all(), msg=msg)


# TODO: test adjacency
# check parameters:
# * algorithms - annoy, ball_tree, lshf, kd_tree,
# pyflann, cyflann (k, radius)
# * default: nearest_neighbor_kwargs
# * default: adjacency_kwargs

# TODO: test create_adjacency
# check parameters:
# * distances
# * indices
# * weight - heat, angle
# * weight_kwargs

# TODO: test create_constraint
# check parameters:
# * adjacency_matrix
# * constraint - degree, identity, k-scaling
# * laplacian_matrix (for K-Scaling)

# TODO: test create_laplacian
# TODO: maximum
