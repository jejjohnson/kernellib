"""
General and conceptual tests for nearest neighbors routines

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
from manilearn.utils.neighbors import KnnSolver


def test_sklearn_brute_kdist():
    """ sklearn-brute k-Nearest Neighbors Test
    """
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    # set parameters
    n_neighbors = 2
    algorithm = 'brute'
    method = 'knn'

    # run sklearn k-nearest neighbor routine
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm).fit(X)
    sklearn_dist, sklearn_indx = nbrs.kneighbors(X)

    # run my k-nearest neighbor routine
    nbrs = KnnSolver(n_neighbors=n_neighbors, method=method, algorithm=algorithm)
    my_dist, my_indx = nbrs.find_knn(X)

    # assert distances are equal
    msg = 'Distance values comparison.'
    assert_equal(sklearn_dist.all(), my_dist.all(), msg=msg)

    # assert indices are equal
    msg = 'Indice values comparison.'
    assert_equal(sklearn_indx.all(), my_indx.all(), msg=msg)


def test_sklearn_brute_rdist():
    """sklearn-brute Radius Neighbors Test
    """
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    # set parameters
    algorithm = 'brute'
    radius = 1.5
    method = 'radius'

    # run sklearn routine
    nbrs = NearestNeighbors(radius=radius, algorithm=algorithm).fit(X)
    sklearn_dist, _ = nbrs.radius_neighbors(X)
    sklearn_dist = np.concatenate(list(sklearn_dist))

    # run my routine
    nbrs = KnnSolver(radius=radius, method=method, algorithm=algorithm)
    my_dist, _ = nbrs.find_knn(X)
    my_dist = np.concatenate(list(my_dist))

    # assert equal
    msg = 'distances values are not equal.'
    assert_equal(sklearn_dist.all(), my_dist.all(), msg=msg)

# TODO: test sklearn kd_tree nearest neighbor
# TODO: test sklearn ball_tree nearest neighbor
# TODO: test sklearn lshf nearest neighbor
# TODO: test annoy nearest neighbor
# TODO: test hdidx nearest neighbor
#
# def test_sklearn_kd_tree():
#     pass
#
#
# def test_sklearn_ball_tree():
#     pass
#
#
# def test_sklearn_lshf():
#     pass
#
#
# def test_annoy():
#     pass
#
#
# def test_hdidx():
#     pass
