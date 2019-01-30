"""
Created: Sunday, 5th February, 2017
License:

Author  : J. Emmanuel Johnson
Email   : emanjohnson91@gmail.com
"""
import numpy as np
from scipy.sparse.linalg import lobpcg, eigsh
from scipy.linalg import eigh
from sklearn.utils import check_array
from sklearn.utils.validation import check_random_state
from pyamg import smoothed_aggregation_solver


class EigSolver(object):
    """Eigenvalue decompostion class


    Parameters
    ----------
    n_components : integer, default = 2
        number of coordinates

    algorithm : str, default = 'arpack'
        ['arpack'|'robust'|'dense'|'pyamg'|'lobpcg'|'rsvd']
        algorithm to perform the eigenvalue decomposition

    Information
    ----------
    Author  : J. Emmanuel Johnson
    Date    : 5th February, 2017
    Email   : emanjohnson91@gmail.com
    """
    def __init__(self, n_components=2, algorithm='arpack',
                 random_state=None, eigen_solver_kwargs=None):
        self.n_components = n_components
        self.algorithm = algorithm
        self.random_state = random_state
        self.eigen_solver_kwargs = eigen_solver_kwargs

    def find_eigenvalues(self, A, B=None):

        if self.algorithm in ['arpack']:
            eigenvalues, eigenvectors = \
                eigenvalues_arpack(A=A, B=B, n_components=self.n_components)

            return eigenvalues, eigenvectors

        elif self.algorithm in ['dense']:
            eigenvalues, eigenvectors = \
                eigenvalues_dense(A=A, B=B, n_components=self.n_components)

        elif self.algorithm in ['pyamg']:
            eigenvalues, eigenvectors = \
                eigenvalues_pyamg(A=A, B=B, n_components=self.n_components)
        else:
            raise ValueError('Unrecognized decomposition algorithm.')

        return eigenvalues, eigenvectors


def eigenvalues_dense(A, B=None, n_components=2):

    return eigh(a=A, b=B, eigvals=(1, n_components), type=1)


def eigenvalues_arpack(A, B=None, n_components=2+1):
    """


    :param A:
    :param B:
    :param n_components:
    :return:

    Notes
    -----
    * there is a bug for a low number of nodes for this solver.
    will calculate more eigenvalues than necessary to
    circumvent.
    """
    if n_components <= 10 and np.shape(A)[0] >= int(2):
        n_components = n_components+15

    # solve the eigenvalue decomposition problem
    eigenvalues, eigenvectors = eigsh(A=A, M=B, k=n_components,
                                      which='SM')

    return eigenvalues[:n_components+1], \
           eigenvectors[:, :n_components+1]

def eigenvalues_pyamg(A, B=None, n_components=2, tol=1E-12,
                      random_state=None):
    """Solves the generalized Eigenvalue problem:
    A x = lambda B x using the multigrid method.
    Works well with very large matrices but there are some
    instabilities sometimes.
    """
    random_state = check_random_state(random_state)
    # convert matrix A and B to float
    A = A.astype(np.float64);

    if B is not None:
        B = B.astype(np.float64)

    # import the solver
    ml = smoothed_aggregation_solver(check_array(A, accept_sparse = ['csr']))

    # preconditioner
    M = ml.aspreconditioner()

    n_nodes = A.shape[0]
    n_find = min(n_nodes, 5 + 2*n_components)
    # initial guess for X
    np.random.RandomState(seed=1234)
    X = random_state.rand(n_nodes, n_find)

    # solve using the lobpcg algorithm
    eigenvalues, eigenvectors = lobpcg(A, X, M=M, B=B,tol=tol,
                                       largest='False')

    sort_order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_order]
    eigenvectors = eigenvectors[:, sort_order]

    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    return eigenvalues, eigenvectors
