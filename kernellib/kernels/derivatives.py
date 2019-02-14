import numpy as np
import numba
from numba import float64
from numba import prange
from kernellib.kernels import ard_kernel
from kernellib.kernels import rbf_kernel
from sklearn.metrics import pairwise_kernels
# from kernellib.krr import KernelRidge
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances
from sklearn.gaussian_process.kernels import _check_length_scale
from scipy.linalg import cholesky, cho_solve

# TODO: Write tests for derivative functions, gradients
# TODO: Implement Derivative w/ 1 loop for memory conservation
# TODO: Implement 2nd Derivative for all
# TODO: Do Derivative for other kernel methods (ARD, Polynomial)



def hsic_lin_derivative(X, Y, H, Kx, Ky):

    # ===============
    # X
    # ===============
    n_samples, d_dimensions = X.shape
    factor = 1 / (n_samples - 1)**2
    # initialize params
    derX = np.zeros((n_samples, d_dimensions))
    HKyH = H @ Ky @ H

    return None


def hsic_rbf_derivative(X, Y, H, Kx, Ky, sigma_x, sigma_y):

    # ===============
    # X
    # ===============
    n_samples, d_dimensions = X.shape
    factor = 1 / (n_samples - 1)**2
    # initialize params
    derX = np.zeros((n_samples, d_dimensions))
    HKyH = H @ Ky @ H

    # Loop Through Dimensions & Samples
    for idx in range(d_dimensions):
        for isample in range(n_samples):
            de = ((X[isample, idx] - X[:, idx]) * Kx[:, isample])
            derX[isample, idx] = np.einsum(
                'ij, ij->', HKyH[isample, :][:, None], de[:, None])

    derX *= factor * (-1 / sigma_x**2)
    # ===============
    # Y
    # ===============
    n_samples, d_dimensions = Y.shape
    derY = np.zeros((n_samples, d_dimensions))
    HKxH = H @ Kx @ H
    # Loop Through Dimensions & Samples
    for idx in range(d_dimensions):
        for isample in range(n_samples):
            de = ((Y[isample, idx] - Y[:, idx]) * Ky[:, isample])
            derY[isample, idx] = np.einsum(
                'ij, ij->', HKxH[isample, :][:, None], de[:, None])

    derY *= factor * (-1 / sigma_y**2)

    return derX, derY

def rhsic_rff_derivative(X, Y, H, Wx, Wy, Zx, Zy, sigma_x, sigma_y):

    derX, derY = np.zeros(X.shape), np.zeros(Y.shape)
    Jx = np.zeros((1, derX.shape[1]))

    Zxc = Zx - Zx.mean(axis=1)
    Zyc = Zy - Zy.mean(axis=1)

    BBx = Zyc @ np.matrix.getH(Zyc) @ Zx
    BBy = Zxc @ np.matrix.getH(Zxc) @ Zy

    for idim in range(derX.shape[1]):
        for isample in range(derX.shape[0]):
            Jx[:, idim] = 1
            aux = 1j * Jx @ Wx
            Jx[:, idim] = 0
            derX = self.Zx[isample, :] * aux
            mapX[isample, idim] = np.real(
                BBx[isample, :][None, :] @ np.matrix.getH(derX)).squeeze()
            
    derX *= factor

    for idim in range(derX=Y.shape[1]):
        for isample in range(derY.shape[0]):
            Jy[:, idim] = 1
            aux = 1j * Jy @ Wx
            Jx[:, idim] = 0
            derX = Zx[isample, :] * aux
            mapX[isample, idim] = np.real(
                BBx[isample, :][None, :] @ np.matrix.getH(derX)).squeeze()

    derX *= factor

    return derX, derY

@numba.jit
def numba_hsic_rbf_derivative(X, H, Kx, Ky, sigma):

    # Initialize Parameters
    n_samples, d_dimensions = np.shape(X)
    der = np.zeros((n_samples, d_dimensions))
    HKyH = np.dot(H, np.dot(Ky, H))
    factor = 1 / (n_samples - 1)**2

    for idx in range(d_dimensions):
        for isample in range(n_samples):
            de = ((X[isample, idx] - X[:, idx]) * Kx[:, isample])
            der[isample, idx] = np.trace(np.dot(np.expand_dims(
                HKyH[isample, :], axis=0), np.expand_dims(de, axis=1)))

    der = der * factor * (-1 / sigma**2)

    return der

class RBFDerivative(object):
    def __init__(self, krr_model):
        self.krr_model = krr_model

        self.weights = krr_model.dual_coef_.flatten()
        # print(self.weights.shape)
        # if np.ndim(self.weights) == 1:
        #     self.weights = np.atleast_2d(self.weights).T
        # print(self.weights.shape)
        self.length_scale = krr_model.length_scale
        self.signal_variance = krr_model.signal_variance
        self.x_train = krr_model.X_fit_

    def __call__(self, x, full=False, nder=1):
        K = rbf_kernel(x, self.x_train, length_scale=self.length_scale, signal_variance=self.signal_variance)
        # print(self.x_train.shape, x.shape, K.shape, self.weights.shape, self.length_scale)
        return y_rbf_derivative(self.x_train, x, K, self.weights.flatten(), self.length_scale)
                
    def sensitivity(self, x_test, sample='point', method='squared'):
        derivative = self.__call__(x_test)

        # Define the method of stopping term cancellations
        if method == 'squared':
            derivative **= 2
        else:
            np.abs(derivative, derivative)

        # Point Sensitivity or Dimension Sensitivity
        if sample == 'dim':
            return np.mean(derivative, axis=0)
        elif sample == 'point':
            return np.mean(derivative, axis=1)
        else:
            raise ValueError('Unrecognized sample type.')

    # @staticmethod
    # @numba.njit('float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:],float64)',fastmath=True, nogil=True)
    # def rbf_derivative(x_train, x_function, K, weights, length_scale):
    #     #     # check the sizes of x_train and x_test
    #     #     err_msg = "xtrain and xtest d dimensions are not equivalent."
    #     #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

    #     #     # check the n_samples for x_train and weights are equal
    #     #     err_msg = "Number of training samples for xtrain and weights are not equal."
    #     #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    #     n_test, n_dims = x_function.shape

    #     derivative = np.zeros(shape=x_function.shape)

    #     for itest in range(n_test):
    #         derivative[itest, :] = np.dot((np.expand_dims(x_function[itest, :], axis=0) - x_train).T,
    #                                       (K[itest, :] * weights)).flatten()

    #     derivative *= - 1 / length_scale**2

    #     return derivative

@numba.njit(fastmath=True, nogil=True, debug=True)
def my_rbf_derivative(x_train, x_function, K, weights, length_scale):
    #     # check the sizes of x_train and x_test
    #     err_msg = "xtrain and xtest d dimensions are not equivalent."
    #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

    #     # check the n_samples for x_train and weights are equal
    #     err_msg = "Number of training samples for xtrain and weights are not equal."
    #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    for itest in range(n_test):
        derivative[itest, :] = np.dot((np.expand_dims(x_function[itest, :], axis=0) - x_train).T, 
        (K[itest, :] * weights))


    derivative *= - 1 / length_scale**2

    return derivative


class ARDDerivative(object):
    def __init__(self, gp_model):
        self.gp_model = gp_model
        self.x_train = gp_model.X_train_
        self.n_samples, self.d_dimensions = self.x_train.shape
        self.kernel = gp_model.kernel_

        # check the weights
        if np.ndim(gp_model.alpha_) == 1:
            self.weights = np.atleast_2d(gp_model.alpha_).T
        else:
            self.weights = gp_model.alpha_

        # Check the Length_scale
        # Check the length scale
        length_scale = gp_model.kernel_.get_params()['k1__k2__length_scale']
        self.length_scale = _check_length_scale(self.x_train, length_scale)

        if isinstance(length_scale, float):
            self.length_scale = np.array([self.length_scale])
        if len(self.length_scale) == 1 and len(self.length_scale) != self.x_train.shape[1]:
            self.length_scale = self.length_scale * np.ones(self.x_train.shape[1])
        self.scale = gp_model.kernel_.get_params()['k1__k1__constant_value']
        self.noise = gp_model.kernel_.get_params()['k2__noise_level']


    def __call__(self, X, full=False):
    
        #TODO Check the inputs


        # Calculate the kernel matrix
        K = self.kernel(X, self.x_train)
        # print(self.x_train.shape, X.shape, K.shape, self.weights.shape, self.length_scale.shape)
        return self.ard_derivative_numba(self.x_train, X, K, self.weights, self.length_scale)
                

    def sensitivity(self, x_test, sample='point', method='squared'):

        derivative = self.__call__(x_test)

        # Define the method of stopping term cancellations
        if method == 'squared':
            derivative **= 2
        else:
            np.abs(derivative, derivative)

        # Point Sensitivity or Dimension Sensitivity
        if sample == 'dim':
            return np.mean(derivative, axis=0)
        elif sample == 'point':
            return np.mean(derivative, axis=1)
        else:
            raise ValueError('Unrecognized sample type.')




    @staticmethod
    @numba.njit('float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:])',
                nogil=True, fastmath=True)
    def ard_derivative_numba(x_train, x_function, K, weights, length_scale):
        #     # check the sizes of x_train and x_test
        #     err_msg = "xtrain and xtest d dimensions are not equivalent."
        #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

        #     # check the n_samples for x_train and weights are equal
        #     err_msg = "Number of training samples for xtrain and weights are not equal."
        #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

        n_test, n_dims = x_function.shape

        derivative = np.zeros(shape=x_function.shape)

        length_scale = np.diag(- np.power(length_scale, -2))

        for itest in range(n_test):
            # print( np.expand_dims(x_function[itest, :], axis=0).shape, x_train.shape)
            # print(length_scale.shape, (np.expand_dims(x_function[itest, :], axis=0) - x_train).T.shape)
            # print(np.expand_dims(K[itest, :], axis=1).shape, weights.shape)
            # print(derivative[itest, :].shape)
            derivative[itest, :] = np.dot(np.dot(length_scale, (np.expand_dims(x_function[itest, :], axis=0) - x_train).T),
                                          (np.expand_dims(K[itest, :], axis=1) * weights)).flatten()
            break

        return derivative

def ard_derivative_full(x_train, x_function, K, weights, length_scale):

    n_test, d_dims = x_function.shape
    n_train, d_dims = x_train.shape
    derivative = np.zeros(shape=(n_test, n_train, d_dims))

    constant = np.diag(-np.power(length_scale, -2))

    weights = np.tile(weights, (1, d_dims))

    for itest in range(n_test):
        X = (np.tile(x_function[itest, :], (n_train, 1)) - x_train).dot(constant)

        term3 = np.tile(K[itest, :].T, (1, d_dims)).T
        derivative[itest, :, :] = X * weights * term3

    return derivative

def ard_derivative(x_train, x_function, K, weights, length_scale):

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)
    length_scale = np.diag(- np.power(length_scale, -2))
    for itest in range(n_test):
        derivative[itest, :] = np.dot(length_scale.dot((x_function[itest, :] - x_train).T),
                                        (K[itest, :].reshape(-1, 1) * weights))

    return derivative

    # def sensitivity(self, x_test, method='squared'):

@numba.njit(fastmath=True)
def ard_derivative_full_numba(x_train, x_function, K, weights, length_scale):

    n_test, d_dims = x_function.shape
    n_train, d_dims = x_train.shape

    derivative = np.zeros(shape=(n_test, n_train, d_dims))

    constant = -np.power(length_scale, -2)

    for idim in range(d_dims):
        for itrain in range(n_train):
            for itest in range(n_test):
                derivative[itest, itrain, idim] = \
                    constant[idim] * weights[itrain] \
                    * (x_function[itest, idim] - x_train[itrain, idim]) \
                    * K[itest, itrain]

    return derivative


@numba.njit(parallel=True, fastmath=True)
def ard_derivative_numba(x_train, x_function, K, weights, length_scale):

    #     # check the sizes of x_train and x_test
    #     err_msg = "xtrain and xtest d dimensions are not equivalent."
    #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

    #     # check the n_samples for x_train and weights are equal
    #     err_msg = "Number of training samples for xtrain and weights are not equal."
    #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    length_scale = np.diag(- np.power(length_scale, -2))

    for itest in range(n_test):
        derivative[itest, :] = np.dot(length_scale.dot((x_function[itest, :] - x_train).T),
                                        (K[itest, :].reshape(-1, 1) * weights))

    return derivative

@numba.njit(fastmath=True)
def rbf_derivative_numba(x_train, x_function, K, weights, gamma):
    #     # check the sizes of x_train and x_test
    #     err_msg = "xtrain and xtest d dimensions are not equivalent."
    #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

    #     # check the n_samples for x_train and weights are equal
    #     err_msg = "Number of training samples for xtrain and weights are not equal."
    #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    constant = - 2 * gamma

    for itest in range(n_test):
        derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T,
                                        (K[itest, :].reshape(-1, 1) * weights))

    derivative *= constant

    return derivative

@numba.njit(fastmath=True)
def rbf_derivative_full_numba(x_train, x_function, K, weights, gamma, nder=1):
    n_test, d_dims = x_function.shape
    n_train, d_dims = x_train.shape

    derivative = np.zeros(shape=(n_test, n_train, d_dims))

    if nder == 1:
        for idim in range(d_dims):
            for itrain in range(n_train):
                w = weights[itrain]
                for itest in range(n_test):
                    #                 print(weights.shape)
                    derivative[itest, itrain, idim] = \
                        w * (x_function[itest, idim] - x_train[itrain, idim]) * K[itest, itrain]

        derivative *= - 2 * gamma

    else:
        constant = 2 * gamma
        for idim in range(d_dims):
            for itrain in range(n_train):
                for itest in range(n_test):
                    derivative[itest, itrain, idim] = \
                        weights[itrain] \
                        * (constant * (x_function[itest, idim] - x_train[itrain, idim]) ** 2 - 1) \
                        * K[itest, itrain]
        derivative *= constant

    return derivative


    #     """"""
    #     der = ard_derivative(self.x_train, x_test, self.weights,
    #                          self.length_scale, self.scale)
    #
    #     if method == 'squared':
    #         return np.mean(np.mean(der**2))
    #     else:
    #         return np.mean(np.mean(np.abs(der)))


    # def ard_full_derivative(self, X):
    #
    #     length_scale = np.diag(np.power(self.length_scale**2, -1))
    #
    #
    #     n_samples, d_dimensions = self.x_train.shape
    #     m_samples, d_dimensions = X.shape
    #
    #     # K Matrix
    #     K = self.kernel(self.x_train, X)
    #
    #     weights = self.weights
    #     if np.ndim(weights) == 1:
    #         weights = np.atleast_2d(weights)
    #     elif weights.shape[1] != 1:
    #         weights = weights.T
    #
    #     derivative = self.scale * \
    #                  length_scale.dot(self.x_train.T).dot(np.diag(weights.flatten()).dot(K) - np.diag(weights.dot(K))).T
    #
    #     return derivative


@numba.njit(fastmath=True)
def ard_derivative_full_numba(x_train, x_function, K, weights, length_scale):

    n_test, d_dims = x_function.shape
    n_train, d_dims = x_train.shape

    derivative = np.zeros(shape=(n_test, n_train, d_dims))

    constant = -np.power(length_scale, -2)

    for idim in range(d_dims):
        for itrain in range(n_train):
            for itest in range(n_test):
                derivative[itest, itrain, idim] = \
                    constant[idim] * weights[itrain] \
                    * (x_function[itest, idim] - x_train[itrain, idim]) \
                    * K[itest, itrain]

    return derivative

def ard_derivative(x_train, x_function, K, weights, length_scale):

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)
    length_scale = np.diag(- np.power(length_scale, -2))
    for itest in range(n_test):
        derivative[itest, :] = np.dot(length_scale.dot((x_function[itest, :] - x_train).T),
                                        (K[itest, :].reshape(-1, 1) * weights))

    return derivative


@numba.njit(fastmath=True, nogil=True)
def ard_derivative_numba(x_train, x_function, K, weights, length_scale):
    #     # check the sizes of x_train and x_test
    #     err_msg = "xtrain and xtest d dimensions are not equivalent."
    #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

    #     # check the n_samples for x_train and weights are equal
    #     err_msg = "Number of training samples for xtrain and weights are not equal."
    #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    length_scale = np.diag(- np.power(length_scale, -2))

    for itest in prange(n_test):
        derivative[itest, :] = np.dot(np.dot(length_scale, (x_function[itest, :] - x_train).T),
                                      (K[itest, :].reshape(-1, 1) * weights))

    return derivative

@numba.njit(fastmath=True, nogil=True)
def rbf_derivative_numba(x_train, x_function, K, weights, gamma):
    #     # check the sizes of x_train and x_test
    #     err_msg = "xtrain and xtest d dimensions are not equivalent."
    #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

    #     # check the n_samples for x_train and weights are equal
    #     err_msg = "Number of training samples for xtrain and weights are not equal."
    #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    constant = - 2 * gamma

    for itest in range(n_test):
        derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T,
                                      (K[itest, :].reshape(-1, 1) * weights))

    derivative *= - 1 / length_scale**2

    return derivative


def rbf_derivative_full(xtrain, xtest, K, weights, length_scale):
    """The Derivative of the RBF kernel. It returns the full 
    derivative as a 3D matrix.
    
    Parameters
    ----------
    xtrain : array, (n_train_samples x d_dimensions)
    
    xtest : array, (ntest_samples, d_dimensions)
    
    K : array, (ntest_samples, ntrain_samples)
    
    weights : array, (ntrain_samples)
    
    length_scale : float,
    
    Return
    ------
    
    Derivative : array, (n_test, n_train, d_dimensions)
    
    """
    n_test, d_dims = xtest.shape
    n_train, d_dims = xtrain.shape

    derivative = np.zeros(shape=(n_test, n_train, d_dims))

    weights = np.tile(weights, (1, d_dims))

    for itest in range(n_test):
        term1 = (np.tile(xtest[itest, :], (n_train, 1)) - xtrain)
        term3 = np.tile(K[itest, ].T, (1, d_dims)).T
        derivative[itest, :, :] = term1 * weights * term3


    derivative *= - 1 / (length_scale**2)
    return derivative


def ard_derivative(x_train, x_test, weights, length_scale, scale, n_der=1):
    """Derivative of the GP mean function of the ARD Kernel. This function 
    computes the derivative of the mean function that has been trained with an
    ARD kernel with respect to the testing points.
    
    Parameters
    ----------
    x_train : array-like, (n_train_samples x d_dimensions)
        The training samples used to train the weights and the length scale 
        parameters.
        
    x_test : array-like, (n_test_samples x d_dimensions)
        The test samples that will be used to compute the derivative.
        
    weights : array-like, (n_train_samples, 1)
        The weights used from the training samples
        
    length_scale : array, (d_dimensions)
        The length scale for the ARD kernel. This includes a sigma value
        for each dimension.
    
    n_der : int, default: 1, ('1', '2')
        The nth derivative for the mean GP/KRR function with the ARD kernel
        
    Returns
    -------
    derivative : array-like, (n_test_samples x d_dimensions)
        The computed derivative.
        
    Information
    -----------
    Author : Juan Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    
    References
    ----------
    Differenting GPs:
        http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf
    """
    # check the sizes of x_train and x_test
    err_msg = "xtrain and xtest d dimensions are not equivalent."
    np.testing.assert_equal(x_test.shape[1], x_train.shape[1], err_msg=err_msg)
    
    n_train_samples, d_dimensions = x_train.shape
    n_test_samples = x_test.shape[0]
    length_scale = _check_length_scale(x_train, length_scale)
    
    # Make the length_scale 1 dimensional
    if np.ndim(length_scale) == 0:
        length_scale = np.array([length_scale])
    if np.ndim(weights) == 1:
        weights = weights[:, np.newaxis]

    if len(length_scale) == 1 and d_dimensions > 1:
        length_scale = length_scale * np.ones(shape=d_dimensions)
    elif len(length_scale) != d_dimensions:
        raise ValueError('Incorrect Input for length_scale.')
    
    # check the n_samples for x_train and weights are equal
    err_msg = "Number of training samples for xtrain and weights are not equal."
    np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    if int(n_der) == 1:
        constant_term = np.diag(- np.power(length_scale**2, -1))
    
    else:
        constant_term2 = (1 / length_scale)**2
        constant_term4 = (1 / length_scale)**4
    
    # calculate the ARD Kernel
    kernel_mat = ard_kernel(x_test, x_train, length_scale=length_scale, scale=scale)
    
    # initialize derivative matrix
    derivative = np.zeros(shape=(n_test_samples, d_dimensions))
    if int(n_der) == 1:
        for itest in range(n_test_samples):
            
            x_tilde = (x_test[itest, :] - x_train).T
            
            kernel_term = (kernel_mat[itest, :][:, np.newaxis] * weights)

            derivative[itest, :] = constant_term.dot(x_tilde).dot(kernel_term).squeeze()
            
    else:
        for itest in range(n_test_samples):
            
            x_term = np.dot(constant_term2, np.ones(shape=(d_dimensions,
                                                          n_train_samples)))
            
            x_term += np.dot(constant_term4, (x_test[itest, :] - x_train).T**2)
            
            derivative[itest, :] = np.dot(x_term, kernel_mat[itest, :] * weights).T 
            
    return derivative


@staticmethod
def ard_derivative_full(x_train, x_function, K, weights, length_scale):

    n_test, d_dims = x_function.shape
    n_train, d_dims = x_train.shape
    derivative = np.zeros(shape=(n_test, n_train, d_dims))

    constant = np.diag(-np.power(length_scale, -2))

    weights = np.tile(weights, (1, d_dims))

    for itest in range(n_test):
        X = (np.tile(x_function[itest, :], (n_train, 1)) - x_train).dot(constant)

        term3 = np.tile(K[itest, :].T, (1, d_dims)).T
        derivative[itest, :, :] = X * weights * term3

    return derivative

def rbf_full_derivative(x_train, x_test, weights, gamma):

    if np.ndim(x_test) == 1:
        x_test = x_test[np.newaxis, :]

    if np.ndim(weights) == 1:
        weights = weights[:, np.newaxis]

    n_test, d_dims = x_test.shape
    n_train, d_dimst = x_train.shape

    assert(d_dims == d_dimst)

    full_derivative = np.zeros(shape=(n_test, n_train, d_dims))

    K = pairwise_kernels(x_test, x_train, gamma=gamma, metric='rbf')
    constant = -2 * gamma

    for itest in range(n_test):

        term1 = (np.tile(x_test[itest, :], (n_train, 1)) - x_train)
        term2 = np.tile(weights, (1, d_dims))
        term3 = np.tile(K[itest, :].T, (1, d_dims)).T

        full_derivative[itest, :, :] = term1 * term2 * term3

    full_derivative *= constant

    return full_derivative


def rbf_derivative_full(x_train, x_function, K, weights, length_scale, nder=1):

    n_test, d_dims = x_function.shape
    n_train, d_dims = x_train.shape

    derivative = np.zeros(shape=(n_test, n_train, d_dims))

    weights = np.tile(weights, (1, d_dims))

    if nder == 1:

        constant = - 1 / length_scale**2

        for itest in range(n_test):

            term1 = (np.tile(x_function[itest, :], (n_train, 1)) - x_train)
            term3 = np.tile(K[itest, :].T, (1, d_dims)).T
            derivative[itest, :, :] = term1 * weights * term3

    else:

        constant = 1 / length_scale**2
        for itest in range(n_test):


            term1 = constant * (np.tile(x_function[itest, :], (n_train, 1)) - x_train) ** 2 - 1
            term3 = np.tile(K[itest, :].T, (1, d_dims)).T
            derivative[itest, :, :] = term1 * weights * term3

    derivative *= constant
    return derivative

def rbf_full_derivative_loops(x_train, x_function, weights, gamma):

    n_test, d_dims = x_function.shape
    n_train, d_dims = x_train.shape

    K = pairwise_kernels(x_function, x_train, gamma=gamma)

    full_derivative = np.zeros(shape=(n_test, n_train, d_dims))

    constant = - 2 * gamma

    for itest in range(n_test):
        for itrain in range(n_train):
            for idim in range(d_dims):

                full_derivative[itest, itrain, idim] = \
                    weights[itrain] \
                    * (x_function[itest, idim] - x_train[itrain, idim]) \
                    * K[itest, itrain]

    full_derivative *= constant

    return full_derivative


def y_rbf_derivative(x_train, x_function, K, weights, length_scale):
    """The Derivative of the RBF kernel. It returns the 
    derivative as a 2D matrix.
    
    Parameters
    ----------
    xtrain : array, (n_train_samples x d_dimensions)
    
    xtest : array, (ntest_samples, d_dimensions)
    
    K : array, (ntest_samples, ntrain_samples)
    
    weights : array, (ntrain_samples)
    
    length_scale : float,
    
    Return
    ------
    
    Derivative : array, (n_test,d_dimensions)
    
    """
    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    for itest in range(n_test):
        t1 = (np.expand_dims(x_function[itest, :], axis=0) - x_train).T
        t2 = K[itest, :] * weights.squeeze()
        t3 = np.dot(t1, t2)

        derivative[itest, :] = t3

    derivative *= - 1 / length_scale**2

    return derivative


def rbf_derivative(x_train, x_function, weights, gamma):
    
    # check the sizes of x_train and x_test
    err_msg = "xtrain and xtest d dimensions are not equivalent."
    np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)
    
    # check the n_samples for x_train and weights are equal
    err_msg = "Number of training samples for xtrain and weights are not equal."
    np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    K = pairwise_kernels(x_function, x_train, gamma=gamma, metric='rbf')

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    constant = - 2 * gamma

    for itest in range(n_test):

        if n_dims < 2: 
            derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T, 
                                (K[itest, :][:, np.newaxis] * weights))
            
        else:
            derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T,
                    (K[itest, :] * weights).T)

    derivative *= constant
        
    return derivative


def rbf_derivative_slow(x_train, x_function, weights,
                        n_derivative=1, gamma=1.0):
    """This function calculates the rbf derivative
    Parameters
    ----------
    x_train : array, [N x D]
        The training data used to find the kernel model.

    x_function  : array, [M x D]
        The test points (or vector) to use.

    weights   : array, [N x D]
        The weights found from the kernel model
            y = K * weights

    kernel_mat: array, [N x M], default: None
        The rbf kernel matrix with the similarities between the test
        points and the training points.

    n_derivative : int, (default = 1) {1, 2}
        chooses which nth derivative to calculate

    gamma : float, default: None
        the parameter for the rbf_kernel matrix function

    Returns
    -------

    derivative : array, [M x D]
        returns the derivative with respect to training points used in
        the kernel model and the test points.

    Information
    -----------
    Author: Juan Emmanuel Johnson
    Email : jej2744@rit.edu
            juan.johnson@uv.es
    """

    # initialize rbf kernel
    derivative = np.zeros(np.shape(x_function))

    # check for kernel mat
    K = pairwise_kernels(x_function, x_train, gamma=gamma)

    # consolidate the parameters
    theta = 2 * gamma

    # 1st derivative
    if n_derivative == 1:

        # loop through dimensions
        for dim in np.arange(0, np.shape(x_function)[1]):

            # loop through the number of test points
            for iTest in np.arange(0, np.shape(x_function)[0]):

                # loop through the number of test points
                for iTrain in np.arange(0, np.shape(x_train)[0]):

                    # calculate the derivative for the test points
                    derivative[iTest, dim] += theta * weights[iTrain] * \
                                              (x_train[iTrain, dim] -
                                              x_function[iTest, dim]) * \
                                              K[iTrain, iTest]

    # 2nd derivative
    elif n_derivative == 2:

        # loop through dimensions
        for dim in np.arange(0, np.shape(x_function)[1]):

            # loop through the number of test points
            for iTest in np.arange(0, np.shape(x_function)[0]):

                # loop through the number of test points
                for iTrain in np.arange(0, np.shape(x_train)[0]):
                    derivative[iTest, dim] += weights[iTrain] * \
                                              (theta ** 2 *
                                               (x_train[iTrain, dim] - x_function[iTest, dim]) ** 2
                                               - theta) * \
                                              K[iTrain, iTest]

    return derivative


def rbf_full_derivative_memory(x_train, x_function, weights, gamma):
    """This function calculates the rbf derivative using no
    loops but it requires a large memory load.

    Parameters
    ----------
    x_train : array, [N x D]
        The training data used to find the kernel model.

    x_function  : array, [M x D]
        The test points (or vector) to use.

    weights   : array, [N x D]
        The weights found from the kernel model
            y = K * weights

    kernel_mat: array, [N x M], default: None
        The rbf kernel matrix with the similarities between the test
        points and the training points.

    n_derivative : int, (default = 1) {1, 2}
        chooses which nth derivative to calculate

    gamma : float, default: None
        the parameter for the rbf_kernel matrix function

    Returns
    -------

    derivative : array, [M x D]
        returns the derivative with respect to training points used in
        the kernel model and the test points.

    Information
    -----------
    Author: Juan Emmanuel Johnson
    Email : jej2744@rit.edu
            juan.johnson@uv.es
    """
    n_train_samples = x_train.shape[0]
    n_test_samples = x_function.shape[0]
    n_dimensions = x_train.shape[1]
    
    K = pairwise_kernels(x_function, x_train, gamma=gamma)

    # create empty block matrices and sum
    derivative = np.tile(weights[:, np.newaxis, np.newaxis],
                           (1, n_test_samples, n_dimensions)) * \
                      (np.tile(x_function[np.newaxis, :, :],
                              (n_train_samples, 1, 1)) - \
                      np.tile(x_train[:, np.newaxis, :],
                           (1, n_test_samples, 1))) * \
                      np.tile(K[:, :, np.newaxis],
                              (1, 1, n_dimensions))

    # TODO: Write code for 2nd Derivative
    # multiply by the constant
    derivative *= -2 * gamma

    return derivative


def main():

    from sklearn.kernel_ridge import KernelRidge
    import numpy as np
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y = rng.randn(n_samples)
    x = rng.randn(n_samples, n_features)

    lam = 1.0
    gamma = 1.0

    print('Initializing Model...')
    krr_model = KernelRidge(kernel='rbf',
                            alpha=lam,
                            gamma=gamma)

    print('Fitting kernel model...')
    krr_model.fit(x, y)

    print(krr_model)

    weights = krr_model.dual_coef_


    return None


if __name__ == "__main__":
    main()
