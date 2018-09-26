import numpy as np
import numba
from numba import float64
from numba import prange
from kernellib.kernels import ard_kernel
from kernellib.kernels import rbf_kernel
from sklearn.metrics import pairwise_kernels
from kernellib.krr import KernelRidge
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances
from sklearn.gaussian_process.kernels import _check_length_scale
from scipy.linalg import cholesky, cho_solve

# TODO: Write tests for derivative functions, gradients
# TODO: Implement Derivative w/ 1 loop for memory conservation
# TODO: Implement 2nd Derivative for all
# TODO: Do Derivative for other kernel methods (ARD, Polynomial)


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
        x = x.astype(np.float64)
        K = rbf_kernel(x, self.x_train, length_scale=self.length_scale, signal_variance=self.signal_variance)
        
        return self.rbf_derivative(
            self.x_train, x, K, self.weights.flatten(), self.length_scale)
                

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
    def rbf_derivative(x_train, x_function, K, weights, length_scale):
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
        return self.ard_derivative(self.x_train, X, K, self.weights, self.length_scale)
                
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
    def ard_derivative(x_train, x_function, K, weights, length_scale):

        n_test, n_dims = x_function.shape

        derivative = np.zeros(shape=x_function.shape)
        length_scale = np.diag(- np.power(length_scale, -2))
        for itest in range(n_test):
            derivative[itest, :] = np.dot(length_scale.dot((x_function[itest, :] - x_train).T),
                                            (K[itest, :].reshape(-1, 1) * weights))

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
    kernel_mat = ard_kernel(x_test, x_train, length_scale=length_scale, signal_variance=scale)
    
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


    return None


if __name__ == "__main__":
    main()
