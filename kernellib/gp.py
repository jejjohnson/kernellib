import sys

import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (WhiteKernel, RBF, ConstantKernel,
                                              _check_length_scale)
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from kernellib.kernels import (rbf_kernel, calculate_q_numba, rbf_kernel_weighted_1d,
                               ard_kernel, ard_kernel_weighted)
from kernellib.derivatives import rbf_derivative, ard_derivative



def fit_gp(x_train, y_train, kernel='ard', scale=None,
           length_scale_bounds = (0.001, 100),
           noise_level_bounds = (1e-4, 10),
           n_restarts=3):
    
    warnings.simplefilter('ignore')
    x_train, y_train = check_X_y(x_train, y_train)
    warnings.simplefilter('default')
    n_train, d_dims = x_train.shape
    
    if kernel in ['ard', 'ARD']:
        length_scale_init = np.ones(shape=(d_dims))
    elif kernel in ['rbf', 'RBF']:
        length_scale_init = 1.0
    else:
        raise ValueError('Unrecognized kernel function...')
    
    noise_level_init = 1.0

    gp_kernel = RBF(length_scale=length_scale_init,
                    length_scale_bounds=length_scale_bounds) + \
                WhiteKernel(noise_level=noise_level_init,
                            noise_level_bounds=noise_level_bounds)
    
    gpr_model = GaussianProcessRegressor(kernel=gp_kernel, random_state=123,
                                         n_restarts_optimizer=n_restarts)
    
    # Fit the GP Model
    gpr_model.fit(x_train, y_train)
    
    # the parameters
    length_scale = gpr_model.kernel_.k1.length_scale
    sigma = gpr_model.kernel_.k2.noise_level
    
    return length_scale, sigma

def gp_simple(x_train, x_test, y_train, length_scale, sigma_y, y_test=None):

     # Calculate the training Kernel
    K_train = rbf_kernel(x_train, length_scale=length_scale)
    
    # calculate the weights
    L = np.linalg.cholesky(K_train + sigma_y**2 * np.eye(K_train.shape[0]))
    weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    # Predictions
    K_traintest = rbf_kernel(x_test, x_train, length_scale=length_scale)
    prediction = K_traintest.dot(weights)
    
    # Score for test points
    if y_test is not None:
        score = mean_absolute_error(prediction, y_test)
    else:
        score = None
    
    # Variance
    K_test = rbf_kernel(x_test, length_scale=length_scale)
    v = np.linalg.solve(L, K_traintest.T)
    variance = np.diag(K_test - v.T.dot(v))   

    return prediction, variance, score

class GP_Simple(object):
    def __init__(self, length_scale=None, sigma_y=None, scale=None, kernel='ard',
                 x_covariance=None):
        self.length_scale = length_scale
        self.sigma_y = sigma_y
        self.scale = scale
        self.kernel = kernel
        self.x_covariance = x_covariance
        
    def fit(self, x, y):
        
        # check input dimensions
        x_train, y_train = check_X_y(x, y)
        
        self.n_train, self.d_dim = x_train.shape
        
        # check if length scale and sigma y are there
        if self.length_scale.any() is None or self.sigma_y is None:            
            self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        
            
        if self.scale is None:
            self.scale = 1.0
            
        # Calculate the training kernel (ARD)
        if self.kernel in ['rbf', 'ard']:
            K_train = ard_kernel(x_train, length_scale=self.length_scale)
        elif self.kernel in ['weighted']:
            K_train = ard_kernel_weighted(x_train, x_cov=self.x_covariance,
                                          length_scale=self.length_scale)
        else:
            raise ValueError('Unrecognized kernel function.')
        
        # add white noise kernel
        L = np.linalg.cholesky(K_train + self.sigma_y**2 * np.eye(K_train.shape[0])) 
        
        # Calculate the weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        # save variables
        self.x_train = x_train
        self.y_train = y_train
        self.K_ = K_train
        self.L_ = L
        self.weights_ = weights
        
        
    def predict(self, x, return_std=False):
        
        x_test = check_array(x)
        
        # Calculate the train test kernel
        if self.kernel in ['rbf', 'ard']:
            K_traintest = ard_kernel(x_test, self.x_train, length_scale=self.length_scale)
            
        elif self.kernel in ['weighted']:
            K_traintest = ard_kernel_weighted(x_test, self.x_train, x_cov=self.x_covariance,
                                              length_scale=self.length_scale)
        else:
            raise ValueError('Unrecognized kernel function.')
        
        if not return_std:
            return K_traintest.dot(self.weights_)
        
        else:
            predictions = K_traintest.dot(self.weights_)
            
            variance = self._calculate_variance(x_test, K_traintest)
            return predictions, variance
    
    def _calculate_variance(self, x, K_traintest=None):
        
        x_test = check_array(x)
        
        K_test = ard_kernel(x_test, length_scale=self.length_scale)
        
        if K_traintest is None:
            if self.kernel in ['rbf', 'ard']:
                K_traintest = ard_kernel(x_test, self.x_train, length_scale=self.length_scale)

            elif self.kernel in ['weighted']:
                K_traintest = ard_kernel_weighted(x_test, self.x_train, x_cov=self.x_covariance,
                                                  length_scale=self.length_scale)
            else:
                raise ValueError('Unrecognized kernel function.')
        
        v = np.linalg.solve(self.L_, K_traintest.T)
        
        return np.diag(K_test - np.dot(v.T, v))


def gp_derivative(x_train, x_test, y_train, length_scale, sigma_y, x_cov, y_test=None):
    
    # K_train matrix
    K_train = rbf_kernel(x_train, length_scale=length_scale)
    
    # Calculate initial weights
    # calculate the weights
    L = np.linalg.cholesky(K_train + sigma_y**2 * np.eye(K_train.shape[0]))
    init_weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    # Calculate the derivative
    derivative = rbf_derivative(x_train, x_train, weights=init_weights, length_scale=length_scale)
    derivative_term =  np.diag(np.diag(x_cov * derivative.dot(derivative.T)))
    
    # Calculate the weights
    L = np.linalg.cholesky(K_train + sigma_y**2 * np.eye(K_train.shape[0]) + derivative_term)
    weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    
    # Predictions
    K_traintest = rbf_kernel(x_test, x_train, length_scale=length_scale)
    prediction = K_traintest.dot(weights)
    
    # Calculate score
    if y_test is not None:
        score = mean_absolute_error(prediction, y_test)
    else:
        score = None
    
    # Variance
    K_test = rbf_kernel(x_test, length_scale=length_scale)
    v = np.linalg.solve(L, K_traintest.T)
    variance = np.diag(K_test - v.T.dot(v))
    
    return prediction, variance, score


class GP_Derivative(object):
    def __init__(self, length_scale=None, x_covariance=1.0, sigma_y=None, scale=None):
        
        if isinstance(length_scale, float):
            self.length_scale = np.array([length_scale])
        
        if isinstance(x_covariance, float):
            self.x_covariance = np.asarray([x_covariance])
        self.length_scale = np.asarray(length_scale)
        self.x_covariance = np.asarray(x_covariance)
        self.sigma_y = sigma_y
        self.scale = scale
        
    def fit(self, x, y):
        
        # check input dimensions
        x_train, y_train = check_X_y(x, y)
        
        self.n_train, self.d_dim = x_train.shape

        self.length_scale = _check_length_scale(x, self.length_scale)
        self.x_covariance = _check_length_scale(x, self.x_covariance)
        
        if np.ndim(self.x_covariance) == 0:
            self.x_covariance = np.array([self.x_covariance])
        
        # check if length scale and sigma y are there
        if self.sigma_y is None:            
            self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        
            
        if self.scale is None:
            self.scale = 1.0
            
        # Calculate the training kernel (ARD)
        K_train = ard_kernel(x_train, length_scale=self.length_scale, scale=self.scale)
        
        # calculate the derivative
        derivative = self._calculate_derivative(x_train, y_train, K_train)
        # calculate the diagonal elements
        derivative_term = np.diag(np.diag(derivative.dot(np.diag(self.x_covariance)).dot(derivative.T)))
            
        
        # add white noise kernel and diagonal derivative term 
        L = np.linalg.cholesky(K_train + self.sigma_y**2 * np.eye(N=self.n_train) + derivative_term)
        
        # Calculate the weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        # save variables
        self.x_train = x_train
        self.y_train = y_train
        self.K_ = K_train
        self.L_ = L
        self.derivative_ = derivative
        self.weights_ = weights
        
        
    def predict(self, x, return_std=False):
        
        x_test = check_array(x)
        
        # Calculate the weights
        K_traintest = ard_kernel(x_test, self.x_train,
                                 length_scale=self.length_scale)
        
        if not return_std:
            return K_traintest.dot(self.weights_)
        
        else:
            predictions = K_traintest.dot(self.weights_)
            
            variance = self._calculate_variance(x_test, K_traintest)
            return predictions, variance
        
    
    def _calculate_derivative(self, x, y, K_train=None):
        
        # Calculate the training Kernel (ARD)
        if K_train is None:
            K_train = ard_kernel(x, length_scale=self.length_scale, scale=self.scale)

        # Calculate the weights for the initial kernel
        L = np.linalg.cholesky(K_train + self.sigma_y**2 * np.eye(self.n_train))
        initial_weights = np.linalg.solve(L.T, np.linalg.solve(L, y))[:, np.newaxis]
        
        # Calculate the derivative  
        return ard_derivative(x, x, weights=initial_weights, length_scale=self.length_scale)

    
    def _calculate_variance(self, x, K_traintest=None):
        
        x_test = check_array(x)
        
        K_test = ard_kernel(x_test, length_scale=self.length_scale)
        
        if K_traintest is None:
            K_traintest = ard_kernel(x_test, self.x_train, length_scale=self.length_scale)
        
        v = np.linalg.solve(self.L_, K_traintest.T)
        
        return np.diag(K_test - np.dot(v.T, v))


def gp_corrective(x_train, x_test, y_train, length_scale, sigma_y, x_cov, y_test=None, return_var=None):
    
    scale = 1.0

    # Calculate training Kernel
    K_train = rbf_kernel(x_train, length_scale=length_scale)

    # Calculate initial weights
    L = np.linalg.cholesky(K_train + sigma_y**2 * np.eye(K_train.shape[0]))
    init_weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Calculate the derivative term
    derivative = rbf_derivative(x_train, x_train, weights=init_weights, length_scale=length_scale)
    derivative_term = np.diag(np.diag(x_cov * derivative.dot(derivative.T)))
    # Calculate the kernel matrices
    L = np.linalg.cholesky(K_train + sigma_y**2 * np.eye(K_train.shape[0]) + derivative_term)
    weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    K_train_inv = np.linalg.inv(K_train + sigma_y**2 * np.eye(K_train.shape[0]) + derivative_term)

    # Predictions
    K_traintest = rbf_kernel_weighted_1d(x_test, x_train, x_cov=x_cov, length_scale=length_scale)
    
    # Predictions
    K_traintest = rbf_kernel_weighted_1d(x_test, x_train, x_cov=x_cov, length_scale=length_scale)
    mean = K_traintest.dot(weights)
    
    # Calculate score
    if y_test is not None:
        score = mean_absolute_error(mean, y_test)
    else:
        score = None

    K_traintest = rbf_kernel(x_train, x_test, length_scale=length_scale)

    # Precalculated terms
    det_term = 1 / ((2 * x_cov * (length_scale**2)**(-1) + 1)**(1/2))
    exp_scale = 1 / (length_scale**2 + 0.5 * length_scale**4 * x_cov**(-1))
    
    if return_var is None:
        var = None
    
    else:
        
        var = np.zeros(shape=x_test.shape[0])
        trace_term = var.copy()
        q_weight_term = var.copy()
        pred_term = var.copy()

        for counter, itest in enumerate(x_test):

            Q = calculate_q_numba(x_train, 
                                  x_test[counter, :], 
                                  K_traintest[:, counter], 
                                  det_term, 
                                  np.array(exp_scale))

            # get the Q matrix for the test point
            # calculate the final mean function

            trace_term[counter] = float(np.trace(np.dot(K_train_inv, Q)))
            q_weight_term[counter]  = float(np.dot(weights.T, np.dot(Q, weights)))
            pred_term[counter] = float(mean[counter])**2

            var[counter] = scale**2 - \
                float(trace_term[counter]) + \
                float(q_weight_term[counter]) - \
                float(pred_term[counter])

            # if counter % 100 == 0:
            #     print('\nIteration: {}\n'.format(counter))
            #     print('Q: {:.5f}, {:.5f}, {}'.format(Q.min(), Q.max(), Q.shape))
            #     print('Weights: {}'.format(weights.shape))
            #     print('Trace Term: {:.5f}'.format(trace_term[counter]))
            #     print('Q_term Term: {:.5f}'.format(q_weight_term[counter]))
            #     print('mpred: {:.5f}'.format(pred_term[counter]))
            #     print('Variance: {:.5f}, {:.5f}'.format(
            #         var[counter].min(), 
            #         var[counter].max()))

    return mean, var, score


class GP_Corrective(object):