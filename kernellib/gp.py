import sys
sys.path.insert(0, '/home/emmanuel/github_repos/kernellib/')

import numpy as np 
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (WhiteKernel, RBF, ConstantKernel,
                                              _check_length_scale)
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from kernellib.kernels import (rbf_kernel, calculate_q_numba,
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


class GP_Simple(BaseEstimator, RegressorMixin):
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
        try:
            if self.length_scale.any() is None or self.sigma_y is None:            
                self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        except:
            if self.length_scale is None or self.sigma_y is None:
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
        L = np.linalg.cholesky(K_train + self.sigma_y * np.eye(K_train.shape[0])) 
        
        # Calculate the weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        if weights.ndim == 1:
            weights = weights[:, np.newaxis]
            
        # save variables
        self.x_train = x_train
        self.y_train = y_train
        self.K_ = K_train
        self.L_ = L
        self.weights_ = weights
            
    def predict(self, x, return_variance=False):
        
        x_test = check_array(x)
        
        # Calculate the train test kernel
        if self.kernel in ['rbf', 'ard']:
            K_traintest = ard_kernel(x_test, self.x_train, length_scale=self.length_scale)
            
        elif self.kernel in ['weighted']:
            K_traintest = ard_kernel_weighted(x_test, self.x_train, x_cov=self.x_covariance,
                                              length_scale=self.length_scale)
        else:
            raise ValueError('Unrecognized kernel function.')
        
        if not return_variance:
            return K_traintest.dot(self.weights_)
        
        else:
            predictions = K_traintest.dot(self.weights_)
            
            variance = self._calculate_variance(x_test, K_traintest)
            return predictions, variance
    
    def _calculate_variance(self, x, K_traintest=None):
        
        x_test = check_array(x)
        
        K_test = np.diag(ard_kernel(x_test, length_scale=self.length_scale))
        
        if K_traintest is None:
            if self.kernel in ['rbf', 'ard']:
                K_traintest = ard_kernel(x_test, self.x_train, length_scale=self.length_scale)

            elif self.kernel in ['weighted']:
                K_traintest = ard_kernel_weighted(x_test, self.x_train, x_cov=self.x_covariance,
                                                  length_scale=self.length_scale)
            else:
                raise ValueError('Unrecognized kernel function.')
        
        v = np.linalg.solve(self.L_, K_traintest.T)
        
        return self.sigma_y + K_test - np.diag(np.dot(v.T, v))

class GP_Derivative(BaseEstimator, RegressorMixin):
    def __init__(self, length_scale=None, x_covariance=1.0, sigma_y=None, scale=None, 
                 variance_func='diagonal'):
        
        if isinstance(length_scale, float):
            self.length_scale = np.array([length_scale])
        
        if isinstance(x_covariance, float):
            self.x_covariance = np.asarray([x_covariance])
        self.length_scale = np.asarray(length_scale)
        self.x_covariance = np.asarray(x_covariance)
        self.sigma_y = sigma_y
        self.scale = scale
        self.variance_func = variance_func
        
    def fit(self, x, y):
        
        # check input dimensions
        x_train, y_train = check_X_y(x, y)
        
        self.n_train, self.d_dim = x_train.shape

        self.length_scale = _check_length_scale(x, self.length_scale)
        self.x_covariance = _check_length_scale(x, self.x_covariance)
        
        if np.ndim(self.x_covariance) == 0:
            self.x_covariance = np.array([self.x_covariance])
        
        # check if length scale and sigma y are there
        try:
            if self.length_scale.any() is None or self.sigma_y is None:            
                self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        except:
            if self.length_scale is None or self.sigma_y is None:
                self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        
            
        if self.scale is None:
            self.scale = 1.0
            
        # Calculate the training kernel (ARD)
        K_train = ard_kernel(x_train, length_scale=self.length_scale, scale=self.scale)
        
        # add white noise kernel and diagonal derivative term 
        L = np.linalg.cholesky(K_train + self.sigma_y * np.eye(N=self.n_train))
        
        # Calculate the weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        if weights.ndim == 1:
            weights = weights[:, np.newaxis]
            
        # Calculate the derivative for the training points
        if self.variance_func == 'diagonal':
            derivative = ard_derivative(x_train, x_train, 
                                        weights=weights, 
                                        length_scale=self.length_scale)
            derivative = np.diag(np.diag(
                np.dot(derivative, np.dot(np.diag(self.x_covariance), derivative.T))))
            
        elif self.variance_func == 'full':
            derivative = ard_derivative(x_train, x_train, 
                                        weights=weights, 
                                        length_scale=self.length_scale)
            
            derivative = derivative.dot(np.diag(self.x_covariance)).dot(derivative.T)
        else:
            raise ValueError('Unrecognized variance function type.')
        
        # add white noise kernel
        L_der = np.linalg.cholesky(K_train + self.sigma_y * np.eye(K_train.shape[0]) + derivative)
        
        # save variables
        self.x_train = x_train
        self.y_train = y_train
        self.K_ = K_train
        self.L_ = L
        self.derivative = derivative
        self.L_der_ = L_der
        self.weights_ = weights
        
    def predict(self, x, return_variance=False):
        
        x_test = check_array(x)
        
        # Calculate the weights
        K_traintest = ard_kernel(x_test, self.x_train,
                                 length_scale=self.length_scale)
        
        if not return_variance:
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
        L = np.linalg.cholesky(K_train + self.sigma_y * np.eye(self.n_train))
        initial_weights = np.linalg.solve(L.T, np.linalg.solve(L, y))[:, np.newaxis]
        
        # Calculate the derivative  
        return ard_derivative(x, x, weights=initial_weights, length_scale=self.length_scale)

    def _calculate_variance(self, x, K_traintest=None):
        
        x_test = check_array(x)
        
        K_test = np.diag(ard_kernel(x_test, length_scale=self.length_scale))
        
        if K_traintest is None:
            K_traintest = ard_kernel(x_test, self.x_train, length_scale=self.length_scale)
        

        
        v = np.linalg.solve(self.L_der_, K_traintest.T)
        
        
        # calculate the derivative for the testing points
        derivative = ard_derivative(self.x_train, x_test, weights=self.weights_, 
                                    length_scale=self.length_scale)
        
        derivative = np.diag(np.dot(derivative, np.dot(np.diag(self.x_covariance), derivative.T)))
        
        return self.sigma_y + derivative + K_test - np.diag(np.dot(v.T, v))

class GP_Corrective(object):
    def __init__(self, length_scale=None, x_covariance=1.0, sigma_y=None, scale=None):
        self.length_scale = length_scale
        self.x_covariance = x_covariance
        self.sigma_y = sigma_y
        self.scale = scale
        
    def fit(self, x, y):
        
        # check input dimensions
        x_train, y_train = check_X_y(x, y)
        
        self.n_train, self.d_dim = x_train.shape
        
        
        self.length_scale = _check_length_scale(x, self.length_scale)
        self.x_covariance = _check_length_scale(x, self.x_covariance)
        
        if np.ndim(self.length_scale) == 0:
            self.length_scale = np.array([self.length_scale])
            
        if np.ndim(self.x_covariance) == 0:
            self.x_covariance = np.array([self.x_covariance])
            
        # check if length scale and sigma y are there
        try:
            if self.length_scale.any() is None or self.sigma_y is None:            
                self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        except:
            if self.length_scale is None or self.sigma_y is None:
                self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        
            
        if self.scale is None:
            self.scale = 1.0
        
        # --------------------
        # Initial Weights
        # --------------------
        
        # Calculate the training Kernel (ARD)
        K_train = ard_kernel(x_train, length_scale=self.length_scale, scale=self.scale)
        
        # add white noise kernel and diagonal derivative term 
        L = np.linalg.cholesky(K_train + self.sigma_y * np.eye(N=self.n_train)) 
        
        
        # Calculate the weights
        init_weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        if init_weights.ndim == 1:
            init_weights = init_weights[:, np.newaxis]
        
        # Calculate some initial weights
        derivative = ard_derivative(x_train, x_train,  weights=init_weights, 
                                    length_scale=self.length_scale)
        derivative = np.diag(np.diag(
            np.dot(derivative, np.dot(np.diag(self.x_covariance), derivative.T))))
        
        # ----------------------
        # Final Weights
        # ----------------------
        L = np.linalg.cholesky(K_train + self.sigma_y * np.eye(N=self.n_train) + derivative) 
                    
        # Calculate the weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        if weights.ndim == 1:
            weights = weights[:, np.newaxis]
        
        # save variables
        self.x_train = x_train
        self.y_train = y_train
        self.K_ = K_train
        self.L_ = L
        self.derivative_ = derivative
        self.weights_ = weights
        
        return self
    
    def predict(self, x, return_variance=False):
        
        x_test = check_array(x)
        
        # Calculate the weights
        K_traintest = ard_kernel(x_test, self.x_train,
                                 length_scale=self.length_scale)
        
        if not return_variance:
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
        self.der_weights_ = initial_weights
        
        # Calculate the derivative        
        return ard_derivative(x, x, weights=initial_weights, length_scale=self.length_scale)
    
    def _calculate_variance(self, x, K_traintest=None):
        
        x_test = check_array(x)
        
        K_test = np.diag(ard_kernel(x_test, length_scale=self.length_scale))
        
        if K_traintest is None:
            K_traintest = ard_kernel(x_test, self.x_train, length_scale=self.length_scale)
        

        
        v = np.linalg.solve(self.L_, K_traintest.T)
        
        
        # calculate the derivative for the testing points
        derivative = ard_derivative(self.x_train, x_test, weights=self.weights_, 
                                    length_scale=self.length_scale)
        
        derivative = np.diag(np.dot(derivative, np.dot(np.diag(self.x_covariance), derivative.T)))
        
        return self.sigma_y + derivative + K_test - np.diag(np.dot(v.T, v))

        
class GP_DCorrective(object):
    def __init__(self, length_scale=None, x_covariance=1.0, sigma_y=None, scale=None):
        self.length_scale = length_scale
        self.x_covariance = x_covariance
        self.sigma_y = sigma_y
        self.scale = scale
        
    def fit(self, x, y):
        
        # check input dimensions
        x_train, y_train = check_X_y(x, y)
        
        self.n_train, self.d_dim = x_train.shape
        
        
        self.length_scale = _check_length_scale(x, self.length_scale)
        self.x_covariance = _check_length_scale(x, self.x_covariance)
        
        if np.ndim(self.length_scale) == 0:
            self.length_scale = np.array([self.length_scale])
            
        if np.ndim(self.x_covariance) == 0:
            self.x_covariance = np.array([self.x_covariance])
            
        # check if length scale and sigma y are there
        try:
            if self.length_scale.any() is None or self.sigma_y is None:            
                self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        except:
            if self.length_scale is None or self.sigma_y is None:
                self.length_scale, self.sigma_y = fit_gp(x_train, y_train, n_restarts=0)
        
            
        if self.scale is None:
            self.scale = 1.0
            
        # Calculate the training Kernel (ARD)
        K_train = ard_kernel(x_train, length_scale=self.length_scale, scale=self.scale)
        
        # Calculate the derivative
        derivative = self._calculate_derivative(x_train, y_train, K_train)

        # Add white noise kernel and derivative term
        derivative_term = np.diag(np.diag(derivative.dot(np.diag(self.x_covariance)).dot(derivative.T)))
        
        K_train += self.sigma_y**2 * np.eye(N=self.n_train) + derivative_term
            
        K_train_inv = np.linalg.inv(K_train)
        
        # Calculate the weights
        weights = K_train_inv.dot(y_train)
        
        # save variables
        self.x_train = x_train
        self.y_train = y_train
        self.K_ = K_train
        self.K_inv_ = K_train_inv
        self.derivative_ = derivative
        self.weights_ = weights
        
        return self
    
    def predict(self, x, return_std=False):
        
        x_test = check_array(x)
        
        # Calculate the weights
        K_traintest = ard_kernel_weighted(x_test, self.x_train, 
                                          x_cov=self.x_covariance, 
                                          length_scale=self.length_scale)
            
        if not return_std:
            return K_traintest.dot(self.weights_)
        
        else:
            predictions = K_traintest.dot(self.weights_)
            variance = self._calculate_variance(x_test, predictions=predictions)
            return predictions, variance
           
    def _calculate_derivative(self, x, y, K_train=None):
        
        # Calculate the training Kernel (ARD)
        if K_train is None:
            K_train = ard_kernel(x, length_scale=self.length_scale, scale=self.scale)

        # Calculate the weights for the initial kernel
        L = np.linalg.cholesky(K_train + self.sigma_y**2 * np.eye(self.n_train))
        initial_weights = np.linalg.solve(L.T, np.linalg.solve(L, y))[:, np.newaxis]
        self.der_weights_ = initial_weights
        
        # Calculate the derivative        
        return ard_derivative(x, x, weights=initial_weights, length_scale=self.length_scale)
    
    def _calculate_variance(self, x, predictions=None):
        
        x_test = check_array(x)
        n_test = x_test.shape[0]
        
        if predictions is None:
            predictions = self.predict(x_test, return_std=False)

        # Determinant Term
        det_term = 2 * self.x_covariance * np.power(self.length_scale, -2) + 1
        det_term = 1 / np.sqrt(np.linalg.det(np.diag(det_term)))
        
        # Exponential Term
        exp_scale = np.power(np.power(self.length_scale, 2) 
                             + 0.5 * np.power(self.length_scale, 4) 
                             * np.power(self.x_covariance, -1), -1)
                
        K = ard_kernel(self.x_train, x_test, length_scale=self.length_scale)
        
        
        variance = np.zeros(shape=(n_test))
        trace_term = variance.copy()
        q_weight_term = variance.copy()
        pred_term = variance.copy()
        
        # Loop through test points
        for itertest in range(n_test):
            
            # Calculate Q matrix
            Q = calculate_q_numba(self.x_train, x_test[itertest, :], K[:, itertest], det_term, exp_scale)
            
            # Terms
            trace_term[itertest] = float(np.trace(np.dot(self.K_inv_, Q)))
            q_weight_term[itertest] = float(self.weights_.T.dot(Q).dot(self.weights_))
            pred_term[itertest] = float(predictions[itertest]**2)
            # calculate the final predictive variance
            variance[itertest] = self.scale - trace_term[itertest] + \
                q_weight_term[itertest] - pred_term[itertest]
        
        # Negative variances due to numerical issues.
        # Set those variances to 0.
        var_negative = variance < 0
        if np.any(var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those varinaces to 0.")
            
            variance[var_negative] = 0.0
        return variance
        

def main():

    rng = np.random.RandomState(0)

    X = 15 * rng.rand(100, 1)
    y = np.sin(X).ravel()

    y += 3 * (0.5 - rng.rand(X.shape[0]))

    # try GP simple
    length_scale, sigma_y = fit_gp(X, y)
    print('Length Scale: {:.3f}'.format(length_scale))
    print('Sigma y: {:.3f}'.format(sigma_y))

     # ------------------------
    # Simple GP (w/o) fitting
    # -------------------------
    gp_standard = GP_Simple(length_scale=length_scale, sigma_y=sigma_y)
    
    gp_standard.fit(X, y.squeeze())

    mean = gp_standard.predict(X)

    score = mean_absolute_error(mean, y.squeeze())
    print('GP (Simple) - MAE: {:.4f}'.format(score))

     # ------------------------
    # Simple GP (w/o) fitting
    # -------------------------
    gp_corrective = GP_Corrective(length_scale=length_scale, sigma_y=None, x_covariance=0.0001)
    
    gp_corrective.fit(X, y.squeeze())

    mean = gp_corrective.predict(X)

    score = mean_absolute_error(mean, y.squeeze())
    print('GP (Corrective) - MAE: {:.4f}'.format(score))

     # ------------------------
    # Simple GP (w/o) fitting
    # -------------------------
    gp_derivative = GP_Derivative(length_scale=length_scale, sigma_y=None, x_covariance=0.0001)
    
    gp_derivative.fit(X, y.squeeze())

    mean = gp_derivative.predict(X)

    score = mean_absolute_error(mean, y.squeeze())
    print('GP (Derivative) - MAE: {:.4f}'.format(score))

     # ------------------------
    # Simple GP (w) fitting
    # -------------------------
    gp_standard = GP_Simple()
    
    gp_standard.fit(X, y.squeeze())

    mean = gp_standard.predict(X)

    score = mean_absolute_error(mean, y.squeeze())
    print('GP (Simple) - MAE: {:.4f}'.format(score))

     # ------------------------
    # Simple GP (w/o) fitting
    # -------------------------
    gp_corrective = GP_Corrective(x_covariance=0.0001)
    
    gp_corrective.fit(X, y.squeeze())

    mean = gp_corrective.predict(X)

    score = mean_absolute_error(mean, y.squeeze())
    print('GP (Corrective) - MAE: {:.4f}'.format(score))

     # ------------------------
    # Simple GP (w/o) fitting
    # -------------------------
    gp_derivative = GP_Derivative(x_covariance=0.0001)
    
    gp_derivative.fit(X, y.squeeze())

    mean = gp_derivative.predict(X)

    score = mean_absolute_error(mean, y.squeeze())
    print('GP (Derivative) - MAE: {:.4f}'.format(score))

    return None

if __name__ == '__main__':
    main()