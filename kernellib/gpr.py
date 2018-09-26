import numpy as np
import numba
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (WhiteKernel, RBF, ConstantKernel,
                                              _check_length_scale)
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
from kernellib.kernels import (calculate_q_numba,
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
