import numpy as np
from time import time
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error
from kernellib.kernels import ard_kernel
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from kernellib.utils import estimate_sigma, get_grid_estimates
# from kernellib.batch import kernel_model_batch
from sklearn.utils import check_random_state
import scipy as scio
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# TODO - Test Derivative
# TODO - Test Variance


class KernelRidge(BaseEstimator, RegressorMixin):
    """A simple Kernel Ridge Regression Algorithm. It is based off of
    the scikit-learn implementation and it uses the  scikit-learn 
    base/regression estimator system to gain access to the gridsearch
    features.

    Parameters
    ----------
    length_Scale : float, optional(default=1.0)
        the parameter for the kernel function which controls the 
        gaussian window along for the kernel function.
        NOTE - gamma in scikit learn is defined as follows:
            gamma = 1 / (2 * sigma ^ 2)

    sigma_y : float, options(default=0.01)
        The noise parameter estimate to control the variance of the estimates.

    scale : float, (default = 1.0)
        The constant scale parameter for the signal variance of the data.

    calculate_variance : bool, default=False
        The flag whether or not to calculate the derivative of the kernel
        function.

    Attributes
    ----------
    weights_ : array, [N x D]
        the weights found fromminimizing the cost function

    K_ : array, [N x N]
        the kernel matrix with sigma parameter
    
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
           : juan.johnson@uv.es
    Date   : 6 - July - 2018
    """
    def __init__(self, length_scale=1.0, sigma=0.01, scale=1.0, kernel='rbf'):
        self.length_scale = length_scale
        self.sigma = sigma
        self.scale = scale
        self.kernel = kernel

    def fit(self, x, y=None):
        
        # check the input data
        X, y = check_X_y(x, y, multi_output=True,
                         y_numeric=True)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True
        
        # check input dimensions
        self.n_train, self.d_dim = X.shape

        #
        self.sigma = np.atleast_1d(self.sigma)

        # calculate kernel function
        K_train = ard_kernel(X, length_scale=self.length_scale, scale=self.scale)

        # Solve for the weights
        weights = _solve_cholesky_kernel(K_train, y, self.sigma)

        self.X_fit_ = X
        self.K_ = K_train
        self.weights_ = weights

        return self

    def predict(self, X, return_variance=False):

        check_is_fitted(self, ["X_fit_", "weights_"])

        # check inputs
        X = check_array(X)

        # calculate the kernel function with new points
        K_traintest = ard_kernel(x=self.X_fit_, y=X, length_scale=self.length_scale)

        return np.dot(K_traintest.T, self.weights_)


def train_krr(X, Y, seed=0, grid=None):

    # Split data into training and validation
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.33,
                                                        random_state=seed)
    
    # Get length scale and sigma y values to check
    length_scales, sigma_ys = get_grid_estimates(X, method='mean', n_grid=20, gamma_grid='extra')
    
    
    
    rmse = np.inf
    best_length_scale = 0
    best_sigma_y = 0
    
    for ilength_scale in length_scales:
        
        Ktrain = ard_kernel(x_train, length_scale=ilength_scale)
        Ktest = ard_kernel(x_test, x_train, length_scale=ilength_scale)
        
        for isigma in sigma_ys:
            
            # Train
            weights = _solve_cholesky_kernel(Ktrain, y_train, isigma)
            
            # Validate
            y_pred = np.dot(Ktest, weights)
            
            # check results
            residuals = mean_squared_error(y_pred, y_test)
            
            if residuals < rmse:
                best_length_scale = ilength_scale
                best_sigma = isigma
                
    krr_model = KernelRidge(kernel='rbf', length_scale=best_length_scale, sigma=best_sigma)
    krr_model.fit(X, Y)
    
    return krr_model

def main():
    """Example script to test the KRR function.
    """
    # generate dataset
    random_state = 123
    num_points = 2000
    x_data, y_data = get_sample_data(random_state=random_state,
                                     num_points=num_points)


    # Split Data into Training and Testing
    train_prnt = 0.2

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        train_size=train_prnt,
                                                        random_state=random_state)

    # remove the mean from y training ONLY
    y_mean = np.mean(y_train)
    y_train -= y_mean
    y_test -= y_mean

    # Estimate the parameters

    # initialize the kernel ridge regression model
    krr_model = train_krr(x_train, y_train, grid='extra')


    # predict using the krr model
    y_pred= krr_model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print('\nMean Absolute Error: {:.4f}\n'.format(mae))
    print('\nMean Squared Error: {:.4f}\n'.format(mse))

    # plot the results
    fig, ax = plt.subplots()

    # plot kernel model
    ax.scatter(x_test, y_pred, color='k', label='KRR Model')

    # plot data
    ax.scatter(x_test, y_test, color='r', label='Data')

    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.title('Fitted Model')

    plt.show()

    # -------------------------------
    # KRR with Cross Validation
    # -------------------------------
       # fit model to data
    length_scale = estimate_sigma(x_train, method='mean')
    sigma_y = 0.01
    krr_model = KernelRidge(sigma=sigma_y, length_scale=length_scale, kernel='rbf')
    krr_model.fit(x_train, y_train.squeeze())
    # predict using the krr model
    y_pred= krr_model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print('\nMean Absolute Error: {:.4f}\n'.format(mae))
    print('\nMean Squared Error: {:.4f}\n'.format(mse))

    # plot the results
    fig, ax = plt.subplots()

    # plot kernel model
    ax.scatter(x_test, y_pred, color='k', label='KRR Model')

    # plot data
    ax.scatter(x_test, y_test, color='r', label='Data')

    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.title('Fitted Model')
    plt.show()

    return None


def get_sample_data(random_state=123, num_points=1000, plot=None):

    # generate datasets
    x_data = np.linspace(-2 * np.pi, 2 * np.pi, num=num_points)
    y_data = np.sin(x_data)

    # add some noise
    generator = check_random_state(random_state)
    y_data += 0.2 * generator.randn(num_points)

    # convert to 2D, float array for scikit-learn input
    x_data = x_data[:, np.newaxis].astype(np.float)
    y_data = y_data[:, np.newaxis].astype(np.float)

    if plot:
        fig, ax = plt.subplots()

        # plot kernel model
        ax.plot(x_data[::10], y_data[::10],
                color='k', label='data')

        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.title('Original Data')

        plt.show()

    return x_data, y_data


if __name__ == "__main__":
    main()

