import numpy as np
from time import time
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
from kernellib.kernels import ard_kernel
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from kernellib.utils import estimate_sigma
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
    def __init__(self, length_scale=1.0, sigma_y=0.01, scale=1.0, kernel='rbf'):
        self.length_scale = length_scale
        self.sigma_y = sigma_y
        self.scale = scale
        self.kernel = kernel

    def fit(self, x, y=None):
        
        # check the input data
        x_train, y_train = check_X_y(x, y)

        # check input dimensions
        self.n_train, self.d_dim = x_train.shape

        # calculate kernel function
        self.x_train = x
        K_train = ard_kernel(self.x_train, length_scale=self.length_scale)

        # Solve for the weights
        K_train_inv = np.linalg.inv(K_train + self.sigma_y * np.eye(N=x_train.shape[0]))

        try:
            weights = np.dot(K_train_inv, y)
        except:
            weights = _solve_cholesky_kernel(K_train, y, self.sigma_y)

        # make sure weights is a 2d array
        if weights.ndim == 1:
            weights = weights[:, np.newaxis]

        self.K_ = K_train
        self.K_inv_ = K_train_inv
        self.weights_ = weights

        return self

    def predict(self, x, return_variance=False):

        # check inputs
        x_test = check_array(x)

        # calculate the kernel function with new points
        K_traintest = ard_kernel(x=self.x_train, y=x, length_scale=self.length_scale)

        if not return_variance:
            return np.dot(K_traintest.T, self.weights_)

        else:
            predictions = np.dot(K_traintest.T, self.weights_)

            return predictions, self._calculate_variance(x_test, K_traintest)

    def _calculate_variance(self, x, K_traintest=None):

        x_test = check_array(x)

        K_test = ard_kernel(x_test, length_scale=self.length_scale)

        if K_traintest is None:
            K_traintest = ard_kernel(x_test, length_scale=self.length_scale)

        return self.sigma_y  + np.diag(K_test - np.dot(K_traintest.T, np.dot(self.K_inv_, K_traintest)))


def main():
    """Example script to test the KRR function.
    """
    # generate dataset
    random_state = 123
    num_points = 1000
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
    length_scale = estimate_sigma(x_train, method='mean')
    sigma_y = 0.01
    # initialize the kernel ridge regression model
    krr_model = KernelRidge(length_scale=length_scale, sigma_y=sigma_y)



    # fit model to data
    krr_model.fit(x_train, y_train.squeeze())

    # predict using the krr model
    y_pred, var = krr_model.predict(x_test, return_variance=True)

    error = mean_absolute_error(y_test, y_pred)
    print('\nMean Absolute Error: {:.4f}\n'.format(error))

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

