import numpy as np
from time import time
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
from kernels import ard_kernel
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model.ridge import _solve_cholesky_kernel
from utils import estimate_sigma
from sklearn.utils import check_random_state
import scipy as scio
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# TODO - Test Derivative
# TODO - Test Variance


class KRR(BaseEstimator, RegressorMixin):
    """Kernel Ridge Regression with different regularizers.
    An implementation of KRR algorithm with different
    regularization parameters (weights, 1st derivative and the
    2nd derivative). Used the scikit-learn class system for demonstration
    purposes.

    Parameters
    ----------
    sigma : float, optional(default=None)
        the parameter for the kernel function.
        NOTE - gamma in scikit learn is defined as follows:
            gamma = 1 / (2 * sigma ^ 2)

    lam : float, options(default=None)
        the trade-off parameter between the mean squared error
        and the regularization term.

        alpha = inv(K + lam * reg) * y

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


    def __init__(self, sigma=None, lam=None):
        self.sigma = sigma
        self.lam = lam

    def fit(self, x, y=None):

        # regularization trade off parameter
        if self.lam is None:

            # common heuristic for minimizing the lambda value
            self.lam = 1.0e-4

        # kernel length scale parameter
        if self.sigma is None:

            # common heuristic for finding the sigma value
            self.sigma = estimate_sigma(x, method='mean')

        # calculate kernel function
        self.X_fit_ = x
        self.K_ = ard_kernel(self.X_fit_, y=self.X_fit_, length_scale=self.sigma)

        # Solve for the weights
        self.weights_ = _solve_cholesky_kernel(self.K_, y, self.lam)

        # make sure weights is a 2d array
        if self.weights_.ndim == 1:
            self.weights_ = self.weights_[:, np.newaxis]

        return self

    def predict(self, x, return_variance=False):

        # calculate the kernel function with new points
        K_traintest = ard_kernel(x=self.X_fit_, y=x, length_scale=self.sigma)

        # calculate the predictions
        predictions = np.dot(K_traintest.T, self.weights_)

        # calculate the variance
        if return_variance:
            K_test = ard_kernel(x, length_scale=self.sigma)
            self.K_inverse_ = np.linalg.inv(self.K_)


            self.variance_ = np.diag(K_test) - \
                             np.diag(
                                 np.dot(K_traintest.T,
                                        np.dot(self.K_inverse_,
                                               K_traintest)))

        # return the project points
        return predictions


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

    # initialize the kernel ridge regression model
    krr_model = KRR()

    # fit model to data
    krr_model.fit(x_train, y_train)

    # predict using the krr model
    y_pred = krr_model.predict(x_test)

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

