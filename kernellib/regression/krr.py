from sklearn.datasets import load_boston, make_regression
import numpy as np
import pandas as pd
from time import time

import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from scipy.spatial.distance import pdist
from scipy.linalg import cho_factor, cho_solve
from sklearn.utils import check_random_state
from sklearn.model_selection import KFold
import multiprocessing as mp
from joblib import Parallel, delayed


import scipy as scio
from matplotlib import pyplot as plt



class KRR(BaseEstimator, RegressorMixin):
    """Kernel Ridge Regression with different regularizers.
    An implementation of KRR algorithm with different
    regularization parameters (weights, 1st derivative and the
    2nd derivative). Used the scikit-learn class system for demonstration
    purposes.

    Parameters
    ----------
    solver : str, {'reg', 'chol', 'batch'}, (default='reg')
        the Ax=b solver used for the weights

    n_batches : int, default=None
        the number of samples used per batch

    sigma : float, optional(default=None)
        the parameter for the kernel function.
        NOTE - gamma in scikit learn is defined as follows:
            gamma = 1 / (2 * sigma ^ 2)

    lam : float, options(default=None)
        the trade-off parameter between the mean squared error
        and the regularization term.

        alpha = inv(K + lam * reg) * y

    Attributes
    ----------
    weights_ : array, [N x D]
        the weights found fromminimizing the cost function

    K_ : array, [N x N]
        the kernel matrix with sigma parameter
    """

    def __init__(self, sigma=None, lam=None, calculate_variance=False, batch_size=None, n_jobs=1):
        self.batch_size = batch_size
        self.sigma = sigma
        self.lam = lam
        self.calculate_variance = calculate_variance
        self.n_jobs = n_jobs

    def fit(self, x, y=None):

        # regularization trade off parameter
        if self.lam is None:

            # common heuristic for minimizing the lambda value
            self.lam = 1.0e-4

        # kernel length scale parameter
        if self.sigma is None:

            # common heuristic for finding the sigma value
            self.sigma = np.mean(pdist(x, metric='euclidean'))

        # check batch processes
        if self.batch_size is None:
            self.batch_size = 1
        if ( self.batch_size < 0 or self.batch_size > np.inf):
            raise ValueError('n_batches should be between 0 and a reasonable number.')

        # gamma parameter for the kernel matrix
        self.gamma = 1 / (2 * self.sigma ** 2)

        # calculate kernel function
        self.X_fit_ = x
        self.K_ = rbf_kernel(self.X_fit_, Y=self.X_fit_, gamma=self.gamma)
        self.K_inverse_ = np.linalg.inv(self.K_)

        # Try the cholesky factor method
        try:

            # Cholesky Factor Method
            R, lower = cho_factor(self.K_ + self.lam * np.eye(self.K_.shape[0], 1))

            # Cholesky Solve Method
            self.weights_ = cho_solve((R, lower), y)

        except np.linalg.LinAlgError:
            warnings.warn("Singular Matrix. Trying Regular Solver.")

            self.weights_ = scio.linalg.solve(self.K_ + self.lam * np.eye(self.K_.shape[0], 1),
                                              y)

        # make sure weights is a 2d array
        if self.weights_.ndim == 1:
            self.weights_ = self.weights_[:, np.newaxis]

        return self

    def predict(self, x):

        # check for batch processing
        if self.batch_size > 1:

            if self.n_jobs == 1:

                # perform batch_processing
                predictions, self.variance = \
                    krr_batch_predictions(x_train=self.X_fit_,
                                          x_test=x,
                                          weights=self.weights_,
                                          gamma=self.gamma,
                                          n_batches=self.batch_size,
                                          calculate_variance=self.calculate_variance)

            else:

                # perform multi-core batch processing (joblib)
                results = Parallel(n_jobs=self.n_jobs)(delayed(
                    krr_predictions)(
                    self.X_fit_, x[start:end], self.weights_,
                    self.gamma, calculate_variance=self.calculate_variance,
                    K_train_inv=self.K_inverse_)

                    for (start, end) in generate_batches(x.shape[0],
                                                         batch_size=self.batch_size)
                )

                predictions, variance = tuple(zip(*results))
                predictions = np.vstack(predictions)
                self.variance = np.vstack(variance)

        else:
            # calculate the kernel function with new points
            K_traintest = rbf_kernel(X=self.X_fit_, Y=x, gamma=self.gamma)

            # calculate the predictions
            predictions = np.dot(K_traintest.T, self.weights_)

            if self.calculate_variance is True:
                K_test = rbf_kernel(x, gamma=self.gamma)

                self.variance_ = np.diag(K_test) - \
                                 np.diag(np.dot(K_traintest.T, np.dot(self.K_inverse_, K_traintest)))

        # return the project points
        return predictions


def krr_batch_predictions(x_train, x_test, weights, gamma,
                          n_batches=None, calculate_variance=False):

    # split the data into K folds
    n_samples, n_dimensions = x_test.shape

    # default batch number
    if n_batches is None:
        n_batches = int(np.round(n_samples / 500))

    variance = None

    # check variance
    if calculate_variance is True:
        K_train = rbf_kernel(x_train, gamma=gamma)
        K_train_inverse = np.linalg.inv(K_train)
        variance = np.empty(shape=(n_samples, 1))

    # create a batch iterator object
    BatchIterator = KFold(n_splits=n_batches)

    # predefine matrices
    y_pred = np.empty(shape=(n_samples, 1))

    for (ibatch, (_, ibatch_index)) in enumerate(BatchIterator.split(x_test)):

        # calculate the train_test kernel
        K_traintest = rbf_kernel(x_train, x_test[ibatch_index], gamma=gamma)

        # calculate the predictions
        y_pred[ibatch_index] = np.dot(K_traintest.T, weights)

        if calculate_variance is True:

            # calculate the Kbatch
            K_batch = rbf_kernel(x_test[ibatch_index], gamma=gamma)

            # calculate the variance
            variance[ibatch_index, 0] = np.diag(K_batch) - \
                np.diag(np.dot(K_traintest.T, np.dot(K_train_inverse, K_traintest)))


    return y_pred, variance

def _calculate_predictions(x_train, x_test, indices, weights, gamma):

    # calculate train-test kernel
    K_traintest = rbf_kernel(x_train, x_test[indices, :],
                             gamma=gamma)

    # calculate the predictions
    return np.dot(K_traintest.T, weights)

def _calculate_variance(x_test, indices, K_traintest,
                        K_train_inverse, gamma):
    # TODO: Add optional x_train parameter,
    K_batch = rbf_kernel(x_test[indices, :], gamma=gamma)


    return np.diag(K_batch) - np.diag(np.dot(K_traintest.T,
                                             np.dot(K_train_inverse,
                                                    K_traintest)))


def generate_batches(n_samples, batch_size):
    """A generator to split an array of 0 to n_samples
    into an array of batch_size each.

    Parameters
    ----------
    n_samples : int
        the number of samples

    batch_size : int,
        the size of each batch


    Returns
    -------
    start_index, end_index : int, int
        the start and end indices for the batch

    Source:
        https://github.com/scikit-learn/scikit-learn/blob/master
        /sklearn/utils/__init__.py#L374
    """
    start_index = 0

    # calculate number of batches
    n_batches = int(n_samples // batch_size)

    for _ in range(n_batches):

        # calculate the end coordinate
        end_index = start_index + batch_size

        # yield the start and end coordinate for batch
        yield start_index, end_index

        # start index becomes new end index
        start_index = end_index

    # special case at the end of the segment
    if start_index < n_samples:

        # yield the remaining indices
        yield start_index, n_samples


def krr_predictions(x_train, x_test, weights, gamma, calculate_variance=False,
                    K_train_inv=None):
    variance_predictions = None

    # calculate train_test kernel
    K_traintest = rbf_kernel(x_train, x_test, gamma)

    # calculate the predictions
    mean_predictions = K_traintest.T @ weights

    # calculate the variance
    if calculate_variance:

        # calculate the kernel for test points
        K_batch = rbf_kernel(x_test, gamma=gamma)

        # calculate K_traininverse if necessary
        if K_train_inv is None:
            K_train_inv = np.linalg.inv(rbf_kernel(x_train, gamma=gamma))

        # calculate the variance
        variance_predictions = np.diag(K_batch) - \
                               np.diag(K_traintest.T @ K_train_inv @ K_traintest)

    return mean_predictions, variance_predictions

def krr_batch_multi_pred(x_train, x_test, weights, gamma,
                         batch_size=None, n_jobs=1,
                         calculate_variance=None):

    # check for num of proc vs n_jobs selected
    num_procs = mp.cpu_count()

    if num_procs < n_jobs:
        Pool = mp.Pool(processes=num_procs)
    else:
        Pool = mp.Pool(processes=n_jobs)

    # get dimensions of the data
    n_samples, n_dimensions = x_test.shape

    # default batch number
    if batch_size is None:
        batch_size = int(np.round(n_samples / 500))

    # initialize variance
    variance = None

    # check for variance entry
    if calculate_variance is True:
        K_train = rbf_kernel(x_train, gamma=gamma)
        K_train_inverse = np.linalg.inv(K_train)
        variance = np.empty(shape=(n_samples, 1))

    # create a batch iterator generator
    BatchIterator = KFold(n_splits=batch_size)

    # predefine matrices
    y_pred = np.empty(shape=(n_samples, 1))

    # Perform multiprocessing
    pred_pool = \
        [Pool.apply(_calculate_predictions, args=(
            x_train, x_test, indices, weights, gamma))
         for _, indices in BatchIterator.split(x_test)]

    # get pooled results
    for i, (_, indices) in enumerate(BatchIterator.split(x_test)):
        y_pred[indices] = pred_pool[i]

    # Calculate Variance
    if calculate_variance is True:

        # TODO: IMPORTANT - Combine prediction and variance to improve kernel
        # calculation.

        # calculate the train-test kernel
        K_traintest = rbf_kernel(x_train, x_test, gamma=gamma)

        # Perform Multiprocessing
        var_pool = \
            [Pool.apply(calculate_variance, args=(
                x_train, x_test, indices, K_traintest,
                K_train_inverse, gamma))
             for _, indices in BatchIterator.split(x_test)]

        # get pooled results
        for i, (_, indices) in enumerate(BatchIterator.split(x_test)):
            variance[indices] = var_pool[i]

    return y_pred, variance

def sklearn_krr_pred(KRR_Model,
                     x, calculate_variance=False,
                     K_train_inv=None):



    # calculate the predictions
    predictions = KRR_Model.predict(x)[:, np.newaxis]

    #
    variance = None

    # calculate the variance
    if calculate_variance:

        # calculate train-test kernel
        K_traintest = rbf_kernel(KRR_Model.X_fit_,
                                 x, gamma=KRR_Model.gamma)

        # calculate the kernel for test points
        K_batch = rbf_kernel(x, gamma=KRR_Model.gamma)

        # calculate K_traininverse if necessary
        if K_train_inv is None:

            K_train_inv = np.linalg.inv(rbf_kernel(x, gamma=KRR_Model.gamma))

        # calculate the variance
        variance = np.diag(K_batch) - \
                       np.diag(K_traintest.T @ K_train_inv @ K_traintest)



    return predictions, variance

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

def boston_example():

    x_data, y_data = load_boston(True)

    random_state = 100

    # split data into training and testing
    train_percent = 0.4

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=train_percent,
        random_state=random_state
    )

    # remove the mean from the training data
    y_mean = np.mean(y_train)

    y_train -= y_mean
    y_test -= y_mean

    # initialize the kernel ridge regression model
    krr_model = KRR(n_batches=1)

    # fit model to data
    krr_model.fit(x_train, y_train)

    # predict using the krr model
    y_pred = krr_model.predict(x_test)


    return None

def times_multi_exp_variance():

    sample_sizes = 10000 * np.arange(1, 11)
    print(sample_sizes)

    n_features = 50
    random_state = 123

    batch_times = []
    batch_n_times = []
    naive_times = []

    for iteration, n_samples in enumerate(sample_sizes):
        print('\nIteration: {:.2f} %'.format(100 * (iteration+1) / len(sample_sizes)))

        # create data
        x_data, y_data = make_regression(n_samples=n_samples,
                                         n_features=n_features,
                                         random_state=random_state)

        # split data into training and testing
        train_size = 5000

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=train_size,
            random_state=random_state
        )

        # remove the mean from the training data
        y_mean = np.mean(y_train)

        y_train -= y_mean
        y_test -= y_mean

        # -------------------
        # NAIVE KERNEL MODEL
        # -------------------

        # initialize the kernel ridge regression model
        krr_model = KRR(batch_size=None,
                        calculate_variance=True)

        # fit model to data
        krr_model.fit(x_train, y_train)

        # PREDICTING TIMES
        # predict using the krr model
        start = time()
        _ = krr_model.predict(x_test)
        naive_time = time() - start

        print('Normal Predictions: {:.2f} secs'.format(naive_time))

        naive_times.append(naive_time)

        # BATCH PROCESSING
        # initialize the kernel ridge regression model
        batch_size = 5000

        krr_model = KRR(batch_size=batch_size,
                        calculate_variance=True)

        # fit model to data
        krr_model.fit(x_train, y_train)

        # PREDICTING TIMES
        # predict using the krr model
        start = time()
        _ = krr_model.predict(x_test)

        batch_time = time() - start

        print('Batch Predictions: {:.2f} secs'.format(batch_time))

        batch_times.append(batch_time)


        # Multi-Core BATCH PROCESSING
        # initialize the kernel ridge regression model
        batch_size = 5000
        n_jobs = 10

        krr_model = KRR(batch_size=batch_size, n_jobs=n_jobs,
                        calculate_variance=True)

        # fit model to data
        krr_model.fit(x_train, y_train)

        # PREDICTING TIMES
        # predict using the krr model
        start = time()
        _ = krr_model.predict(x_test)

        batch_n_time = time() - start
        print('Batch {} jobs, Predictions: {:.2f} secs'.format(n_jobs, batch_n_time))
        batch_n_times.append(batch_n_time)

    fig, ax = plt.subplots()

    ax.plot(sample_sizes, naive_times, color='k', label='Naive KRR')
    ax.plot(sample_sizes, batch_times, color='r', label='Batch KRR')
    ax.plot(sample_sizes, batch_n_times, color='g', label=str(n_jobs) + '-Core Batch KRR')

    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.title('Batch vs Regular KRR (sample, size)')
    plt.savefig('/media/disk/users/emmanuel/code/kernelib/test_batch.png')


    return None

def times_multi_exp_custom():

    sample_sizes = 100000 * np.arange(1, 11)
    print(sample_sizes)

    n_features = 50
    random_state = 123
    batch_size = 2000
    n_jobs = 10
    train_percent = 5000
    calculate_variance = False

    batch_n_times = []
    naive_sk_times = []
    sk_batch_n_times = []

    for iteration, n_samples in enumerate(sample_sizes):
        print('\nIteration: {:.2f} %'.format(100 * (iteration+1) / len(sample_sizes)))

        # create data
        x_data, y_data = make_regression(n_samples=n_samples,
                                         n_features=n_features,
                                         random_state=random_state)

        # split data into training and testing
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=train_percent,
            random_state=random_state
        )

        # remove the mean from the training data
        y_mean = np.mean(y_train)

        y_train -= y_mean
        y_test -= y_mean

        # -------------------
        # SKLEARN PREDICTIONS
        # -------------------
        krr_model = KernelRidge(alpha=1e-04,
                                gamma=np.mean(pdist(x_train, metric='euclidean')))

        # fit model to data
        krr_model.fit(x_train, y_train)

        # PREDICTING TIMES
        # predict using the krr model
        start = time()
        y_pred = krr_model.predict(x_test)
        naive_sk_time = time() - start

        print('Normal (sklearn) Predictions: {:.2f} secs'.format(naive_sk_time))

        naive_sk_times.append(naive_sk_time)

        error = mean_absolute_error(y_test, y_pred)
        print('\nMean Absolute Error: {:.4f}\n'.format(error))

        # -------------------------------------
        # MULTI-CORE BATCH PROCESSING (SKLEARN)
        # -------------------------------------
        krr_model = KernelRidge(alpha=1e-04,
                                gamma=np.mean(pdist(x_train, metric='euclidean')))

        # Fit Model to the Data
        krr_model.fit(x_train, y_train)

        # Prediction Times
        start = time()

        results = Parallel(n_jobs=n_jobs)(
            delayed(sklearn_krr_pred)(
                krr_model, x_test[start:end])
            for (start, end) in generate_batches(x_test.shape[0],
                                                 batch_size=batch_size)
        )

        predictions, variance = tuple(zip(*results))
        y_pred_mp = np.vstack(predictions)
        _ = np.vstack(variance)

        sk_batch_n_time = time() - start
        print('Batch {} jobs, Predictions: {:.2f} secs'.format(n_jobs, sk_batch_n_time))
        sk_batch_n_times.append(sk_batch_n_time)

        error_batch = mean_absolute_error(y_test, y_pred_mp)

        print('\nMean Absolute Error: {:.4f}\n'.format(error_batch))

        print('Errors are equal:', np.equal(error, error_batch))

    # Plot the results
    fig, ax = plt.subplots()

    ax.plot(sample_sizes, naive_sk_times, color='k',
            label='Naive (sklearn) KRR')
    ax.plot(sample_sizes, sk_batch_n_times, color='r',
            label=str(n_jobs) + '-Core Batch Sklearn ')

    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.title('Batch vs Regular KRR (sample, size)')
    plt.savefig('/media/disk/users/emmanuel/code/kernelib/test_batch.png')

    print(naive_sk_times)
    print(sk_batch_n_times)
    print(batch_n_times)

    return None

def times_multi_exp():

    sample_sizes = 100000 * np.arange(1, 11)
    print(sample_sizes)

    n_features = 50
    random_state = 123

    batch_times = []
    batch_n_times = []
    naive_times = []
    naive_sk_times = []

    for iteration, n_samples in enumerate(sample_sizes):
        print('\nIteration: {:.2f} %'.format(100 * (iteration+1) / len(sample_sizes)))

        # create data
        x_data, y_data = make_regression(n_samples=n_samples,
                                         n_features=n_features,
                                         random_state=random_state)

        # split data into training and testing
        train_percent = 5000

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=train_percent,
            random_state=random_state
        )

        # remove the mean from the training data
        y_mean = np.mean(y_train)

        y_train -= y_mean
        y_test -= y_mean

        # # -------------------
        # # NAIVE KERNEL MODEL
        # # -------------------
        #
        # # initialize the kernel ridge regression model
        # krr_model = KRR(batch_size=None)
        #
        # # fit model to data
        # krr_model.fit(x_train, y_train)
        #
        # # PREDICTING TIMES
        # # predict using the krr model
        # start = time()
        # _ = krr_model.predict(x_test)
        # naive_time = time() - start
        #
        # print('Normal Predictions: {:.2f} secs'.format(naive_time))
        #
        # naive_times.append(naive_time)

        # ---------------------------
        # scikit learn implementation
        # ---------------------------

        krr_model = KernelRidge(alpha=1e-04,
                                gamma=np.mean(pdist(x_train, metric='euclidean')))

        # fit model to data
        krr_model.fit(x_train, y_train)

        # PREDICTING TIMES
        # predict using the krr model
        start = time()
        _ = krr_model.predict(x_test)
        naive_sk_time = time() - start

        print('Normal (sklearn) Predictions: {:.2f} secs'.format(naive_sk_time))

        naive_sk_times.append(naive_sk_time)



        # # BATCH PROCESSING
        # # initialize the kernel ridge regression model
        # batch_size = 5000
        #
        # krr_model = KRR(batch_size=batch_size)
        #
        # # fit model to data
        # krr_model.fit(x_train, y_train)
        #
        # # PREDICTING TIMES
        # # predict using the krr model
        # start = time()
        # _ = krr_model.predict(x_test)
        #
        # batch_time = time() - start
        #
        # print('Batch Predictions: {:.2f} secs'.format(batch_time))
        #
        # batch_times.append(batch_time)


        # Multi-Core BATCH PROCESSING
        # initialize the kernel ridge regression model
        batch_size = 5000
        n_jobs = 20

        krr_model = KRR(batch_size=batch_size, n_jobs=n_jobs)

        # fit model to data
        krr_model.fit(x_train, y_train)

        # PREDICTING TIMES
        # predict using the krr model
        start = time()
        _ = krr_model.predict(x_test)

        batch_n_time = time() - start
        print('Batch {} jobs, Predictions: {:.2f} secs'.format(n_jobs, batch_n_time))
        batch_n_times.append(batch_n_time)

    fig, ax = plt.subplots()

    # ax.plot(sample_sizes, naive_times, color='k', label='Naive KRR')
    ax.plot(sample_sizes, naive_sk_times, color='k', label='Naive (sklearn) KRR')
    # ax.plot(sample_sizes, batch_times, color='r', label='Batch KRR')
    ax.plot(sample_sizes, batch_n_times, color='g', label=str(n_jobs) + '-Core Batch KRR')

    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.title('Batch vs Regular KRR (sample, size)')
    plt.savefig('/media/disk/users/emmanuel/code/kernelib/test_batch.png')


    return None

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

    # initialize the kernel ridge regression model
    krr_model = KRR(n_batches=1)

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

def test_batch_joblib():
    sample_sizes = 100000 * np.arange(1, 11)
    print(sample_sizes)
    random_state = 123
    n_features = 50

    # create data
    x_data, y_data = make_regression(n_samples=n_samples,
                                     n_features=n_features,
                                     random_state=random_state)

    # split data into training and testing
    train_percent = 0.1

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=train_percent,
        random_state=random_state
    )

    # remove the mean from the training data
    y_mean = np.mean(y_train)

    y_train -= y_mean
    y_test -= y_mean

    # NAIVE MODEL
    # initialize the kernel ridge regression model
    krr_model_naive = KRR()

    # fit model to data
    krr_model_naive.fit(x_train, y_train)

    # predict using the krr model
    start = time()
    y_pred_naive = krr_model_naive.predict(x_test)
    naive_time = time() - start

    print('Normal Predictions: {:.2f} secs'.format(naive_time))

    error = mean_absolute_error(y_test, y_pred_naive)
    print('\nMean Absolute Error: {:.4f}\n'.format(error))


    # initialize the kernel ridge regression model
    krr_model = KRR(batch_size=2000, n_jobs=20)

    # fit model to data
    krr_model.fit(x_train, y_train)

    # PREDICTING TIMES
    # predict using the krr model
    start = time()
    y_pred_batch = krr_model.predict(x_test)
    batch_time = time() - start

    print('Batch Predictions: {:.2f} secs'.format(batch_time))

    error_batch = mean_absolute_error(y_test, y_pred_batch)
    print('\nMean Absolute Error: {:.4f}\n'.format(error_batch))

    print('Errors are equal:', np.equal(error, error_batch))


    return None


def test_sklearn_joblib():


    sample_sizes = 100000
    random_state = 123
    n_features = 50
    n_jobs = 10
    train_percent = 5000
    batch_size = 2000

    # create data
    x_data, y_data = make_regression(n_samples=sample_sizes,
                                     n_features=n_features,
                                     random_state=random_state)

    # split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=train_percent,
        random_state=random_state
    )

    # remove the mean from the training data
    y_mean = np.mean(y_train)

    y_train -= y_mean
    y_test -= y_mean

    # ---------------------------
    # scikit learn implementation
    # ---------------------------

    krr_model = KernelRidge(alpha=1e-04,
                            gamma=np.mean(pdist(x_train, metric='euclidean')))

    # fit model to data
    krr_model.fit(x_train, y_train)

    # PREDICTING TIMES
    # predict using the krr model
    start = time()
    y_pred = krr_model.predict(x_test)
    naive_sk_time = time() - start

    print('Normal (sklearn) Predictions: {:.2f} secs'.format(naive_sk_time))

    error = mean_absolute_error(y_test, y_pred)
    print('\nMean Absolute Error: {:.4f}\n'.format(error))

    # -------------------------------------
    # MULTI-CORE BATCH PROCESSING (SKLEARN)
    # -------------------------------------
    krr_model = KernelRidge(alpha=1e-04,
                            gamma=np.mean(pdist(x_train, metric='euclidean')))

    # Fit Model to the Data
    krr_model.fit(x_train, y_train)

    # Prediction Times
    start = time()

    results = Parallel(n_jobs=n_jobs)(
        delayed(sklearn_krr_pred)(
            krr_model, x_test[start:end])
        for (start, end) in generate_batches(x_test.shape[0],
                                             batch_size=batch_size)
    )

    predictions, variance = tuple(zip(*results))
    y_pred_mp = np.vstack(predictions)
    _ = np.vstack(variance)

    sk_batch_n_time = time() - start
    print('Batch {} jobs, Predictions: {:.2f} secs'.format(n_jobs, sk_batch_n_time))

    error_batch = mean_absolute_error(y_test, y_pred_mp)

    print('\nMean Absolute Error: {:.4f}\n'.format(error_batch))

    print('Errors are equal:', np.equal(error, error_batch))


    return None



if __name__ == "__main__":
    times_multi_exp_custom()

