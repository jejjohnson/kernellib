from sklearn.datasets import make_regression
import numpy as np
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
from joblib import Parallel, delayed

# TODO - Test Derivative
# TODO - Test Variance

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

    def __init__(self, sigma=None, lam=None, calculate_variance=False):
        self.sigma = sigma
        self.lam = lam
        self.calculate_variance=calculate_variance

    def fit(self, x, y=None):

        # regularization trade off parameter
        if self.lam is None:

            # common heuristic for minimizing the lambda value
            self.lam = 1.0e-4

        # kernel length scale parameter
        if self.sigma is None:

            # common heuristic for finding the sigma value
            self.sigma = np.mean(pdist(x, metric='euclidean'))

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

        # calculate the kernel function with new points
        K_traintest = rbf_kernel(X=self.X_fit_, Y=x, gamma=self.gamma)

        # calculate the predictions
        predictions = np.dot(K_traintest.T, self.weights_)

        if self.calculate_variance is True:
            K_test = rbf_kernel(x, gamma=self.gamma)

            self.variance_ = np.diag(K_test) - \
                             np.diag(
                                 np.dot(K_traintest.T,
                                        np.dot(self.K_inverse_,
                                               K_traintest)))

        # return the project points
        return predictions


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


def krr_batch(x, krr_model, batch_size=1000,
              calculate_predictions=True,
              calculate_derivative=False,
              calculate_variance=False):

    # initialize the predicted values
    n_samples = x.shape[0]

    # predefine matrices
    if calculate_predictions:
        predictions = np.empty(shape=(n_samples, 1))
    else:
        predictions = None

    if calculate_derivative:
        derivative = np.empty(shape=x)
    else:
        derivative = None

    if calculate_variance:

        K_train = rbf_kernel(krr_model.X_fit_,
                             gamma=krr_model.gamma)
        K_train_inverse = np.linalg.inv(K_train)
        variance = np.empty(shape=(n_samples, 1))
    else:
        variance = None

    for start_idx, end_idx in generate_batches(n_samples, batch_size):

        if calculate_predictions:

            # calculate the predictions
            predictions[start_idx:end_idx, 0] = \
                krr_model.predict(x[start_idx:end_idx])

        if calculate_derivative:

            K_traintest = rbf_kernel(krr_model.X_fit_,
                                     x[start_idx:end_idx],
                                     gamma=krr_model.gamma)

            # calculate the derivative
            derivative[start_idx:end_idx, :] = \
                rbf_derivative_memory(x_train=np.float64(krr_model.X_fit_),
                                    x_function = np.float64(x[start_idx:end_idx]),
                                    kernel_mat = K_traintest,
                                    weights = krr_model.dual_coef_.squeeze(),
                                    gamma = np.float(krr_model.gamma),
                                    n_derivative = int(1))

        if calculate_variance:

            # calculate the Kbatch
            K_batch = rbf_kernel(x[start_idx:end_idx],
                                 gamma=krr_model.gamma)

            # calculate the variance
            variance[start_idx:end_idx, 0] = np.diag(K_batch) - \
                np.diag(np.dot(K_traintest.T,
                               np.dot(K_train_inverse, K_traintest)))

    return predictions, derivative, variance


def krr_parallel(x, krr_model, n_jobs=10, batch_size=1000,
                 calculate_predictions=True,
                 calculate_derivative=False,
                 calculate_variance=False,
                 verbose=10):

    # calculate the inverse transform
    if calculate_variance:
        K_train_inv = np.linalg.inv(rbf_kernel(X=krr_model.X_fit_,
                                               gamma=krr_model.gamma))
    else:
        K_train_inv = None

    # Perform parallel predictions using joblib
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(krr_predictions)(
            krr_model, x[start:end],
            K_train_inv=K_train_inv,
            calculate_predictions=calculate_predictions,
            calculate_derivative=calculate_derivative,
            calculate_variance=calculate_variance)
        for (start, end) in generate_batches(x.shape[0],
                                             batch_size=batch_size)
    )

    # Aggregate results (predictions, derivatives, variances)
    predictions, derivative, variance = tuple(zip(*results))
    predictions = np.vstack(predictions)
    derivative = np.vstack(derivative)
    variance = np.vstack(variance)


    return predictions, derivative, variance


def krr_predictions(KRR_Model, x, calculate_predictions=True,
                    calculate_derivative=False,
                    calculate_variance=False,
                    K_train_inv=None):

    # initialize the predicted values
    predictions = None
    derivative = None
    variance = None

    if calculate_predictions:
        # calculate the predictions
        predictions = KRR_Model.predict(x)

        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]

    if calculate_derivative or calculate_variance:
        # calculate train-test kernel
        K_traintest = rbf_kernel(KRR_Model.X_fit_,
                                 x, gamma=KRR_Model.gamma)

    if calculate_derivative:

        # calculate the derivative
        derivative = rbf_derivative_memory(x_train=np.float64(KRR_Model.X_fit_),
                                    x_function=np.float64(x),
                                    kernel_mat=K_traintest,
                                    weights=KRR_Model.dual_coef_.squeeze(),
                                    gamma=np.float(KRR_Model.gamma),
                                    n_derivative=int(1))

    if calculate_variance:

        # calculate the kernel for test points
        K_test = rbf_kernel(x, gamma=KRR_Model.gamma)

        # calculate K_traininverse if necessary
        if K_train_inv is None:
            K_train_inv = np.linalg.inv(rbf_kernel(x, gamma=KRR_Model.gamma))

        # calculate the variance
        variance = np.diag(K_test) - \
                   np.diag(np.dot(K_traintest.T, np.dot(K_train_inv,
                                                        K_traintest)))
        if variance.ndim == 1:
            variance = variance[:, np.newaxis]

    return predictions, derivative, variance


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


def times_multi_exp():

<<<<<<< HEAD
    sample_sizes = 10000 * np.arange(1, 10)
=======
    sample_sizes = 100000 * np.arange(1, 11)
>>>>>>> batch_processing
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
        print(x_train.shape, x_test.shape)
        # remove the mean from the training data
        y_mean = np.mean(y_train)

        y_train -= y_mean
        y_test -= y_mean

        # initialize the kernel ridge regression model
        krr_model = KernelRidge(alpha=1e-04,
                                gamma=np.mean(pdist(x_train, metric='euclidean')))

        # fit model to data
        krr_model.fit(x_train, y_train)

        # -------------------
        # NAIVE KERNEL MODEL
        # -------------------

        # PREDICTING TIMES
        # predict using the krr model
        start = time()

<<<<<<< HEAD
        # BATCH PROCESSING
        # initialize the kernel ridge regression model
        n_samples_per_batch = 5000
        n_batches = int(np.round(n_samples / n_samples_per_batch))
=======
        y_pred = krr_model.predict(x_test)
>>>>>>> batch_processing

        naive_time = time() - start

        print('Normal Predictions: {:.2f} secs'.format(naive_time))

        naive_times.append(naive_time)

        # BATCH PROCESSING
        # initialize the kernel ridge regression model
        batch_size = 1000

        # PREDICTING TIMES
        # predict using the krr model
        start = time()

<<<<<<< HEAD
        # Multi-Core BATCH PROCESSING
        # initialize the kernel ridge regression model
        n_samples_per_batch = 5000
        n_batches = int(np.round(n_samples / n_samples_per_batch))
        print(n_batches)
        n_jobs = 30
=======
        ypred, _, _ = krr_batch(x=x_test,
                                krr_model=krr_model,
                                batch_size=batch_size,
                                calculate_predictions=True,
                                calculate_variance=False,
                                calculate_derivative=False)
>>>>>>> batch_processing

        batch_time = time() - start

        print('Batch Predictions: {:.2f} secs'.format(batch_time))

        batch_times.append(batch_time)

        # -------------------------------------
        # MULTI-CORE BATCH PROCESSING (SKLEARN)
        # -------------------------------------

        # initialize the kernel ridge regression model
        batch_size = 1000
        n_jobs = 16

        # PREDICTING TIMES
        # predict using the krr model
        start = time()

        ypred, _, _ = krr_parallel(x=x_test,
                                   krr_model=krr_model,
                                   n_jobs=n_jobs,
                                   batch_size=batch_size,
                                   calculate_predictions=True,
                                   calculate_variance=False,
                                   calculate_derivative=False,
                                   verbose=0)

        batch_n_time = time() - start
        print('Batch {} jobs, Predictions: {:.2f} secs'.format(n_jobs, batch_n_time))
        batch_n_times.append(batch_n_time)

    fig, ax = plt.subplots()

    ax.plot(sample_sizes, naive_times, color='k', label='KRR')
    ax.plot(sample_sizes, batch_times, color='r', label='Batch KRR')
    ax.plot(sample_sizes, batch_n_times, color='g', label=str(n_jobs) + '-Core Batch KRR')

    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.title('Batch vs Regular KRR (sample, size)')
    fig.savefig('/media/disk/users/emmanuel/code/kernelib/test_batch.png')


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


def test_sklearn_joblib():

    sample_sizes = 900000
    random_state = 123
    n_features = 50
    n_jobs = 16
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

    # -----------------------------
    # NAIVE PREDICT (SKLEARN)
    # -----------------------------

    # PREDICTING TIMES
    # predict using the krr model
    start = time()
    y_pred = krr_model.predict(x_test)
    naive_sk_time = time() - start

    print('Normal (sklearn) Predictions: {:.2f} secs'.format(naive_sk_time))

    error = mean_absolute_error(y_test, y_pred)
    print('\nMean Absolute Error: {:.4f}\n'.format(error))

    # -------------------------------------
    # BATCH PROCESSING (SKLEARN)
    # -------------------------------------
    # Prediction Times
    start = time()

    ypred, _, _ = krr_batch(x=x_test,
                            krr_model=krr_model,
                            batch_size=batch_size,
                            calculate_predictions=True,
                            calculate_variance=False,
                            calculate_derivative=False)

    sk_batch_time = time() - start
    print('Batch Predictions: {:.2f} secs'.format(sk_batch_time))

    error_batch = mean_absolute_error(y_test, ypred)

    print('\nMean Absolute Error: {:.4f}\n'.format(error_batch))

    print('Errors are equal:', np.equal(error, error_batch))

    # -------------------------------------
    # MULTI-CORE BATCH PROCESSING (SKLEARN)
    # -------------------------------------

    # Prediction Times
    start = time()

    ypred, _, _ = krr_parallel(x=x_test,
                               krr_model=krr_model,
                               n_jobs=n_jobs,
                               batch_size=batch_size,
                               calculate_predictions=True,
                               calculate_variance=False,
                               calculate_derivative=False,
                               verbose=10)

    sk_batch_n_time = time() - start
    print('Batch {} jobs, Predictions: {:.2f} secs'.format(n_jobs, sk_batch_n_time))

    error_batch = mean_absolute_error(y_test, ypred)

    print('\nMean Absolute Error: {:.4f}\n'.format(error_batch))

    print('Errors are equal:', np.equal(error, error_batch))


    return None



if __name__ == "__main__":
    main()

