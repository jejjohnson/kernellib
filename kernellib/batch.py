from sklearn.datasets import make_regression
import numpy as np
from time import time
import warnings
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from kernellib.krr import KernelRidge
from kernellib.gp import GP_Simple
from scipy.spatial.distance import pdist
from sklearn.externals.joblib import Parallel, delayed
import matplotlib
from matplotlib import pyplot as plt
from kernellib.derivatives import rbf_derivative, ard_derivative

# TODO - Test Derivative and Variance Thoroughly
# TODO - Investigate Pre-Dispatch for joblib
# https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/model_selection/_search.py#L630
# TODO - Move Testing Procedure in Main function to testing function with smaller dataset
# TODO - Get logs instead of print statements
# TODO - Fix/Merge Time Experiment Into Main Script
# TODO - Add command-line arguments



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


def kernel_model_batch(x, kernel_model, batch_size=1000, return_variance=False,
                       return_derivative=False):

    # initialize the predicted values
    n_samples = x.shape[0]

    # predefine matrices
    if return_variance:
        variance = np.empty(shape=(n_samples, 1))
        predictions = variance.copy()
    else:
        predictions = np.empty(shape=(n_samples, 1))

    if return_derivative:
        derivative = np.empty(shape=x.shape)

    for start_idx, end_idx in generate_batches(n_samples, batch_size):

        ipred, ider, ivar = kernel_model_predictions(kernel_model, x[start_idx:end_idx], 
                                              return_derivative=return_derivative,
                                              return_variance=return_variance)
        # --------------------------
        # Predictive Variance
        # --------------------------
        if return_variance:
            predictions[start_idx:end_idx, :] = ipred
            variance[start_idx:end_idx, :] = ivar

        # --------------------------
        # Derivative
        # --------------------------
        if return_derivative:
            derivative[start_idx:end_idx, :] = ider

        # ---------------------------
        # Predictive Mean
        # ---------------------------
        if not return_variance:
            predictions[start_idx:end_idx, :] = ipred


    return predictions, derivative, variance


def kernel_model_parallel(x, kernel_model, n_jobs=10, batch_size=1000,
                 return_variance=False,
                 return_derivative=False,
                 verbose=10):

    if n_jobs > 1:
        # Perform parallel predictions using joblib
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(kernel_model_predictions)(
                kernel_model, x[start:end],
                return_variance=return_variance,
                return_derivative=return_derivative)
            for (start, end) in generate_batches(x.shape[0], batch_size=batch_size)
        )

        # Aggregate results (predictions, derivatives, variances)
        predictions, derivative, variance = tuple(zip(*results))
        predictions = np.vstack(predictions)
        derivative = np.vstack(derivative)
        variance = np.vstack(variance)

    elif n_jobs == 1:
        predictions, derivative, variance = \
            kernel_model_predictions(kernel_model, x, 
                              return_derivative=return_derivative,
                              return_variance=return_variance)
    else:
        raise ValueError('Unrecognized number of n_jobs...')

    return predictions, derivative, variance


def kernel_model_predictions(kernel_model, x, return_derivative=False,
                             return_variance=False, K_train_inv=None):

    # initialize the predicted values
    predictions = None
    derivative = None
    variance = None

    # ---------------------------------
    # Predictive Variance
    # ---------------------------------
    if return_variance:
        predictions, variance = kernel_model.predict(x, return_variance=True)
        variance = variance[:, np.newaxis]

    # ---------------------------------
    # Derivative
    # ---------------------------------
    if return_derivative:
        derivative = ard_derivative(x_train=kernel_model.x_train,
                                    x_test=x,
                                    weights=kernel_model.weights_,
                                    length_scale=kernel_model.length_scale,
                                    scale=kernel_model.scale,
                                    n_der=1)

    # ------------------------
    # Predictive Mean
    # ------------------------
    if not return_variance:
        predictions = kernel_model.predict(x)

    return predictions, derivative, variance


def gp_model_parallel(x, gp_model, n_jobs=10, batch_size=100,
                       return_variance=False,
                       return_derivative=False, 
                       verbose=1):
    
    if n_jobs > 1:
        # Perform parallel predictions using joblib
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(gp_model_predictions)(
                gp_model, x[start:end],
                return_variance=return_variance,
                return_derivative=return_derivative)
            for (start, end) in generate_batches(x.shape[0], batch_size=batch_size)

        )

        # Aggregate results (predictions, derivatives, variances)
        predictions, derivative, variance = tuple(zip(*results))
        predictions = np.vstack(predictions)
        derivative = np.vstack(derivative)
        variance = np.vstack(variance)
    
    elif n_jobs == 1:
        predictions, derivative, variance = gp_model_predictions(
            gp_model, x, 
            return_derivative=return_derivative,
            return_variance=return_variance)
        
    return predictions, derivative, variance


def gp_model_predictions(gp_model, x, 
                         return_derivative=False,
                         return_variance=False):
    
    # initialize the output values
    predictions = None
    derivative = None
    variance = None
    
    # --------------------
    # Predictive Variance
    # --------------------
    if return_variance:
        predictions, variance = gp_model.predict(x, return_std=True)
        predictions = predictions[:, np.newaxis]
        variance = variance[:, np.newaxis]
    
    # --------------------
    # Derivatives
    # --------------------
    if return_derivative:
        derivative = ard_derivative(x_train=gp_model.x_train,
                                    x_test=x,
                                    weights=gp_model.weights_[:, np.newaxis],
                                    length_scale=gp_model.length_scale,
                                    scale=gp_model.scale,
                                    n_der=1)
        
    # ----------------
    # Predictive Mean
    # ----------------
    if not return_variance:
        predictions = gp_model.predict(x)[:, np.newaxis]
    return predictions, derivative, variance


def times_multi_exp():

    sample_sizes = 10000 * np.arange(1, 10)

    sample_sizes = 100000 * np.arange(1, 11)
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


        # BATCH PROCESSING
        # initialize the kernel ridge regression model
        n_samples_per_batch = 5000
        n_batches = int(np.round(n_samples / n_samples_per_batch))
        y_pred = krr_model.predict(x_test)


        naive_time = time() - start

        print('Normal Predictions: {:.2f} secs'.format(naive_time))

        naive_times.append(naive_time)

        # BATCH PROCESSING
        # initialize the kernel ridge regression model
        batch_size = 1000

        # PREDICTING TIMES
        # predict using the krr model
        start = time()

        # Multi-Core BATCH PROCESSING
        # initialize the kernel ridge regression model
        n_samples_per_batch = 5000
        n_batches = int(np.round(n_samples / n_samples_per_batch))
        print(n_batches)
        n_jobs = 30
        ypred, _, _ = krr_batch(x=x_test,
                                krr_model=krr_model,
                                batch_size=batch_size,
                                calculate_predictions=True,
                                calculate_variance=False,
                                calculate_derivative=False)

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

def gp_test():
    print('Starting main script...')

    sample_sizes = 3000
    random_state = 123
    n_features = 50
    n_jobs = 4
    train_percent = 0.1
    batch_size = 500
    calculate_variance = True
    calculate_derivative = True

    print('Calculating variance: {}'.format(str(calculate_variance)))
    print('Calculating derivative: {}'.format(str(calculate_derivative)))

    # create data
    x_data, y_data = make_regression(n_samples=sample_sizes,
                                     n_features=n_features,
                                     random_state=random_state)

    # split data into training and testing
    warnings.simplefilter("ignore")
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=train_percent,
        random_state=random_state
    )
    warnings.simplefilter("default")

    # remove the mean from the training data
    y_mean = np.mean(y_train)

    y_train -= y_mean
    y_test -= y_mean
    length_scale = np.mean(pdist(x_train, metric='euclidean'))
    sigma_y = 1e-04
    # -------------------------------
    # My GP Implementation
    # -------------------------------
    gp_model = GP_Simple(sigma_y=sigma_y, length_scale=length_scale)

    gp_model.fit(x_train, y_train)

     # -----------------------------
    # NAIVE PREDICT (SKLEARN)
    # -----------------------------
    print('Predicting using naive Scikit-Learn function...')

    # predict using the naive krr model
    start = time()

    y_pred, der, var = kernel_model_predictions(gp_model, x_test,
                                                return_derivative=calculate_derivative,
                                                return_variance=calculate_variance)

    naive_sk_time = time() - start

    print('Normal Predictions: {:.2f} secs'.format(naive_sk_time))

    error = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error: {:.4f}'.format(error))

    # -------------------------------------
    # BATCH PROCESSING (SKLEARN)
    # -------------------------------------
    print('\nPredicting using batch method...')

    # Prediction Times
    start = time()

    ypred_batch, der_batch, var_batch = kernel_model_batch(x=x_test,
                            kernel_model=gp_model,
                            batch_size=batch_size,
                            return_variance=calculate_variance,
                            return_derivative=calculate_derivative)

    sk_batch_time = time() - start
    print('Batch Predictions: {:.2f} secs'.format(sk_batch_time))

    error_batch = mean_absolute_error(y_test, ypred_batch)

    np.testing.assert_almost_equal(error, error_batch, err_msg='Batch MSE Error are no equal')
    np.testing.assert_array_almost_equal(ypred_batch, y_pred, err_msg='Batch Predictions are not equal...')
    np.testing.assert_array_almost_equal(var_batch, var, err_msg='Batch Variances are not equal...')
    np.testing.assert_array_almost_equal(der_batch, der, err_msg='Batch Derivatives are not equal...')

    print('Speedup: x{:.2f}'.format(sk_batch_time / naive_sk_time))

    # -------------------------------------
    # MULTI-CORE BATCH PROCESSING (SKLEARN)
    # -------------------------------------
    print('\nPredicting using batches with {} cores...'.format(n_jobs))

    # Prediction Times
    start = time()

    ypred_mp, der_mp, var_mp = kernel_model_parallel(x=x_test,
                               kernel_model=gp_model,
                               n_jobs=n_jobs,
                               batch_size=batch_size,
                               return_variance=calculate_variance,
                               return_derivative=calculate_derivative,
                               verbose=1)

    sk_batch_n_time = time() - start
    print('Batch {} jobs, Predictions: {:.2f} secs'.format(n_jobs, sk_batch_n_time))

    error_mp = mean_absolute_error(y_test, ypred_mp)

    np.testing.assert_almost_equal(error, error_mp, err_msg='Batch Cores MSE Error are no equal')
    np.testing.assert_array_almost_equal(ypred_mp, y_pred, err_msg='Batch Cores Predictions are not equal...')
    np.testing.assert_array_almost_equal(var_mp, var, err_msg='Batch Cores Variances are not equal...')
    np.testing.assert_array_almost_equal(der_mp, der, err_msg='Batch Cores Derivatives are not equal...')

    print('Speedup (naive): x{:.2f}'.format(naive_sk_time / sk_batch_n_time))
    print('Speedup (batch): x{:.2f}'.format(sk_batch_time / sk_batch_n_time))   


    return None

def krr_test():
    print('Starting main script...')

    sample_sizes = 10000
    random_state = 123
    n_features = 50
    n_jobs = 4
    train_percent = 0.1
    batch_size = 500
    calculate_variance = True
    calculate_derivative = True

    print('Calculating variance: {}'.format(str(calculate_variance)))
    print('Calculating derivative: {}'.format(str(calculate_derivative)))

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
    length_scale = np.mean(pdist(x_train, metric='euclidean'))
    sigma_y = 1e-04
    # ---------------------------
    # My KRR implementation
    # ---------------------------

    krr_model = KernelRidge(sigma_y=sigma_y,
                            length_scale=length_scale)

    # fit model to data
    krr_model.fit(x_train, y_train)

    # -----------------------------
    # NAIVE PREDICT (SKLEARN)
    # -----------------------------
    print('Predicting using naive Scikit-Learn function...')

    # predict using the naive krr model
    start = time()

    y_pred, der, var = kernel_model_predictions(krr_model, x_test,
                                         return_derivative=calculate_derivative,
                                         return_variance=calculate_variance)

    naive_sk_time = time() - start

    print('Normal (sklearn) Predictions: {:.2f} secs'.format(naive_sk_time))

    error = mean_absolute_error(y_test, y_pred)
    print('Mean Absolute Error: {:.4f}'.format(error))

    # -------------------------------------
    # BATCH PROCESSING (SKLEARN)
    # -------------------------------------
    print('\nPredicting using batch method...')

    # Prediction Times
    start = time()

    ypred_batch, der_batch, var_batch = kernel_model_batch(x=x_test,
                            kernel_model=krr_model,
                            batch_size=batch_size,
                            return_variance=calculate_variance,
                            return_derivative=calculate_derivative)

    sk_batch_time = time() - start
    print('Batch Predictions: {:.2f} secs'.format(sk_batch_time))

    error_batch = mean_absolute_error(y_test, ypred_batch)

    np.testing.assert_almost_equal(error, error_batch, err_msg='Batch MSE Error are no equal')
    np.testing.assert_array_almost_equal(ypred_batch, y_pred, err_msg='Batch Predictions are not equal...')
    np.testing.assert_array_almost_equal(var_batch, var, err_msg='Batch Variances are not equal...')
    np.testing.assert_array_almost_equal(der_batch, der, err_msg='Batch Derivatives are not equal...')

    print('Speedup: x{:.2f}'.format(naive_sk_time / sk_batch_time))

    # -------------------------------------
    # MULTI-CORE BATCH PROCESSING (SKLEARN)
    # -------------------------------------
    print('\nPredicting using batches with {} cores...'.format(n_jobs))

    # Prediction Times
    start = time()

    ypred_mp, der_mp, var_mp = kernel_model_parallel(x=x_test,
                               kernel_model=krr_model,
                               n_jobs=n_jobs,
                               batch_size=batch_size,
                               return_variance=calculate_variance,
                               return_derivative=calculate_derivative,
                               verbose=1)

    sk_batch_n_time = time() - start
    print('Batch {} jobs, Predictions: {:.2f} secs'.format(n_jobs, sk_batch_n_time))

    error_mp = mean_absolute_error(y_test, ypred_mp)

    np.testing.assert_almost_equal(error, error_mp, err_msg='Batch Cores MSE Error are no equal')
    np.testing.assert_array_almost_equal(ypred_mp, y_pred, err_msg='Batch Cores Predictions are not equal...')
    np.testing.assert_array_almost_equal(var_mp, var, err_msg='Batch Cores Variances are not equal...')
    np.testing.assert_array_almost_equal(der_mp, der, err_msg='Batch Cores Derivatives are not equal...')

    print('Speedup (naive): x{:.2f}'.format(naive_sk_time / sk_batch_n_time))
    print('Speedup (batch): x{:.2f}'.format(sk_batch_time / sk_batch_n_time))

    return None

def main():

    # krr_test()
    gp_test()

    return None


if __name__ == "__main__":
    main()
