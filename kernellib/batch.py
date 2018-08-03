import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from kernellib.derivatives import ard_derivative

# TODO - Test Derivative and Variance Thoroughly
# TODO - Investigate Pre-Dispatch for joblib
# https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/model_selection/_search.py#L630
# TODO - Move Testing Procedure in Main function to testing function with smaller dataset
# TODO - Get logs instead of print statements
# TODO - Fix/Merge Time Experiment Into Main Script


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


def kernel_model_batch(x, kernel_model, batch_size=1000,
                       return_derivative=False):

    # initialize the predicted values
    n_samples = x.shape[0]

    # predefine matrices
    predictions = np.empty(shape=(n_samples, 1))

    if return_derivative:
        derivative = np.empty(shape=x.shape)

    for start_idx, end_idx in generate_batches(n_samples, batch_size):

        ipred, ider = kernel_model_predictions(kernel_model, x[start_idx:end_idx],
                                               return_derivative=return_derivative)


        # --------------------------
        # Derivative
        # --------------------------
        if return_derivative:
            derivative[start_idx:end_idx, :] = ider

        # ---------------------------
        # Predictive Mean
        # ---------------------------
        predictions[start_idx:end_idx, :] = ipred



    return predictions, derivative


def kernel_model_parallel(x, kernel_model, n_jobs=10, batch_size=1000,
                          return_derivative=False,
                          verbose=10):

    if n_jobs > 1:
        # Perform parallel predictions using joblib
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(kernel_model_predictions)(
                kernel_model, x[start:end],
                return_derivative=return_derivative)
            for (start, end) in generate_batches(x.shape[0], batch_size=batch_size)
        )

        # Aggregate results (predictions, derivatives, variances)
        predictions, derivative = tuple(zip(*results))
        predictions = np.vstack(predictions)
        derivative = np.vstack(derivative)

    elif n_jobs == 1:
        predictions, derivative = \
            kernel_model_predictions(kernel_model, x, 
                              return_derivative=return_derivative)
    else:
        raise ValueError('Unrecognized number of n_jobs...')

    return predictions, derivative


def kernel_model_predictions(kernel_model, x, return_derivative=False):

    # initialize the predicted values
    predictions = None
    derivative = None


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
    predictions = kernel_model.predict(x)

    return predictions, derivative


def gp_model_batch(x, kernel_model, batch_size=1000,
                   return_derivative=False,
                   return_variance=False):

    # initialize the predicted values
    n_samples = x.shape[0]

    # predefine matrices
    if return_variance:
        variance = np.empty(shape=(n_samples, 1))
        predictions = np.empty(shape=(n_samples, 1))
    else:
        predictions = np.empty(shape=(n_samples, 1))

    if return_derivative:
        derivative = np.empty(shape=x.shape)

    for start_idx, end_idx in generate_batches(n_samples, batch_size):

        ipred, ider, ivar = gp_model_predictions(
            kernel_model, x[start_idx:end_idx],
            return_derivative=return_derivative,
            return_variance=return_variance
        )

        # -------------------------
        # Variance
        # -------------------------
        if return_variance:
            variance[start_idx:end_idx, :] = ivar

        # --------------------------
        # Derivative
        # --------------------------
        if return_derivative:
            derivative[start_idx:end_idx, :] = ider

        # ---------------------------
        # Predictive Mean
        # ---------------------------
        predictions[start_idx:end_idx, :] = ipred



    return predictions, derivative


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


