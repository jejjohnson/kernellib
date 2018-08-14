import numpy as np 
from sklearn.utils import check_X_y, check_array
from scipy.spatial.distance import pdist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

import scipy as scio

def estimate_sigma(X, Y=None, method='mean', verbose=0):
    """A function to provide a reasonable estimate of the sigma values
    for the RBF kernel using different methods.

    Parameters
    ----------
    X : array, (n_samples, d_dimensions)
        The data matrix to be estimated.

    Y : array, (n_samples, 1)
        The labels for the supervised approaches.

    method : str {'mean'} default: 'mean'

    Returns
    -------
    sigma : float
        The estimated sigma value

    Resources
    ---------
    - Original MATLAB function: https://goo.gl/xYoJce

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
           : juan.johnson@uv.es
    Date   : 6 - July - 2018
    """
    if Y:
        X, Y = check_X_y(X, Y)
    else:
        X = check_array(X)

    

    # subsampling
    [n_samples, d_dimensions] = X.shape

    if n_samples > 1000:
        n_samples = 1000
        X = np.random.permutation(X)[:n_samples, :]

        if Y:
            Y = np.random.permutation(Y)[:n_samples, :]


    # range of sigmas
    num_sigmas = 20
    sigmas = np.logspace(-3, 3, num_sigmas)

    if method == 'mean':
        sigma = np.mean(pdist(X) > 0)
    
    elif method == 'median':
        sigma = np.median(pdist(X) > 0)

    elif method == 'mode':
        raise NotImplementedError('Method "{}" has not been implemented yet.'.format(method))
        # sigma = scio.stats.mode(pdist(X) > 0)
    elif method == 'silverman':
        sigma = np.median( ((4/(d_dimensions + 2))**(1 / (d_dimensions + 4))) 
                          * n_samples**(-1 / (d_dimensions + 4)) * np.std(X, axis=0))
    elif method == 'scott':


        sigma = np.median( np.diag( n_samples**( - 1 / (d_dimensions + 4)) * np.cov(X)**(1/2)) ) 
    else:
        raise ValueError('Unrecognized mode "{}".'.format(method))


    return sigma


def r_assessment(y_pred, y_test, verbose=0):
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()

    mae = mean_absolute_error(y_pred, y_test)
    mse = mean_squared_error(y_pred, y_test)
    r2 = r2_score(y_pred, y_test)
    rmse = np.sqrt(mse)

    if verbose:
        print('MAE: {:.3f}'.format(mae))
        print('MSE: {:.3f}'.format(mse))
        print('RMSE: {:.3f}'.format(rmse))
        print('R2: {:.3f}'.format(r2))

    return None

def main():

    rng = np.random.RandomState(0)
    X = 5 * rng.rand(10000, 1)


    sigma = estimate_sigma(X, method='silverman')
    print(sigma)
    pass

if __name__ == '__main__':
    main()
