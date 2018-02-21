import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist

# TODO: Write tests for derivative functions, gradients
# TODO: Implement Derivative w/ 1 loop for memory conservation
# TODO: Implement 2nd Derivative for all
# TODO: Do Derivative for other kernel methods (ARD, Polynomial)


def ard_kernel(x, y, gamma):
    
    dists = cdist(x / gamma, y / gamma, metric='sqeuclidean')
    
    K = np.exp(-.5 * dists)
    
    return K

def ard_derivative(x_train, x_function, weights, gamma):
    
    kernel_mat = ard_kernel(x_function, x_train, gamma)
    
    num_test = x_function.shape[0]
    
    derivative = np.zeros(shape=x_function.shape)
    
    for itest in range(num_test):
        
        derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T,
                            (kernel_mat[itest, :] * weights).T)
        
    derivative *= -1 / gamma ** 2
    
    return derivative


def rbf_derivative(x_train, x_function, weights, gamma):
    
    kernel_mat = rbf_kernel(x_function, x_train, gamma=gamma)
    
    n_test = x_function.shape[0]
    
    derivative = np.zeros(shape=x_function.shape)
    
    for itest in range(n_test):
        derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T, 
                            (kernel_mat[itest, :] * weights).T)
    
    derivative *= -2 * gamma
    
    return derivative


def rbf_derivative_slow(x_train, x_function, weights, kernel_mat=None,
                   n_derivative=1, gamma=1.0):
    """This function calculates the rbf derivative
    Parameters
    ----------
    x_train : array, [N x D]
        The training data used to find the kernel model.

    x_function  : array, [M x D]
        The test points (or vector) to use.

    weights   : array, [N x D]
        The weights found from the kernel model
            y = K * weights

    kernel_mat: array, [N x M], default: None
        The rbf kernel matrix with the similarities between the test
        points and the training points.

    n_derivative : int, (default = 1) {1, 2}
        chooses which nth derivative to calculate

    gamma : float, default: None
        the parameter for the rbf_kernel matrix function

    Returns
    -------

    derivative : array, [M x D]
        returns the derivative with respect to training points used in
        the kernel model and the test points.

    Information
    -----------
    Author: Juan Emmanuel Johnson
    Email : jej2744@rit.edu
            juan.johnson@uv.es
    """

    # initialize rbf kernel
    derivative = np.zeros(np.shape(x_function))

    # check for kernel mat
    if kernel_mat is None:
        kernel_mat = rbf_kernel(x_train, x_function, gamma=gamma)

    # consolidate the parameters
    theta = 2 * gamma

    # 1st derivative
    if n_derivative == 1:

        # loop through dimensions
        for dim in np.arange(0, np.shape(x_function)[1]):

            # loop through the number of test points
            for iTest in np.arange(0, np.shape(x_function)[0]):

                # loop through the number of test points
                for iTrain in np.arange(0, np.shape(x_train)[0]):

                    # calculate the derivative for the test points
                    derivative[iTest, dim] += theta * weights[iTrain] * \
                                              (x_train[iTrain, dim] -
                                               x_function[iTest, dim]) * \
                                              kernel_mat[iTrain, iTest]

    # 2nd derivative
    elif n_derivative == 2:

        # loop through dimensions
        for dim in np.arange(0, np.shape(x_function)[1]):

            # loop through the number of test points
            for iTest in np.arange(0, np.shape(x_function)[0]):

                # loop through the number of test points
                for iTrain in np.arange(0, np.shape(x_train)[0]):
                    derivative[iTest, dim] += weights[iTrain] * \
                                              (theta ** 2 *
                                               (x_train[iTrain, dim] - x_function[iTest, dim]) ** 2
                                               - theta) * \
                                              kernel_mat[iTrain, iTest]

    return derivative


def rbf_derivative_memory(x_train, x_function, weights, gamma, n_derivative=1):
    """This function calculates the rbf derivative using no
    loops but it requires a large memory load.

    Parameters
    ----------
    x_train : array, [N x D]
        The training data used to find the kernel model.

    x_function  : array, [M x D]
        The test points (or vector) to use.

    weights   : array, [N x D]
        The weights found from the kernel model
            y = K * weights

    kernel_mat: array, [N x M], default: None
        The rbf kernel matrix with the similarities between the test
        points and the training points.

    n_derivative : int, (default = 1) {1, 2}
        chooses which nth derivative to calculate

    gamma : float, default: None
        the parameter for the rbf_kernel matrix function

    Returns
    -------

    derivative : array, [M x D]
        returns the derivative with respect to training points used in
        the kernel model and the test points.

    Information
    -----------
    Author: Juan Emmanuel Johnson
    Email : jej2744@rit.edu
            juan.johnson@uv.es
    """
    n_train_samples = x_train.shape[0]
    n_test_samples = x_function.shape[0]
    n_dimensions = x_train.shape[1]
    
    kernel_mat = rbf_kernel(x_train, x_function, gamma=gamma)

    # create empty derivative matrix
    derivative = np.empty(shape=(n_train_samples,
                                 n_test_samples,
                                 n_dimensions))

    # create empty block matrices and sum
    derivative = np.tile(weights[:, np.newaxis, np.newaxis],
                           (1, n_test_samples, n_dimensions)) * \
                      (np.tile(x_function[np.newaxis, :, :],
                              (n_train_samples, 1, 1)) - \
                      np.tile(x_train[:, np.newaxis, :],
                           (1, n_test_samples, 1))) * \
                      np.tile(kernel_mat[:, :, np.newaxis],
                              (1, 1, n_dimensions))

    # TODO: Write code for 2nd Derivative
    # multiply by the constant
    derivative *= -2 * gamma

    # sum all of the training samples to get M x N matrix
    derivative = derivative.sum(axis=0).squeeze()

    return derivative


def main():

    from sklearn.kernel_ridge import KernelRidge
    import numpy as np
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    y = rng.randn(n_samples)
    x = rng.randn(n_samples, n_features)

    lam = 1.0
    gamma = 1.0

    print('Initializing Model...')
    krr_model = KernelRidge(kernel='rbf',
                            alpha=lam,
                            gamma=gamma)

    print('Fitting kernel model...')
    krr_model.fit(x, y)

    print(krr_model)

    weights = krr_model.dual_coef_


    return None


if __name__ == "__main__":
    main()
