import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances
from kernellib.kernels import rbf_kernel
from sklearn.gaussian_process.kernels import (_check_length_scale)

# TODO: Write tests for derivative functions, gradients
# TODO: Implement Derivative w/ 1 loop for memory conservation
# TODO: Implement 2nd Derivative for all
# TODO: Do Derivative for other kernel methods (ARD, Polynomial)


def ard_kernel(x, y=None, length_scale=None, scale=None, weighted=None, x_cov=None):
    """The Automatic Relevance Determination Kernel.
    
    Parameters
    ----------
    x : array-like, (n_samples x n_dimensions)
        An array for the left argument for the returned kernel K(X,Y)
        
    y : array-like, optional (n_samples x n_dimensions), default = None
        The right argument for the returned kernel K(X, Y). If none, 
        K(X, X) is evaulated instead.
        
    length_scale : array (n_dimensions), default: 1.0
        The length scale of the kernel. 
        
    scale : float, default: 1.0
        The vertical scale relative to the zero mean of the process in the 
        output space.
        
    Returns
    -------
    K : array, (n_samples x n_samples)
        The kernel matrix for K(X,Y) or K(X,X)
        
    Information
    -----------
    Author : Juan Emmanuel Johnson
    
    References
    ----------
    Scikit-Learn (RBF Kernel): https://goo.gl/Sz5icv
    """
    # check if x, y have the same shape
    if y is not None:
        x, y = check_pairwise_arrays(x, y)
    
    # grab samples and dimensions
    n_samples, n_dimensions = x.shape
    
    # get default sigma values
    if length_scale is None:
        length_scale = np.ones(shape=n_dimensions)
    else:
        length_scale = _check_length_scale(x, length_scale)

        
    # get default scale value
    if scale is None:
        scale = 1.0
    
    # compute the euclidean distances
    if y is None:
        dists = pdist(x / length_scale, metric='sqeuclidean')
        
        K = np.exp(-.5 * dists)
        
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)
        
    else:
        dists = cdist(x / length_scale, y / length_scale, metric='sqeuclidean')
        
        # exponentiate the distances
        K = scale * np.exp( -0.5 * dists)
        
    return K


def ard_derivative(x_train, x_test, weights, length_scale, scale=None, n_der=1):
    """Derivative of the GP mean function of the ARD Kernel. This function 
    computes the derivative of the mean function that has been trained with an
    ARD kernel with respect to the testing points.
    
    Parameters
    ----------
    x_train : array-like, (n_train_samples x d_dimensions)
        The training samples used to train the weights and the length scale 
        parameters.
        
    x_test : array-like, (n_test_samples x d_dimensions)
        The test samples that will be used to compute the derivative.
        
    weights : array-like, (n_train_samples, 1)
        The weights used from the training samples
        
    length_scale : array, (d_dimensions)
        The length scale for the ARD kernel. This includes a sigma value
        for each dimension.
    
    n_der : int, default: 1, ('1', '2')
        The nth derivative for the mean GP/KRR function with the ARD kernel
        
    Returns
    -------
    derivative : array-like, (n_test_samples x d_dimensions)
        The computed derivative.
        
    Information
    -----------
    Author : Juan Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    
    References
    ----------
    Differenting GPs:
        http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf
    """
    
    # check the sizes of x_train and x_test
    err_msg = "xtrain and xtest d dimensions are not equivalent."
    np.testing.assert_equal(x_test.shape[1], x_train.shape[1], err_msg=err_msg)
    
    n_train_samples, d_dimensions = x_train.shape
    n_test_samples = x_test.shape[0]
    d_length_scale = np.shape(length_scale)
    
    length_scale = _check_length_scale(x_train, length_scale)
    
    # Make the length_scale 1 dimensional
    if np.ndim(length_scale) == 0:
        length_scale = np.array([length_scale])
    
    # check the n_samples for x_train and weights are equal
    err_msg = "Number of training samples for xtrain and weights are not equal."
    np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)


    if int(n_der) == 1:
        constant_term = np.diag(- np.power(length_scale**2, -1))
    
    else:
        constant_term2 = (1 / length_scale)**2
        constant_term4 = (1 / length_scale)**4
    
    # calculate the ARD Kernel
    kernel_mat = ard_kernel(x_test, x_train, length_scale=length_scale, scale=scale)
    
    # initialize derivative matrix
    derivative = np.zeros(shape=(n_test_samples, d_dimensions))
    
    if int(n_der) == 1:
        for itest in range(n_test_samples):
            
            x_tilde = (x_test[itest, :] - x_train).T
            
            kernel_term = (kernel_mat[itest, :][:, np.newaxis] * weights)

            derivative[itest, :] = constant_term.dot(x_tilde).dot(kernel_term).squeeze()
            
    else:
        for itest in range(n_test_samples):
            
            x_term = np.dot(constant_term2, np.ones(shape=(n_dimensions,
                                                          n_train_samples)))
            
            x_term += np.dot(constant_term4, (x_test[itest, :] - x_train).T**2)
            
            derivative[itest, :] = np.dot(x_term, kernel_mat[itest, :] * weights).T 
            
    return derivative


def rbf_derivative(x_train, x_function, weights, length_scale=1.0):
    
    # check the sizes of x_train and x_test
    err_msg = "xtrain and xtest d dimensions are not equivalent."
    np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)
    
    # check the n_samples for x_train and weights are equal
    err_msg = "Number of training samples for xtrain and weights are not equal."
    np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)
    
    
    kernel_mat = rbf_kernel(x_function, x_train, length_scale=length_scale)
    
    n_test, n_dims = x_function.shape
    
    
    derivative = np.zeros(shape=x_function.shape)
    
    for itest in range(n_test):

        if n_dims < 2: 
            derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T, 
                                (kernel_mat[itest, :][:, np.newaxis] * weights))
            
        else:
            derivative[itest, :] = np.dot((x_function[itest, :] - x_train).T, 
                    (kernel_mat[itest, :] * weights).T)
    derivative *= - 1.0 * ( 1 / length_scale**2)
        
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
