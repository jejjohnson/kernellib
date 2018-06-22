cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def rbf_derivative(np.float64_t[:, :] x_train, 
                   np.float64_t[:, :] x_function,
                   np.float64_t[:] weights,
                   np.float64_t[:, :] kernel_mat,
                   np.int_t n_derivative,
                   np.float64_t gamma):
    """This function calculates the rbf derivative using
    Cython. It has been fairly optimized and provides x100
    speedup over the original python function.
    
    Parameters
    ----------
    x_train : array, [N x D], float64
        The training data used to find the kernel model.

    x_function  : array, [M x D], float
        The test points (or vector) to use.

    weights   : array, [N x D], float64
        The weights found from the kernel model
            y = K * weights

    kernel_mat: array, [N x M], float64
        The rbf kernel matrix with the similarities between the test
        points and the training points.

    n_derivative : int, (default = 1) {1, 2}, int
        chooses which nth derivative to calculate

    gamma : float, default: None, float64
        the parameter for the rbf_kernel matrix function

    Returns
    -------

    derivative : array, [M x D], float64
        returns the derivative with respect to training points used in
        the kernel model and the test points.

    Information
    -----------
    Author: Juan Emmanuel Johnson
    Email : jej2744@rit.edu
            juan.johnson@uv.es
    """
    cdef int d_dimensions = x_function.shape[1]
    cdef int n_test = x_function.shape[0]
    cdef int n_train = x_train.shape[0]
    cdef int idim, iTest, iTrain
    
    # initialize the derivative
    cdef np.float64_t[:,:] derivative = np.zeros((n_test, d_dimensions))

    # consolidate the parameters
    cdef np.float64_t theta = 2.0 * gamma


    if n_derivative == 1:
        
        # loop through dimensions
        for idim in np.arange(0, d_dimensions):

            # loop through the number of test points
            for iTest in np.arange(0, n_test):

                # loop through the number of test points
                for iTrain in np.arange(0, n_train):

                    # calculate the derivative for the test points
                    derivative[iTest, idim] += theta * weights[iTrain] * \
                                              (x_train[iTrain, idim] -
                                               x_function[iTest, idim]) * \
                                              kernel_mat[iTrain, iTest]
                        
    # 2nd derivative
    elif n_derivative == 2:

        # loop through dimensions
        for dim in np.arange(0, d_dimensions):

            # loop through the number of test points
            for iTest in np.arange(0, n_test):

                # loop through the number of test points
                for iTrain in np.arange(0, n_train):
                    derivative[iTest, dim] += weights[iTrain] * \
                                              (theta ** 2 *
                                               (x_train[iTrain, dim] - x_function[iTest, dim]) ** 2
                                               - theta) * \
                                              kernel_mat[iTrain, iTest] 
    else:
        raise ValueError('n_derivative should be equal to 1 or 2.')
                            
    return np.asarray(derivative)