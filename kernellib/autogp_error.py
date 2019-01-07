import autograd.numpy as np
from kernellib.kernels import calculate_Q
from autograd import elementwise_grad as egrad, value_and_grad
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import solve_triangular
import warnings
from operator import itemgetter

# TODO: Test x_cov=0.0 same as standard GP

class GPRUncertain(BaseEstimator, RegressorMixin):
    def __init__(self, x_cov=0.0, kernel='ard', jitter=1e-8, normalize_y=None, n_iters=0,
                 random_state=None, optimizer='fmin_l_bfgs_b', derivative_type='full',
                 signal_variance=None, length_scale=None,
                 noise_likelihood=None):
        self.kernel = kernel.lower()
        self.jitter = jitter
        self.normalize_y = normalize_y
        self.n_iters = n_iters
        self.random_state = random_state
        self.optimizer = optimizer
        self.derivative_type = derivative_type
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        self.noise_likelihood = noise_likelihood

        if isinstance(x_cov, float):
            x_cov = np.array([x_cov])
        self.x_cov = x_cov

    def _init_theta(self):
        """Initializes the hyperparameters"""
        signal_variance = 1.0

        if self.kernel == 'ard':
            length_scale = np.ones(self.X_train_.shape[1])
        elif self.kernel == 'rbf':
            length_scale = np.ones(1)
        else:
            raise ValueError('Unrecognized kernel function.'
                             ' Needs to be "rbf" or "ard".')

        noise_likelihood = 1.0

        # stack parameters into one vector
        theta = np.hstack([signal_variance, noise_likelihood, length_scale])

        # transform parameters into logspace

        theta = np.log(theta)

        return theta

    def rand_theta(self):
        tmprng = np.random.RandomState(None)
        signal_variance = tmprng.uniform(1e-5, 1e5, 1)
        if self.kernel == 'ard':
            length_scale = tmprng.uniform(1e-5, 1e5, self.X_train_.shape[1])
        else:
            length_scale = tmprng.uniform(1e-5, 1e5)

        noise_likelihood = tmprng.uniform(1e-5, 1e5, 1)

        theta = np.hstack([signal_variance, noise_likelihood, length_scale])
        return np.log(theta)

    def fit(self, X, y):

        self.X_train_ = X
        self.y_train_ = y

        n_samples, d_dimensions = self.X_train_.shape

        self._rng = np.random.RandomState(self.random_state)

        # Normalize the target value
        if self.normalize_y:
            self._y_train_mean = np.mean(self.y_train_, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        # TODO: Case where the parameters have already been set

        # initialize hyperparameters
        theta0 = self._init_theta()


        # Initialize the derivative at zero
        der_theta = theta0
        self.derivative_term = np.zeros(shape=(n_samples, n_samples))
        #self._derivative_train(theta0, return_term=True)


        # Minimize the objective function
        theta_opt, func_min = self._constrained_optimization(self.obj_func, theta0)
        der_theta = theta_opt

        # Calculate the Derivative
        self.derivative_term = self._derivative_train(der_theta, return_term=True)

        # Minimize Objective Function with Derivative
        theta_opt, func_min = self._constrained_optimization(self.obj_func, theta0)

        best_theta = theta_opt
        best_func = func_min
        best_der_theta = der_theta

        if self.n_iters > 0:



            for iteration in range(self.n_iters):

                # Set the Previous params as the params for Derivative

                # Calculate the derivative
                self.derivative_term = self._derivative_train(theta_opt, return_term=True)

                # Minimize the log likelihood
                itheta, ifunc = self._constrained_optimization(self.obj_func, theta0)

                if ifunc < best_func:
                    # print('Old fun: {:.3f}, New fun: {:.3f}'.format(best_func, ifunc))
                    # print('New params found')
                    best_der_theta = theta_opt
                    best_theta = itheta
                    best_func = ifunc
                else:
                    theta_opt = self.rand_theta()
                    theta0 = self.rand_theta()

        # Gather hyper parameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(best_theta)

        self.signal_variance = np.exp(signal_variance)
        self.noise_likelihood = np.exp(noise_likelihood)
        self.length_scale = np.exp(length_scale)

        self.log_marginal_likelihood_value = -best_func
        self.der_theta = best_der_theta

        # Precompute stuff
        K = ard_covariance(self.X_train_, length_scale=self.length_scale,
                           signal_variance=self.signal_variance)
        # Derivative Term
        self.derivative_term = self._derivative_train(self.der_theta, return_term=True)

        # Ad the noise
        K += self.noise_likelihood * np.eye(n_samples)
        # Add the derivative term
        K += self._derivative_train(self.der_theta, return_term=True)
        # print('K2:', K.min(), K.max(), K.mean())
        L = np.linalg.cholesky(K)

        weights = np.linalg.solve(L.T, np.linalg.solve(L, y))

        self.weights = weights
        self.L = L
        self.K = K

        Linv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        self.Kinv = np.dot(Linv, Linv.T)

        return self

    def log_marginal_likelihood(self, theta):

        # extract parameters (for autograd)
        x_train = self.X_train_
        y_train = self.y_train_
        derivative_term = self.derivative_term
        n_samples, d_dimensions = x_train.shape

        # Support multiout y
        if np.ndim(y_train) == 1:
            y_train = y_train.reshape(-1, 1)

        # Extract hyperparameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(theta)

        # exponentiate the parameters
        signal_variance = np.exp(signal_variance)
        noise_likelihood = np.exp(noise_likelihood)
        length_scale = np.exp(length_scale)

        # Kernel matrix (xtrain, xtrain)
        K = ard_covariance(x_train, length_scale=length_scale, signal_variance=signal_variance)

        # Add the derivative term
        K += derivative_term

        # Add the noise
        K += noise_likelihood * np.eye(n_samples)

        # Cholesky Decomposition
        L = np.linalg.cholesky(K + self.jitter * np.eye(n_samples))

        # TODO: Case of cholesky error

        # Calculate the weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # Calculate the log likelihood per
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, weights)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= (n_samples / 2) * np.log(2 * np.pi)

        # sum for all dimensions
        log_likelihood = log_likelihood_dims.sum(-1)

        return log_likelihood

    def _get_kernel_params(self, theta):
        """Function that extracts the kernel parameters"""
        signal_variance = theta[0]
        noise_likelihood = theta[1] + self.jitter
        length_scale = theta[2:]

        return signal_variance, noise_likelihood, length_scale

    def obj_func(self, theta):
        """Objective function:
        Minimize the negative log marginal likelihppd
        """
        return - self.log_marginal_likelihood(theta)

    def _derivative_train(self, theta, return_term=False):
        """This takes the derivative with respect to the training
        points.

        1. Extract the parameters
        2. Calculate the kernel matrix
        3. Solve to get some initial weights
        4. Use those weights to calculate the derivative

        Parameters
        ----------
        theta : array_like,
            The parameters needed for the RBF kernel and
            noise likelihood.

        Returns
        -------
        derivative : array, (m_test_samples x d_dimensions)
            The derivative with respect to the training points
        """
        x_train = self.X_train_
        y_train = self.X_train_

        if np.ndim(y_train) == 1:
            y_train = y_train.reshape(-1, 1)

        n_samples, d_dimensions = x_train.shape

        # gather hyperparameters
        signal_variance, noise_likelihood, length_scale = self._get_kernel_params(theta)

        signal_variance = np.exp(signal_variance)
        noise_likelihood = np.exp(noise_likelihood)
        length_scale = np.exp(length_scale)

        # Kernel Matrix
        K = ard_covariance(x_train, length_scale=length_scale,
                           signal_variance=signal_variance)

        K += noise_likelihood * np.eye(n_samples)

        # Cholesky Decomposition
        L = np.linalg.cholesky(K + self.jitter * np.eye(n_samples))

        # Weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # Kernel
        Kernel = ard_covariance

        # Define function
        mu = lambda x: np.dot(Kernel(x, length_scale=length_scale,
                                     signal_variance=signal_variance), weights)

        # Wrap it in autograd
        grad_mu = egrad(mu)

        # get derivative
        derivative = grad_mu(x_train)

        if return_term:
            derivative_term = np.dot(derivative, np.dot(np.diag(self.x_cov), derivative.T))

            if self.derivative_type == 'diag':
                derivative_term = np.diag(np.diag(derivative_term))

            return derivative_term
        else:
            return derivative

    def mu_grad(self, X, nder=1):

        mu = lambda x: self.predict(x)

        if nder == 1:
            grad_mu = egrad(mu)
            return grad_mu(X)
        elif nder == 2:
            grad2_mu = egrad(egrad(mu))
            return grad2_mu(X)
        else:
            raise ValueError('Unrecognized "nder". '
                             'Needs to be between 0 and 1.')

    def predict(self, X, return_std=False):

        # Calculate the mean

        # K (xtest, xtrain)
        Ktrans = ard_weighted_covariance(self.X_train_, X, x_cov=self.x_cov,
                                         length_scale=self.length_scale,
                                         signal_variance=self.signal_variance)
        y_mean = np.dot(Ktrans.T, self.weights)
        y_mean = self._y_train_mean + y_mean

        if not return_std:
            return y_mean
        else:
            y_var = self.variance(X, Ktrans=Ktrans, mu=y_mean)
            return y_mean, np.sqrt(y_var)

    def variance(self, X, Ktrans=None, mu=None):


        # Determinant Term
        det_term = np.array([2 * self.x_cov * np.power(self.length_scale, -2) + 1])

        det_term = 1 / np.sqrt(np.linalg.det(det_term))
        # print('Determinant Term:', det_term)

        # Exponential Term
        exp_scale = np.power(np.power(self.length_scale, 2) +
                             0.5 * np.power(self.length_scale, 4) * np.power(self.x_cov, -1), -1)

        if Ktrans is None:
            Ktrans = ard_weighted_covariance(self.X_train_,
                                             X, x_cov=self.x_cov,
                                             length_scale=self.length_scale,
                                             signal_variance=self.signal_variance)

        Ktraintest = ard_covariance(self.X_train_,
                                    X,
                                    length_scale=self.length_scale,
                                    signal_variance=self.signal_variance)

        if mu is None:
            mu = np.dot(Ktrans.T, self.weights)
            mu = self._y_train_mean + mu

        # Ktraintest =
        Q = calculate_Q(self.X_train_, X, Ktraintest, det_term, exp_scale)

        y_var = np.zeros(shape=(X.shape[0]))
        # self.signal_variance - mu.squeeze()**2
        # print('Old yvar:', y_var.min(), y_var.max(), y_var.mean())
        # print(self.Kinv.shape, Q[0, ...].shape)
        for itest in range(y_var.shape[0]):
            y_var[itest] = self.signal_variance
            y_var[itest] -= np.trace(np.dot(self.Kinv, Q[itest, ...]))
            y_var[itest] += np.dot(self.weights.T, np.dot(Q[itest, :, :], self.weights))[0][0]
            y_var[itest] -= mu[itest, 0]**2
        # print('New yvar:', y_var.min(), y_var.max(), y_var.mean())
        return y_var

    def _constrained_optimization(self, obj_func, initial_theta, bounds=None):

        if self.optimizer == 'fmin_l_bfgs_b':
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(value_and_grad(obj_func), initial_theta, bounds)

            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict)

        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(value_and_grad(obj_func), initial_theta, bounds=bounds)

        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min


class GPR(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='ard', x_cov=0.0, jitter=1e-10, normalize_y=None,
                 n_restarts=0, random_state=None,
                 signal_variance=None, length_scale=None,
                 noise_likelihood=None):
        self.kernel = kernel
        self.jitter = jitter
        self.normalize_y = normalize_y
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        self.noise_likelihood = noise_likelihood

        if isinstance(x_cov, float):
            x_cov = np.array([x_cov])
        self.x_cov = x_cov

    def init_theta(self):
        """Initializes the hyperparameters."""
        signal_variance = 1.0

        if self.kernel == 'ard':
            length_scale = np.ones(self.X_train_.shape[1])
        else:
            length_scale = 1.0
        noise_likelihood = 0.01
        theta = np.hstack([signal_variance, noise_likelihood, length_scale])
        return np.log(theta)

    def rand_theta(self):
        tmprng = np.random.RandomState(None)
        signal_variance = tmprng.uniform(1e-5, 1e5, 1)
        if self.kernel == 'ard':
            length_scale = tmprng.uniform(1e-5, 1e5, self.X_train_.shape[1])
        else:
            length_scale = tmprng.uniform(1e-5, 1e5)

        noise_likelihood = tmprng.uniform(1e-5, 1e5, 1)

        theta = np.hstack([signal_variance, noise_likelihood, length_scale])
        return np.log(theta)

    def fit(self, X, y):

        self.X_train_ = X
        self.y_train_ = y

        self._rng = np.random.RandomState(self.random_state)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(self.y_train_, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if self.length_scale is None and self.noise_likelihood is None:

            # initial hyper-parameters
            theta0 = self.init_theta()

            # # minimize the objective function
            # best_params = minimize(value_and_grad(self.log_marginal_likelihood), theta0, jac=True,
            #                        method='L-BFGS-B')

            bounds = None  # ((1e-10, None), (1e-10, None), (1e-10, None))

            optima = [(self._constained_optimization(self.obj_func,
                                                     theta0,
                                                     bounds=bounds))]
            if self.n_restarts > 0:

                for iteration in range(self.n_restarts):
                    optima.append(
                        self._constained_optimization(self.obj_func,
                                                      self.rand_theta(),
                                                      bounds=bounds))

            lml_values = list(map(itemgetter(1), optima))
            theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)

            # Gather hyper parameters
            signal_variance, noise_likelihood, length_scale = \
                self._get_kernel_params(theta)

            self.signal_variance = np.exp(signal_variance)
            self.noise_likelihood = np.exp(noise_likelihood)
            self.length_scale = np.exp(length_scale)

        if self.signal_variance is None:
            self.signal_variance = 1.0
        if self.length_scale is None:
            self.length_scale = 1.0
        if self.noise_likelihood is None:
            self.noise_likelihood = 0.01
        # Calculate the weights
        K = self.rbf_covariance(X, length_scale=self.length_scale,
                                signal_variance=self.signal_variance)
        K += self.noise_likelihood * np.eye(K.shape[0])
        L = np.linalg.cholesky(K + self.noise_likelihood * np.eye(K.shape[0]))
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y))

        self.weights = weights
        self.L = L
        self.K = K

        L_inv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))

        self.K_inv = np.dot(L_inv, L_inv.T)

        return self

    def obj_func(self, theta):
        return - self.log_marginal_likelihood(theta)

    def log_marginal_likelihood(self, theta):
        x_train = self.X_train_
        y_train = self.y_train_

        if np.ndim(y_train) == 1:
            y_train = y_train[:, np.newaxis]

        # Gather hyper parameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(theta)
        signal_variance = np.exp(signal_variance)
        noise_likelihood = np.exp(noise_likelihood)
        length_scale = np.exp(length_scale)

        n_samples = x_train.shape[0]

        # train kernel
        K = self.rbf_covariance(x_train, length_scale=length_scale,
                                signal_variance=signal_variance)
        K += noise_likelihood * np.eye(n_samples)
        try:
            L = np.linalg.cholesky(K + self.jitter * np.eye(n_samples))
        except np.linalg.LinAlgError:
            return -np.inf

        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, weights)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= (K.shape[0] / 2) * np.log(2 * np.pi)

        log_likelihood = log_likelihood_dims.sum(-1)

        return log_likelihood

    def predict(self, X, return_std=False):

        # Train test kernel
        Ktrans = ard_weighted_covariance(X, self.X_train_,
                                         x_cov=self.x_cov,
                                         length_scale=self.length_scale,
                                         signal_variance=self.signal_variance)

        y_mean = np.dot(Ktrans, self.weights)

        y_mean = self._y_train_mean + y_mean  # undo normal.

        if not return_std:
            return y_mean
        else:
            return y_mean, np.sqrt(self.variance(X, Ktrans=Ktrans.T, mu=y_mean))

    def variance(self, X, Ktrans=None, mu=None):


        # Determinant Term
        det_term = np.array([2 * self.x_cov * np.power(self.length_scale, -2) + 1])

        det_term = 1 / np.sqrt(np.linalg.det(det_term))
        # print('Determinant Term:', det_term)

        # Exponential Term
        exp_scale = np.power(np.power(self.length_scale, 2) +
                             0.5 * np.power(self.length_scale, 4) * np.power(self.x_cov, -1), -1)

        if Ktrans is None:
            Ktrans = ard_weighted_covariance(self.X_train_,
                                             X, x_cov=self.x_cov,
                                             length_scale=self.length_scale,
                                             signal_variance=self.signal_variance)

        Ktraintest = ard_covariance(self.X_train_,
                                    X,
                                    length_scale=self.length_scale,
                                    signal_variance=self.signal_variance)

        if mu is None:
            mu = np.dot(Ktrans.T, self.weights)
            mu = self._y_train_mean + mu

        # Ktraintest =
        Q = calculate_Q(self.X_train_, X, Ktraintest, det_term, exp_scale)

        y_var = np.zeros(shape=(X.shape[0]))
        # self.signal_variance - mu.squeeze()**2
        for itest in range(y_var.shape[0]):
            y_var[itest] = self.signal_variance
            y_var[itest] -= np.trace(np.dot(self.K_inv, Q[itest, ...]))
            y_var[itest] += np.dot(self.weights.T, np.dot(Q[itest, :, :], self.weights))[0][0]
            y_var[itest] -= mu[itest, 0]**2
        return y_var

    def _constained_optimization(self, obj_func, initial_theta, bounds):

        theta_opt, func_min, convergence_dict = \
            fmin_l_bfgs_b(value_and_grad(obj_func), initial_theta, bounds=bounds)

        if convergence_dict["warnflag"] != 0:
            warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                          " state: {}".format(convergence_dict))

        return theta_opt, func_min

    def _get_kernel_params(self, theta):

        signal_variance = theta[0]
        noise_likelihood = theta[1] + self.jitter
        length_scale = theta[2:]

        return signal_variance, noise_likelihood, length_scale

    def rbf_covariance(self, X, y=None, signal_variance=1.0, length_scale=1.0):

        if y is None:
            y = X

        D = np.expand_dims(X / length_scale, 1) - np.expand_dims(y / length_scale, 0)

        return signal_variance * np.exp(-0.5 * np.sum(D ** 2, axis=2))

    def mu_grad(self, X, nder=1):

        # Construct the autogradient function for the
        # predictive mean
        mu = lambda x: self.predict(x)

        if nder == 1:
            grad_mu = egrad(mu)

            return grad_mu(X)
        else:
            grad_mu = egrad(egrad(mu))
            return grad_mu(X)

    def mu_kern(self, X, nder=1):

        mu = lambda x: self.rbf_covariance(
            x, self.X_train_,
            length_scale=self.length_scale,
            signal_variance=self.signal_variance)

        if nder == 1:
            grad_kern = egrad(mu)

            return grad_kern(X)
        else:
            grad_kern = egrad(egrad(mu))
            return grad_kern(X)

    def sigma_grad(self, X, nder=1):

        # Construct the autogradient function for the
        # predictive variance
        sigma = lambda x: self.variance(x)

        if nder == 1:
            grad_var = egrad(sigma)
            return grad_var(X)
        else:
            grad_var = egrad(egrad(sigma))
            return grad_var(X)

    def point_sensitivity(self, X, sample='point', method='squared'):

        # Calculate the derivative
        derivative = self.mu_grad(X, nder=1)

        if method == 'squared':
            derivative **= 2
        else:
            derivative = np.abs(derivative)

        # X, Y Point Sensitivity or Dimension
        if sample == 'dim':
            return np.mean(derivative, axis=0)

        else:
            return np.mean(derivative, axis=1)

    def sensitivity(self, X, method='squared'):

        der = self.mu_grad(X, nder=1)

        if method == 'squared':
            return np.mean(np.mean(der ** 2))
        else:
            return np.mean(np.mean(np.abs(der)))


class GPRMean(BaseEstimator, RegressorMixin):
    def __init__(self, x_cov=0.0, kernel='ard', jitter=1e-8, normalize_y=None, n_iters=0,
                 random_state=None, optimizer='fmin_l_bfgs_b', derivative_type='full',
                 signal_variance=None, length_scale=None,
                 noise_likelihood=None):
        self.kernel = kernel.lower()
        self.jitter = jitter
        self.normalize_y = normalize_y
        self.n_iters = n_iters
        self.random_state = random_state
        self.optimizer = optimizer
        self.derivative_type = derivative_type
        self.signal_variance = signal_variance
        self.length_scale = length_scale
        self.noise_likelihood = noise_likelihood

        if isinstance(x_cov, float):
            x_cov = np.array([x_cov])
        self.x_cov = x_cov

    def _init_theta(self):
        """Initializes the hyperparameters"""
        signal_variance = 1.0

        if self.kernel == 'ard':
            length_scale = np.ones(self.X_train_.shape[1])
        elif self.kernel == 'rbf':
            length_scale = np.ones(1)
        else:
            raise ValueError('Unrecognized kernel function.'
                             ' Needs to be "rbf" or "ard".')

        noise_likelihood = 1.0

        # stack parameters into one vector
        theta = np.hstack([signal_variance, noise_likelihood, length_scale])

        # transform parameters into logspace

        theta = np.log(theta)

        return theta

    def rand_theta(self):
        tmprng = np.random.RandomState(None)
        signal_variance = tmprng.uniform(1e-5, 1e5, 1)
        if self.kernel == 'ard':
            length_scale = tmprng.uniform(1e-5, 1e5, self.X_train_.shape[1])
        else:
            length_scale = tmprng.uniform(1e-5, 1e5)

        noise_likelihood = tmprng.uniform(1e-5, 1e5, 1)

        theta = np.hstack([signal_variance, noise_likelihood, length_scale])
        return np.log(theta)

    def fit(self, X, y):

        self.X_train_ = X
        self.y_train_ = y

        n_samples, d_dimensions = self.X_train_.shape

        self._rng = np.random.RandomState(self.random_state)

        # Normalize the target value
        if self.normalize_y:
            self._y_train_mean = np.mean(self.y_train_, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        # TODO: Case where the parameters have already been set

        # initialize hyperparameters
        theta0 = self._init_theta()


        # Initialize the derivative at zero
        der_theta = theta0
        self.derivative_term = np.zeros(shape=(n_samples, n_samples))
        #self._derivative_train(theta0, return_term=True)


        # Minimize the objective function
        theta_opt, func_min = self._constrained_optimization(self.obj_func, theta0)
        der_theta = theta_opt

        # Calculate the Derivative
        self.derivative_term = self._derivative_train(der_theta, return_term=True)

        # Minimize Objective Function with Derivative
        theta_opt, func_min = self._constrained_optimization(self.obj_func, theta0)

        best_theta = theta_opt
        best_func = func_min
        best_der_theta = der_theta

        if self.n_iters > 0:



            for iteration in range(self.n_iters):

                # Set the Previous params as the params for Derivative

                # Calculate the derivative
                self.derivative_term = self._derivative_train(theta_opt, return_term=True)

                # Minimize the log likelihood
                itheta, ifunc = self._constrained_optimization(self.obj_func, theta0)

                if ifunc < best_func:
                    # print('Old fun: {:.3f}, New fun: {:.3f}'.format(best_func, ifunc))
                    # print('New params found')
                    best_der_theta = theta_opt
                    best_theta = itheta
                    best_func = ifunc
                else:
                    theta_opt = self.rand_theta()
                    theta0 = self.rand_theta()

        # Gather hyper parameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(best_theta)

        self.signal_variance = np.exp(signal_variance)
        self.noise_likelihood = np.exp(noise_likelihood)
        self.length_scale = np.exp(length_scale)

        self.log_marginal_likelihood_value = -best_func
        self.der_theta = best_der_theta

        # Precompute stuff
        K = ard_covariance(self.X_train_, length_scale=self.length_scale,
                           signal_variance=self.signal_variance)
        # Derivative Term
        self.derivative_term = self._derivative_train(best_der_theta, return_term=True)

        # Ad the noise
        K += self.noise_likelihood * np.eye(n_samples)

        # Add the derivative term
        K += self.derivative_term

        L = np.linalg.cholesky(K + self.jitter * np.eye(n_samples))

        weights = np.linalg.solve(L.T, np.linalg.solve(L, y))

        self.weights = weights
        self.L = L
        self.K = K

        Linv = solve_triangular(self.L.T, np.eye(self.L.shape[0]))
        self.Kinv = np.dot(Linv, Linv.T)

        return self

    def log_marginal_likelihood(self, theta):

        # extract parameters (for autograd)
        x_train = self.X_train_
        y_train = self.y_train_
        derivative_term = self.derivative_term
        n_samples, d_dimensions = x_train.shape

        # Support multiout y
        if np.ndim(y_train) == 1:
            y_train = y_train.reshape(-1, 1)

        # Extract hyperparameters
        signal_variance, noise_likelihood, length_scale = \
            self._get_kernel_params(theta)

        # exponentiate the parameters
        signal_variance = np.exp(signal_variance)
        noise_likelihood = np.exp(noise_likelihood)
        length_scale = np.exp(length_scale)

        # Kernel matrix (xtrain, xtrain)
        K = ard_covariance(x_train, length_scale=length_scale, signal_variance=signal_variance)

        # Add the derivative term
        K += derivative_term

        # Add the noise
        K += noise_likelihood * np.eye(n_samples)

        # Cholesky Decomposition
        L = np.linalg.cholesky(K + self.jitter * np.eye(n_samples))

        # TODO: Case of cholesky error

        # Calculate the weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # Calculate the log likelihood per
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, weights)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= (n_samples / 2) * np.log(2 * np.pi)

        # sum for all dimensions
        log_likelihood = log_likelihood_dims.sum(-1)

        return log_likelihood

    def _get_kernel_params(self, theta):
        """Function that extracts the kernel parameters"""
        signal_variance = theta[0]
        noise_likelihood = theta[1] + self.jitter
        length_scale = theta[2:]

        return signal_variance, noise_likelihood, length_scale

    def obj_func(self, theta):
        """Objective function:
        Minimize the negative log marginal likelihppd
        """
        return - self.log_marginal_likelihood(theta)

    def _derivative_train(self, theta, return_term=False):
        """This takes the derivative with respect to the training
        points.

        1. Extract the parameters
        2. Calculate the kernel matrix
        3. Solve to get some initial weights
        4. Use those weights to calculate the derivative

        Parameters
        ----------
        theta : array_like,
            The parameters needed for the RBF kernel and
            noise likelihood.

        Returns
        -------
        derivative : array, (m_test_samples x d_dimensions)
            The derivative with respect to the training points
        """
        x_train = self.X_train_
        y_train = self.X_train_

        if np.ndim(y_train) == 1:
            y_train = y_train.reshape(-1, 1)

        n_samples, d_dimensions = x_train.shape

        # gather hyperparameters
        signal_variance, noise_likelihood, length_scale = self._get_kernel_params(theta)

        signal_variance = np.exp(signal_variance)
        noise_likelihood = np.exp(noise_likelihood)
        length_scale = np.exp(length_scale)

        # Kernel Matrix
        K = ard_covariance(x_train, length_scale=length_scale,
                           signal_variance=signal_variance)

        K += noise_likelihood * np.eye(n_samples)

        # Cholesky Decomposition
        L = np.linalg.cholesky(K + self.jitter * np.eye(n_samples))

        # Weights
        weights = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

        # Kernel
        Kernel = ard_covariance

        # Define function
        mu = lambda x: np.dot(Kernel(x, length_scale=length_scale,
                                     signal_variance=signal_variance), weights)

        # Wrap it in autograd
        grad_mu = egrad(mu)

        # get derivative
        derivative = grad_mu(x_train)

        if return_term:
            derivative_term = np.dot(derivative, np.dot(np.diag(self.x_cov), derivative.T))

            if self.derivative_type == 'diag':
                derivative_term = np.diag(np.diag(derivative_term))

            return derivative_term
        else:
            return derivative

    def mu_grad(self, X, nder=1):

        mu = lambda x: self.predict(x)

        if nder == 1:
            grad_mu = egrad(mu)
            return grad_mu(X)
        elif nder == 2:
            grad2_mu = egrad(egrad(mu))
            return grad2_mu(X)
        else:
            raise ValueError('Unrecognized "nder". '
                             'Needs to be between 0 and 1.')

    def predict(self, X, return_std=False):

        # K (xtest, xtrain)
        Ktrans = ard_covariance(X, self.X_train_, length_scale=self.length_scale,
                                signal_variance=self.signal_variance)

        # Calculate the mean
        y_mean = np.dot(Ktrans, self.weights)

        # Add mean to the labels again
        y_mean = self._y_train_mean + y_mean

        if not return_std:
            return y_mean
        else:
            y_var = np.sqrt(self.variance(X, Ktrans=Ktrans))
            return y_mean, y_var

    def variance(self, X, Ktrans):

        x_train = self.X_train_

        if Ktrans is None:
            Ktrans = ard_covariance(X, y=x_train, length_scale=self.length_scale,
                                    signal_variance=self.signal_variance)

        # K (test, test)
        Ktest = ard_covariance(X, length_scale=self.length_scale,
                               signal_variance=self.signal_variance)

        # Calculate the derivative
        derivative = self.mu_grad(X)
        derivative_term = np.einsum("ij,ij->i", np.dot(derivative, np.diag(self.x_cov)), derivative)

        y_var = np.diag(Ktest) + self.noise_likelihood + derivative_term

        y_var -= np.einsum("ij,ij->i", np.dot(Ktrans, self.Kinv), Ktrans)

        return y_var

    def _constrained_optimization(self, obj_func, initial_theta, bounds=None):

        if self.optimizer == 'fmin_l_bfgs_b':
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(value_and_grad(obj_func), initial_theta, bounds)

            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict)

        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(value_and_grad(obj_func), initial_theta, bounds=bounds)

        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min



def ard_covariance(X, y=None, length_scale=1.0, signal_variance=1.0):
    if y is None:
        y = X

    # Calculate the distances
    D = np.expand_dims(X / length_scale, 1) - np.expand_dims(y / length_scale, 0)

    # Calculate the kernel matrix
    K = signal_variance * np.exp(-0.5 * np.sum(D ** 2, axis=2))

    return K

def ard_weighted_covariance(X, Y=None, x_cov=None, length_scale=None,
                            signal_variance=None):

    # grab samples and dimensions
    n_samples, n_dimensions = X.shape

    # get the default sigma values
    if length_scale is None:
        length_scale = np.ones(shape=n_dimensions)

    # check covariance values
    if x_cov is None:
        x_cov = np.array([0.0])

    # Add dimensions to lengthscale and x_cov
    if np.ndim(length_scale) == 0:
        length_scale = np.array([length_scale])

    if np.ndim(x_cov) == 0:
        x_cov = np.array([x_cov])

    # get default scale values
    if signal_variance is None:
        signal_variance = 1.0

    exp_scale = np.sqrt(x_cov + length_scale ** 2)

    scale_term = np.diag(x_cov * (length_scale ** 2) ** (-1)) + np.eye(N=n_dimensions)
    scale_term = np.linalg.det(scale_term)
    scale_term = signal_variance * np.power(scale_term, -1 / 2)


    # Calculate the distances
    D = np.expand_dims(X / exp_scale, 1) - np.expand_dims(Y / exp_scale, 0)

    # Calculate the kernel matrix
    K = scale_term * np.exp(-0.5 * np.sum(D ** 2, axis=2))

    return K

def main():
    from kernellib.data import example_error_1d

    sample_func = 1
    x_error = 0.3
    X, y, error_params = example_error_1d(sample_func, x_error)

    xtrain, xtest = X['train'], X['test']
    ytrain, ytest = y['train'], y['test']
    xplot, yplot = X['plot'], y['plot']
    x_cov, sigma_y = error_params['x'], error_params['y']

    # Mean Noise GP
    ngp_model = GaussianProcessError(x_cov=0.0)
    ngp_model.fit(xtrain, ytrain);
    mean1, std1 = ngp_model.predict(xplot, return_std=True)


    # Uncertainty Noise GP
    ugp_model = GaussianProcessVariance(x_cov=0.3)

    ugp_model.fit(xtrain, ytrain);
    mean, std = ugp_model.predict(xplot, return_std=True)


    return None


if __name__ == '__main__':
    main()
