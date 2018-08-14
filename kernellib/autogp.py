import autograd.numpy as np
from kernellib.kernels import calculate_Q
from autograd import elementwise_grad as egrad, value_and_grad
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import solve_triangular
import warnings
from operator import itemgetter


class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='ard', jitter=1e-10, normalize_y=None,
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
        K_trans = self.rbf_covariance(X, self.X_train_,
                                      length_scale=self.length_scale,
                                      signal_variance=self.signal_variance)

        y_mean = np.dot(K_trans, self.weights)

        y_mean = self._y_train_mean + y_mean  # undo normal.

        if not return_std:
            return y_mean
        else:
            return y_mean, np.sqrt(self.variance(X, K_trans=K_trans))

    def variance(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.rbf_covariance(X, y=self.X_train_,
                                          length_scale=self.length_scale,
                                          signal_variance=self.signal_variance)

        # compute the variance
        y_var = np.diag(self.rbf_covariance(X, length_scale=self.length_scale,
                                            signal_variance=self.signal_variance)) \
                + self.noise_likelihood
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self.K_inv), K_trans)

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

