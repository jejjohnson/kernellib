import numpy as np
import numba
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import check_random_state
from ..kernels import rbf_kernel, estimate_length_scale, kernel_centerer
from ..kernels.derivatives import hsic_rbf_derivative
from ..kernels.kernel_approximation import RFF

# TODO: Allow Other Kernel Approximations for RHSIC
# TODO: Merge Two Classes (option for kernel matrix approximation)

class HSIC(object):
    """Hilbert-Schmidt Independence Criterion (HSIC). This is
    a method for measuring independence between two variables.
    
    Parameters
    ----------
    kernel: str, optional (default='rbf')
        Kernel function to use for X and Y

    sigma_x : float, optional (default=None)
        The length scale for the RBF kernel function for the X
        variable

    sigma_y : float, optional (default=None)
        The length scale for the RBF kernel function for the Y
        variable.

    sub_sample : int, optional (default=1000)

    X_stat : str, optional (default='median')
        {'mean', 'median', 'silverman'}

    Y_stat : str, optional (default='median')
        {'mean', 'median', 'silverman'}
    
    random_state : int, optional (default=1234)

    Attributes
    ----------
    hsic_value : float

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 14-Feb-2019

    Resources
    ---------
    Original MATLAB Implementation : 
        http:// isp.uv.es/code/shsic.zip
    Paper :
        Sensitivity maps of the Hilbert–Schmidt independence criterion
        Perez-Suay et al., 2018
    """
    def __init__(self, 
                 kernel='rbf', 
                 sigma_x=None, 
                 sigma_y=None, 
                 sub_sample=1000,
                 X_stat='median', 
                 Y_stat='median',
                 random_state=1234):
        self.kernel = kernel
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sub_sample = sub_sample
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.X_stat = X_stat 
        self.Y_stat = Y_stat
        self.hsic_fit = None
    
    def fit(self, X, Y):

        # Check sizes of X, Y
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)
        
        # Check samples are the same
        assert(X.shape[0] == Y.shape[0])
        
        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]
        
        self.X_train_ = X
        self.Y_train_ = Y
        factor = (1 / (self.n_samples - 1)**2)
        # Calculate Kernel Matrices for X, Y
        if self.kernel is 'rbf':
            # Estimate sigma parameter (RBF) kernel only
            if self.sigma_x is None:
                self.sigma_x = estimate_length_scale(
                    X, sub_sample=self.sub_sample, method=self.X_stat, random_state=self.rng)

            if self.sigma_y is None:
                self.sigma_y = estimate_length_scale(
                    Y, sub_sample=self.sub_sample, method=self.Y_stat, random_state=self.rng)

            # Calculate Kernel Matrices
            self.K_x = rbf_kernel(X, length_scale=self.sigma_x)
            self.K_y = rbf_kernel(Y, length_scale=self.sigma_y)
            # Center Kernel
            self.H = kernel_centerer(self.n_samples)
            self.K_xc = self.K_x @ self.H
            self.K_yc = self.K_y @ self.H


        elif self.kernel is 'lin':
            print('Here')
            self.K_x = X
            self.K_y = Y

            # Centered Kernels
            self.K_xc = (X-X.mean(axis=0))
            self.K_yc = (Y-Y.mean(axis=0))

        else:
            raise ValueError('No kernel.')


        
        # TODO: check kernelcentering (sklearn)
        
        # Compute HSIC value
        if self.kernel is 'rbf':
            self.hsic_value = (1 / (self.n_samples - 1)**2) * d\
                np.einsum('ji,ij->', self.K_xc, self.K_yc)
        elif self.kernel is 'lin':
            # if self.dx_dimensions < self.n_samples:
            Rxy = (X-X.mean(axis=0)).T @ (Y-Y.mean(axis=0))

            self.Rxy = Rxy
            self.hsic_value = factor * \
                np.real(np.einsum('ij,ji->', Rxy, Rxy.T))

            # else:
            #     Zxx = self.K_xc @ X.T
            #     Zyy = Y @ Y.T
            #     rh = factor * np.real(np.einsum('ij,ji->', Zxx, Zyy))
        else:
            raise ValueError('No kernel.')

        self.hsic_fit= True
        return self
    
    def derivative(self):
        
        # check if HSIC function is fit
        if self.hsic_fit is None:
            raise ValueError("Function isn't fit. Need to fit function to some data.")
        
        if self.kernel is 'rbf':
            self.derX, self.derY = hsic_rbf_derivative(
                self.X_train_, self.Y_train_, self.H, 
                self.K_x, self.K_y, self.sigma_x, self.sigma_y
            )
        elif self.kernel is 'lin':
            self.derX, self.derY = hsic_rbf_derivative(
                self.X_train_, self.Y_train_, self.H,
                self.K_x, self.K_y, 1.0, 1.0
                )
        else:
            raise ValueError('No kernel.')
        
        return self.derX, self.derY
    
    def sensitivity(self, standard=True):

        if (not hasattr(self, 'derX')) or (not hasattr(self, 'derY')):
            print('No derivatives found. Recalculating derivative.')
            self.derX, self.derY = self.derivative()
            
        sens = np.sqrt(self.derX**2 + self.derY**2)
        
        # standardize
        if standard:
            sens = StandardScaler(with_mean=True, with_std=False).fit_transform(sens)
        
        return sens
    
    def test_estat(self):
        pass


class RHSIC(object):
    """Randomized Hilbert-Schmidt Independence Criterion (RHSIC). 
    This is a method for measuring independence between two variables.
    It uses kernel matrix approximation
    
    Parameters
    ----------
    kernel: str, optional (default='rff')
        Kernel function to use for X and Y

    sigma_x : float, optional (default=None)
        The length scale for the RBF/RFF kernel function for the X
        variable

    sigma_y : float, optional (default=None)
        The length scale for the RBF/RFF kernel function for the Y
        variable.

    sub_sample : int, optional (default=1000)

    X_stat : str, optional (default='median')
        {'mean', 'median', 'silverman'}

    Y_stat : str, optional (default='median')
        {'mean', 'median', 'silverman'}
    
    random_state : int, optional (default=1234)

    Attributes
    ----------
    hsic_value : float

    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 14-Feb-2019

    Resources
    ---------
    Original MATLAB Implementation : 
        http:// isp.uv.es/code/shsic.zip
    Paper :
        Sensitivity maps of the Hilbert–Schmidt independence criterion
        Perez-Suay et al., 2018
    """
    def __init__(self,
                 kernel_approx='rff',
                 n_features=100,
                 sigma_x=None,
                 sigma_y=None,
                 sub_sample=1000,
                 X_stat='median',
                 Y_stat='median',
                 random_state=1234):
        self.kernel_approx = kernel_approx
        self.n_features = n_features
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sub_sample = sub_sample
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.X_stat = X_stat
        self.Y_stat = Y_stat
        self.hsic_fit = None
    
    def fit(self, X, Y):
        
        # Check sizes of X, Y
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)
        
        assert(X.shape[0] == Y.shape[0])
        
        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]
        
        factor =  1 / (self.n_samples - 1)**2
        
        self.X_train_ = X
        self.Y_train_ = Y
            
        # Estimate sigma parameter (RBF) kernel only
        if self.sigma_x is None:
            self.sigma_x = estimate_length_scale(
                X, sub_sample=self.sub_sample, method=self.X_stat, random_state=self.rng)
            
        if self.sigma_y is None:
            self.sigma_y = estimate_length_scale(
                Y, sub_sample=self.sub_sample, method=self.Y_stat, random_state=self.rng)
        

        if self.kernel_approx in ['rff', 'rbf']:
            # =================================
            # Calculate Kernel Matrices for X
            # =================================
            # Initialize RFF Approximation Class
            self.rff_model_X = RFF(
                n_components=self.n_features,
                length_scale=self.sigma_x,
                method=self.X_stat,
                center=False,
                random_state=self.rng
            )

            # Fit to X
            self.rff_model_X.fit(X)

            # Transform Data X
            self.Zx = self.rff_model_X.transform(X, return_real=False)
            self.Zxc = self.Zx - self.Zx.mean(axis=0)

            # =================================
            # Calculate Kernel Matrices for Y
            # =================================
            self.rff_model_Y = RFF(
                n_components=self.n_features,
                length_scale=self.sigma_y,
                method=self.Y_stat,
                center=False,
                random_state=self.rng
            )

            self.rff_model_Y.fit(Y)

            self.Zy = self.rff_model_Y.transform(Y, return_real=False)
            self.Zyc = self.Zy - self.Zy.mean(axis=0)
        else:
            raise ValueError('Unrecognized Kernel Approximation method.')

        # ====================
        # Compute HSIC Value
        # ====================
        if self.n_features < self.n_samples:
            Rxy = np.matrix.getH(self.Zxc) @ self.Zyc
            self.Rxy = Rxy
            rh = factor * np.real(np.einsum('ij,ji->', Rxy, np.matrix.getH(Rxy)))

        else:
            Zxx = self.Zx @ np.matrix.getH(self.Zxc)
            Zyy = self.Zy @ np.matrix.getH(self.Zyc)
            rh = factor * np.real(np.einsum('ij,ji->', Zxx, Zyy))
            
        self.hsic_value = rh
        
        self.hsic_fit = True
        return self
    
    def derivative(self):
        
        if self.kernel_approx in ['rff', 'rbf']:
            return self.rff_derivative()
        else:
            raise ValueError('Unrecognized derivative.')
        return None
    
    def rff_derivative(self):
        
        if self.hsic_fit is None:
            raise ValueError("Function isn't fit. Need to fit function to some data.")
            
        factor =  2 / (self.n_samples - 1)**2
        
        mapX = np.zeros((self.X_train_.shape))
        Jx = np.zeros((1, self.dx_dimensions))
        mapY = np.zeros((self.Y_train_.shape))
        Jy = np.zeros((1, self.dy_dimensions))
        
        np.testing.assert_array_almost_equal(
            self.Zyc @ (np.matrix.getH(self.Zyc) @ self.Zx),
            (self.Zyc @ np.matrix.getH(self.Zyc)) @ self.Zx
        )
        
            
        BBx = self.Zyc @ (np.matrix.getH(self.Zyc) @ self.Zx)
        BBy = self.Zxc @ (np.matrix.getH(self.Zxc) @ self.Zy)

        # X Term

        for idim in range(self.dx_dimensions):
            for isample in range(self.n_samples):
                Jx[:, idim]         = 1
                aux                 = 1j * Jx @ self.rff_model_X.W
                Jx[:, idim]         = 0
                derX                = self.Zx[isample, :] * aux
                mapX[isample, idim] = np.real(BBx[isample, :][None, :] @ np.matrix.getH(derX)).squeeze()

        mapX = factor * mapX

        # Y Term

        for idim in range(self.dy_dimensions):
            for isample in range(self.n_samples):
                Jy[:, idim]         = 1
                aux                 = 1j * Jy @ self.rff_model_Y.W
                Jy[:, idim]         = 0
                derY                = self.Zy[isample, :] * aux
                mapY[isample, idim] = np.real(BBy[isample, :][None, :] @ np.matrix.getH(derY)).squeeze()

        mapY = factor * mapY
        
        self.der_X = mapX
        self.der_Y = mapY
        return mapX, mapY
    
    def sensitivity(self, standard=True):
        
        if (not hasattr(self, 'der_X')) or (not hasattr(self, 'der_Y')):
            print('No derivatives found. Recalculating derivative.')
            self.der_X, self.der_Y = self.derivative()
            
        sens = np.sqrt(self.der_X**2 + self.der_Y**2)
        
        # standardize
        if standard:
            sens = StandardScaler(with_mean=True, with_std=False).fit_transform(sens)
        
        return sens


def get_sample_data(dataset='hh', num_points=1000, seed=1234, noise=0.1):
    """Generates sample datasets to go along with a demo for paper.
    
    Parameters
    ----------
    dataset = str, optional (default='hh')
        The dataset generated from the function.
        {'hh', 'hl', 'll'}
        hh : High Correlation, High Dependence
        hl : High Correlation, Low Depedence
        ll : Low Correlation, Low Dependence
        
    num_points : int, optional (default=1000)
        Number points per variable generated.
        
    """
    rng = check_random_state(seed)    
    rng2 = check_random_state(seed+1)
    
    # Dataset I: High Correlation, High Depedence
    if dataset.lower() == 'hh':
        X = rng.rand(num_points, 1)
        Y = X + noise * rng.randn(num_points, 1)
    elif dataset.lower() == 'hl':
        t = 2 * np.pi * rng.rand(num_points, 1)
        X = np.cos(t) + noise * rng.randn(num_points, 1)
        Y = np.sin(t) + noise * rng.randn(num_points, 1)
    elif dataset.lower() == 'll':
        X = rng.rand(num_points, 1)
        Y = check_random_state(seed+1).rand(num_points, 1)
    else:
        raise ValueError(f'Unrecognized dataset: {dataset}')
    
    return X, Y




def main():
    pass


if __name__ == "__main__":
    pass
