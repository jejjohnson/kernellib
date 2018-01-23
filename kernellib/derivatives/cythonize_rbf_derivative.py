from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'RBF Derivative Cython',
    ext_modules = cythonize("rbf_derivative_cy.pyx"),
    include_dirs = [numpy.get_include()]
    
)