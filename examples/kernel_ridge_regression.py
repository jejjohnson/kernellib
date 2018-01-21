#%%
import numpy as np
from sklearn.model_selection import train_test_split
import scipy as scio
from scipy.spatial.distance import pdist
from scipy.linalg import cho_factor, cho_solve, cholesky
from sklearn.metrics.pairwise import rbf_kernel
from time import time

%matplotlib inline
import matplotlib.pyplot as plt

#%%
# generate datasets
random_state = 123
num_points = 1000

x_data = np.arange(0, num_points)
y_data = np.sin(0.01 * x_data)

# split data into training and testing
train_percent = 0.5

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=train_percent,
    random_state=random_state
)

# plot the training data
fig, ax = plt.subplots()

ax.scatter(x_train[::5], y_train[::5], color='k', label='Training')
ax.scatter(x_test[::5], y_test[::5], color='r', marker='+', label='Testing')

ax.legend()
plt.show()

#%%

