# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:25:02 2018

@author: tli725
"""

import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

#######################################################version 1#############################################################
# Use only one feature
diabetes_X = diabetes.data[:, [2]]

# one target attribute
diabetes_y = diabetes.target

# Split the data into training/testing sets
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = \
    train_test_split(diabetes_X, diabetes_y, test_size=0.2, random_state=1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
# Make predictions using the testing set
diabetes_y_train_pred = regr.predict(diabetes_X_train)

print(' ')
# The coefficients
print('Coefficients and Intercept are: ', regr.coef_, "   ", regr.intercept_, ' respectively')
# The mean squared error
print('_________________###################____________________')
print("Mean squared error for testing data: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for testing data: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
print('******************************************************* ')
print("Mean squared error for training data: %.2f"
      % mean_squared_error(diabetes_y_train, diabetes_y_train_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score for training data: %.2f' % r2_score(diabetes_y_train, diabetes_y_train_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(np.arange(start=-0.1, stop=0.2, step=0.06))
plt.yticks(np.arange(500, step=100))

plt.show()

#######################################################version 2#############################################################
from mpl_toolkits.mplot3d import Axes3D

# Use two features
diabetes_X = diabetes.data[:, [3, 2]]

# one target attribute
diabetes_y = diabetes.target

# Split the data into training/testing sets
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = \
    train_test_split(diabetes_X, diabetes_y, test_size=0.2, random_state=1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients and Intercept are: ', regr.coef_, "   ", regr.intercept_, ' respectively')
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
fig = plt.figure(1, figsize=(9, 6))
ax = Axes3D(fig)
# plot testing points
ax.scatter(diabetes_X_test[:, 0], diabetes_X_test[:, 1], diabetes_y_test, depthshade=False, c='black')

x = np.arange(-0.2, 0.2, 0.05)
N = x.size
a, b = np.meshgrid(x, x)
it = np.array([b.ravel(), a.ravel(), np.ones(N * N)]).T
w = np.append(regr.coef_, regr.intercept_)
result = it.dot(w)
# plot fitted hyperplane
ax.plot_surface(np.reshape(it[:, 0], (N, N)), np.reshape(it[:, 1], (N, N)), np.reshape(result, (N, N)), shade=False,
                color=(0., 1., 0., 0.2))
# set axis
ax.set_title("Two features regression")
ax.set_xlabel("4th feature")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("3rd feature")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("target")
ax.w_zaxis.set_ticklabels(np.arange(500, step=100))
plt.show()
