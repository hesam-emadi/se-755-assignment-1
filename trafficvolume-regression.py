import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, precision_recall_curve, average_precision_score
import category_encoders as cs
from sklearn.pipeline import FeatureUnion

from sklearn.linear_model import Ridge


def predictLinearRegression(trainingData, testData, trainingTarget, testTarget):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(trainingData, trainingTarget)

    # Make predictions using the testing set
    testDataPrediction = regr.predict(testData)
    # Make predictions using the testing set
    trainingDataPrediction = regr.predict(trainingData)

    # Trying to calculate precision and recall
    # score = regr.decision_function(testData)
    #     print("average_precision_score: %.2f"
    #           % average_precision_score(testTarget, score))

    print(' ')
    # The coefficients
    print('Coefficients and Intercept are: ', regr.coef_, "   ", regr.intercept_, ' respectively')
    # The mean squared error
    print('** Linear Regression *******************************************************')
    print("Mean squared error for testing data: %.2f"
          % mean_squared_error(testTarget, testDataPrediction))

    # Explained variance score: 1 is perfect prediction
    print('Variance score for testing data: %.2f' % r2_score(testTarget, testDataPrediction))
    print('******************************************************* ')
    print("Mean squared error for training data: %.2f"
          % mean_squared_error(trainingTarget, trainingDataPrediction))
    # Explained variance score: 1 is perfect prediction
    print('Variance score for training data: %.2f' % r2_score(trainingTarget, trainingDataPrediction))


def predictRidgeRegression(trainingData, testData, trainingTarget, testTarget):
    # Create linear regression object
    regr = Ridge(alpha=0.1)

    # Train the model using the training sets
    regr.fit(trainingData, trainingTarget)

    # Make predictions using the testing set
    testDataPrediction = regr.predict(testData)
    # Make predictions using the testing set
    trainingDataPrediction = regr.predict(trainingData)

    print(' ')
    # The coefficients
    print('Coefficients and Intercept are: ', regr.coef_, "   ", regr.intercept_, ' respectively')
    # The mean squared error
    print('** Ridge Regression *******************************************************')
    print("Mean squared error for testing data: %.2f"
          % mean_squared_error(testTarget, testDataPrediction))
    # Explained variance score: 1 is perfect prediction
    print('Variance score for testing data: %.2f' % r2_score(testTarget, testDataPrediction))
    print('******************************************************* ')
    print("Mean squared error for training data: %.2f"
          % mean_squared_error(trainingTarget, trainingDataPrediction))
    # Explained variance score: 1 is perfect prediction
    print('Variance score for training data: %.2f' % r2_score(trainingTarget, trainingDataPrediction))


traffic = pd.read_csv("./resource/traffic-flow/traffic_flow_data.csv")
#
# Separating feature from target
trafficAllFeatures = traffic.iloc[:, np.arange(450)].copy()

# target which is section 23 at t+1 time
targetData = traffic.iloc[:, 450].copy()

# Split the data into training/testing sets
trafficFeatureTrainingData, trafficTestData, trafficTargetTrainingData, trafficTestTarget = \
    train_test_split(trafficAllFeatures, targetData, test_size=0.2, random_state=1)

predictLinearRegression(trafficFeatureTrainingData, trafficTestData, trafficTargetTrainingData, trafficTestTarget)
predictRidgeRegression(trafficFeatureTrainingData, trafficTestData, trafficTargetTrainingData, trafficTestTarget)
