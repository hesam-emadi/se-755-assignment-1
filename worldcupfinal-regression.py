import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
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


worldcup = pd.read_csv("./resource/world-cup-2018/worldcup-2018.csv", index_col=0)

worldcup.drop(['Location', 'Phase', 'Date', 'Team1_Ball_Possession(%)'], axis=1, inplace=True)
worldcup.describe()

# world cup attributes
worldcupAllFeatures = worldcup.iloc[:, np.arange(24)].copy()
# world cup goal result
scoreAsTarget = worldcup.iloc[:, 24].copy()
# wordl cup match result
w_results = worldcup.iloc[:, 25].copy()


# Create a class to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


numericFeatures = worldcupAllFeatures.drop(
    ['Team1', 'Team2', 'Team1_Continent', 'Team2_Continent', 'Normal_Time'], axis=1, inplace=False)

stringFeatures = worldcupAllFeatures[
    ['Team1', 'Team2', 'Team1_Continent', 'Team2_Continent', 'Normal_Time']].copy()

numericFeaturePipeline = Pipeline([
    ('selector', DataFrameSelector(list(numericFeatures))),
    ('imputer', Imputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

stringFeaturePipeline = Pipeline([
    ('selector', DataFrameSelector(list(stringFeatures))),
    ('cat_encoder', cs.OneHotEncoder(drop_invariant=True)),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", numericFeaturePipeline),
    ("cat_pipeline", stringFeaturePipeline),
])

preprocessedFeature = pd.DataFrame(data=full_pipeline.fit_transform(worldcupAllFeatures), index=np.arange(1, 65))
# worldcup_cleaned = pd.concat([feature_prepared, w_goals.to_frame(), w_results.to_frame()], axis=1)

# Split the data into training/testing sets
worldcupFeatureTrainingData, testData, worldcupTargetTrainingData, testTarget = \
    train_test_split(preprocessedFeature, scoreAsTarget, test_size=0.2, random_state=1)

predictLinearRegression(worldcupFeatureTrainingData, testData, worldcupTargetTrainingData, testTarget)
predictRidgeRegression(worldcupFeatureTrainingData, testData, worldcupTargetTrainingData, testTarget)