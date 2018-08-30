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

worldcup = pd.read_csv("./resource/world-cup-2018/worldcup-2018.csv", index_col=0)

worldcup.drop(['Location', 'Phase', 'Date', 'Team1_Ball_Possession(%)'], axis=1, inplace=True)
worldcup.describe()

# world cup attributes
worldcupAllFeatures = worldcup.iloc[:, np.arange(24)].copy()

# wordl cup match result
resultAsTargetStr = worldcup.iloc[:, 25].copy()


def mapResultToNumber(result):
    return {
        'win': 1,
        'draw': 2,
        'loss': 3
    }[result]

resultAsTarget = resultAsTargetStr.map(mapResultToNumber)

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

feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(worldcupAllFeatures), index=np.arange(1, 65))
# worldcup_cleaned = pd.concat([feature_prepared, w_goals.to_frame(), w_results.to_frame()], axis=1)

# Split the data into training/testing sets
worldcupFeatureTrainingData, testData, worldcupTargetTrainingData, testTarget = \
    train_test_split(feature_prepared, resultAsTarget, test_size=0.2, random_state=1)

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

# feature_prepared=pd.read_csv("SVM_demo.csv", index_col=74)

# Testing and training sentences splitting (stratified + shuffled) based on the index (sentence ID)
allFeatures = feature_prepared.index
targetData = resultAsTarget
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(allFeatures, targetData):
    train_ind, test_ind = allFeatures[train_index], allFeatures[test_index]
Test_Matrix = feature_prepared.loc[test_ind]
Test_Target_Matrix = resultAsTarget.loc[test_ind]
Train_Matrix = feature_prepared.loc[train_ind]
Train_Target_Matrix = resultAsTarget.loc[train_ind]

# data training with hyperparameter tuning for C
clf = Pipeline([
    ('std_scaler', StandardScaler()),
    ("svm", SVC())
])
param_grid = [
    {'svm__kernel': ['rbf'], 'svm__C': [2 ** x for x in range(0, 6)]},
]
inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
grid_search = GridSearchCV(clf, param_grid, cv=inner_cv, n_jobs=1, scoring='accuracy', verbose=3)
grid_search.fit(Train_Matrix, Train_Target_Matrix)

clf = grid_search.best_estimator_
# data testing
T_predict = clf.predict(Test_Matrix)

print("The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(
    100 * accuracy_score(Test_Target_Matrix, T_predict)))

# data training without hyperparameter tuning
clf = Pipeline([
    ('std_scaler', StandardScaler()),
    ("svm", SVC())
])
clf.fit(Train_Matrix, Train_Target_Matrix)
# data testing
T_predict = clf.predict(Test_Matrix)
