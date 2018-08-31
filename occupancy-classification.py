import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

worldcup = pd.read_csv("./resource/occupancy-sensor/occupancy_sensor_data.csv")

worldcup.drop(['date', 'HumidityRatio'], axis=1, inplace=True)
worldcup.describe()

# world cup attributes
X_occupancyAllFeatures = worldcup.iloc[:, np.arange(4)].copy()

# wordl cup match result
y_occupancyAsTarget = worldcup.iloc[:, 4].copy()

feature_prepared = X_occupancyAllFeatures

# Split the data into training/testing sets
worldcupFeatureTrainingData, testData, worldcupTargetTrainingData, testTarget = \
    train_test_split(feature_prepared, y_occupancyAsTarget, test_size=0.2, random_state=1)

# feature_prepared=pd.read_csv("SVM_demo.csv", index_col=74)

# Testing and training sentences splitting (stratified + shuffled) based on the index (sentence ID)
allFeatures = feature_prepared.index
targetData = y_occupancyAsTarget
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(allFeatures, targetData):
    train_ind, test_ind = allFeatures[train_index], allFeatures[test_index]
Test_Matrix = feature_prepared.loc[test_ind]
Test_Target_Matrix = y_occupancyAsTarget.loc[test_ind]
Train_Matrix = feature_prepared.loc[train_ind]
Train_Target_Matrix = y_occupancyAsTarget.loc[train_ind]

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

print("SVM: The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(
    100 * accuracy_score(Test_Target_Matrix, T_predict)))

## Perceptron ###############
clfPerceptron = Perceptron(n_iter=100)

clfPerceptron.fit(Train_Matrix, Train_Target_Matrix)

# Make predictions using the testing set
testDataPrediction = clfPerceptron.predict(Test_Matrix)

# Make predictions using the testing set
trainingDataPrediction = clfPerceptron.predict(Train_Matrix)

print("Perceptron: The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(
    100 * accuracy_score(Test_Target_Matrix, testDataPrediction)))

## Naive Bayes ###############
naiveBayesClassifier = GaussianNB()

naiveBayesClassifier.fit(Train_Matrix, Train_Target_Matrix)

# Make predictions using the testing set
testDataPrediction = naiveBayesClassifier.predict(Test_Matrix)

# Make predictions using the testing set
trainingDataPrediction = naiveBayesClassifier.predict(Train_Matrix)

print("Naive Bayes: The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(
    100 * accuracy_score(Test_Target_Matrix, testDataPrediction)))


## Decision trees ###############
dTreeClassifier = DecisionTreeClassifier()

dTreeClassifier.fit(Train_Matrix, Train_Target_Matrix)

# Make predictions using the testing set
testDataPrediction = dTreeClassifier.predict(Test_Matrix)

# Make predictions using the testing set
trainingDataPrediction = dTreeClassifier.predict(Train_Matrix)

print("Decision trees: The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(
    100 * accuracy_score(Test_Target_Matrix, testDataPrediction)))


## Nearest neighbour classifier ###############
nnClassifier = KNeighborsClassifier()

nnClassifier.fit(Train_Matrix, Train_Target_Matrix)

# Make predictions using the testing set
testDataPrediction = nnClassifier.predict(Test_Matrix)

# Make predictions using the testing set
trainingDataPrediction = nnClassifier.predict(Train_Matrix)

print("Nearest neighbour classifier: The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(
    100 * accuracy_score(Test_Target_Matrix, testDataPrediction)))
