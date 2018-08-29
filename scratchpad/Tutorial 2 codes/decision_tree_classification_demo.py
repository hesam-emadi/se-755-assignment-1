# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:58:29 2018

@author: i7
"""
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=1)
grid_search_cv.fit(X_train, y_train)
grid_search_cv.best_estimator_

from sklearn.metrics import accuracy_score
y_pred = grid_search_cv.predict(X_test)
print("The prediction accuracy using the decision tree is : {:.2f}%.".format(100*accuracy_score(y_test, y_pred)))












