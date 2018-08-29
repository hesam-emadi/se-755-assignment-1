# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:01:09 2018

@author: i7
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:01:09 2018

@author: i7
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import category_encoders as cs
from sklearn.pipeline import FeatureUnion

worldcup=pd.read_csv("2018 worldcup.csv",index_col=0)
#match date is assumed to be irrelevant for the match results
worldcup.drop(['Date','Team1_Ball_Possession(%)'],axis=1,inplace=True)
worldcup.describe()

#world cup attributes
w_features=worldcup.iloc[:,np.arange(26)].copy()
#world cup goal result
w_goals=worldcup.iloc[:,26].copy()
#wordl cup match result
w_results=worldcup.iloc[:,27].copy()


# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames in this wise manner yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
#  w_features_num: numerical features
#  w_features_cat: categorical features 
w_features_num = w_features.drop(['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time'], axis=1,inplace=False)
w_features_cat=w_features[['Location','Phase','Team1','Team2','Team1_Continent','Team2_Continent','Normal_Time']].copy()


num_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(w_features_num))),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(list(w_features_cat))),
        ('cat_encoder', cs.OneHotEncoder(drop_invariant=True)),
    ])



full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


feature_prepared = pd.DataFrame(data=full_pipeline.fit_transform(w_features),index=np.arange(1,65))
worldcup_cleaned=pd.concat([feature_prepared,w_goals.to_frame(), w_results.to_frame()], axis=1)

