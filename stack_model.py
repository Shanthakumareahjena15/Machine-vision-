import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost
from xgboost import XGBClassifier

train_features_pandas = pd.read_csv('Training_feature.csv')
train_labels = pd.read_csv('training_label.csv')

#test_features_pandas = pd.read_csv('cod_her_test_xg.csv')


class BaseModels:
    def __init__(self):
        pass
    
    def logistic_regression(self, features, labels, label_ids, test_features):
        self.features = features
        self.labels = labels
        self.label_ids = label_ids
        self.test_features = test_features
        
        lr_classifier  = LogisticRegression(solver='liblinear', random_state=10, C = 1.0, penalty = 'l2')
        lr_model_oneHot = lr_classifier.fit(features, label)
        lr_model_proba = lr_classifier.fit(features. label_ids)
        lr_predict = lr_model_oneHot.predict(test_features)
        lr_predic_proba = lr_model_proba.predict_proba(test_features)
        
        return lr_predic_proba, lr_predict
        
    def ada_boost(self, features, labels, label_ids, test_features):
        self.features = features
        self.labels = labels
        self.test_features = test_features
        adaBoost = AdaBoostClassifier(random_state=1)
        adaBoost_model_one_hot = adaBoost.fit(features, labels) 
        adaBoost_model_proba = adaBoost.fit(features, label_ids)
        adaBoost_oneHot_predict = adaBoost_model_one_hot.predict(test_fetrures)
        adaBoost_predictproba = adaBoost_model_proba.predict_proba(test_features)
        return adaBoost_oneHot_predict, adaBoost_predictproba
    
    def Randomforest(self, features, labels, label_ids, test_features):
        self.features = features
        self.labels = labels
        self.test_features = test_features
        rf = RandomForestClassifier()
        rf_model_one_hot = rf.fit(features, labels) 
        rf_model_proba = rf.fit(features, label_ids)
        rf_oneHot_predict = rf_model_one_hot.predict(test_fetrures)
        rf_predictproba = rf_model_proba.predict_proba(test_features)
        return rf_oneHot_predict, rf_predictproba
