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
    
    
    class MetaTraining_data:
    
    def __init__(self, test_label_index1, test_label_index2, test_label_index3):
        self.test_label_index1 = test_label_index1
        self.test_label_index2 = test_label_index2
        self.test_label_index3 = test_label_index3
    
    def meta_training_data(self, test_pred_1, test_pred_2, test_pred_3 ):  
        self.test_pred_1 = test_pred_1
        self.test_pred_2 = test_pred_2
        self.test_pred_3 = test_pred_3
        prediction = np.zeros((1117, 4))
        for i, j in zip (test_pred_1, np.array(test_label_index1)):
           prediction[j] = i
           
        for i, j in zip (test_pred_2, np.array(test_label_index2)):
           prediction[j] = i
        
        for i, j in zip (test_pred_3, np.array(test_label_index3)):
           prediction[j] = i
           
        return prediction
    
  
    def Concatenate_testData(self, lr_prediction, ab_prediction, rf_prediction):
        self.lr_prediction = lr_prediction
        self.ab_prediction = ab_prediction
        self.rf_prediction = rf_prediction
        concatenated_data =  np.concatenate(( lr_prediction, ab_prediction, rf_prediction), axis=1)
        
        return concatenated_data




class Metamodel:
    def __init__(self):
        pass

        
        def MetaTraining_data():
            
        
        def XGBoost_predictProba(self, concatenated_data):  
            self.concatenated_data = concatenated_data
            params = {'max_depth':3, 'eta':0.01,'silent':1,  'num_class':4,'objective':'multi:softprob' } 
            meta_training_data_data =  xgboost.DMatrix(concatenated_data, label_ids)
            model_xg = xgboost.train(params, dataset, num_boost_round=100)           
            return model_xg 
        def XGBoost_predict(self):
             meta_testing_data =  xgboost.DMatrix(np.concatenate(( lr_prediction, ab_prediction, rf_prediction), axis=1))
             params = {'max_depth':3, 'eta':0.01,'silent':1,  'num_class':4,'objective':'multi:softmax' } 
             dataset = xgboost.DMatrix(meta_training_data, labels )
             model_xg = xgboost.train(params, dataset, num_boost_round=100)
             xg_train_one_hot = model_xg.predict(dataset)

