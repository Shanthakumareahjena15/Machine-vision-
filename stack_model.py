import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost import XGBClassifier
import pickle 

train_features = pd.read_csv('training_feature.csv')
train_labels = pd.read_csv('training_label.csv')
train_labelIds = pd.read_csv('labels_to_ids.csv')

train_1 = pd.read_csv('training_feature_1.csv')
train_2 = pd.read_csv('training_feature_2.csv')
train_3 = pd.read_csv('training_feature_3.csv')

test_1 = pd.read_csv('testing_feature_1.csv')
test_2 = pd.read_csv('testing_feature_2.csv')
test_3 = pd.read_csv('testing_feature_3.csv')


trainLabel_1 = pd.read_csv('train_label_1.csv')
trainLabel_2 = pd.read_csv('train_label_2.csv')
trainLabel_3 = pd.read_csv('train_label_3.csv')

trainLabelId_1 = pd.read_csv('train_label_id1.csv')
trainLabelId_2 = pd.read_csv('train_label_id2.csv')
trainLabelId_3 = pd.read_csv('train_label_id3.csv')

testLabel_1 = pd.read_csv('test_label_1.csv')
testLabel_2 = pd.read_csv('test_label_2.csv')
testLabel_3 = pd.read_csv('test_label_3.csv')

testLabelId_1 = pd.read_csv('test_label_id1.csv')
testLabelId_2 = pd.read_csv('test_label_id2.csv')
testLabelId_3 = pd.read_csv('test_label_id3.csv')

trainIndex_1 = pd.read_csv('train_index1.csv')
trainIndex_2 = pd.read_csv('train_index2.csv')
trainIndex_3 = pd.read_csv('train_index3.csv')

testIndex_1 = pd.read_csv('test_index1.csv')
testIndex_2 = pd.read_csv('test_index2.csv')
testIndex_3 = pd.read_csv('test_index3.csv')


class BaseModels:
    def __init__(self):
        pass
    
    def logistic_regression(self, features, labels, test_features, file_name):
        self.features = features
        self.labels = labels
        #self.label_ids = label_ids
        self.test_features = test_features
        self.file_name = file_name 
        
        lr_classifier  = LogisticRegression(solver='liblinear', random_state=10, C = 1.0, penalty = 'l2')
        lr_model_oneHot = lr_classifier.fit(features, labels)
        lr_model_proba = lr_classifier.fit(features, labels)
        if test_features is not None:            
            lr_predict = lr_model_oneHot.predict(test_features)
            lr_predic_proba = lr_model_proba.predict_proba(test_features)            
            return lr_predic_proba, lr_predict
        else:
            pickle_out = open(file_name, 'wb')
            pickle.dump(lr_model_proba, pickle_out)
            pickle_out.close()
           # return lr_model_proba
        
    def ada_boost(self, features, labels,  test_features, file_name):
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.file_name = file_name
        adaBoost = AdaBoostClassifier(random_state=1)
        adaBoost_model_one_hot = adaBoost.fit(features, labels) 
        adaBoost_model_proba = adaBoost.fit(features, labels)
        if test_features is not None:        
            adaBoost_oneHot_predict = adaBoost_model_one_hot.predict(test_features)
            adaBoost_predictproba = adaBoost_model_proba.predict_proba(test_features)
            return adaBoost_oneHot_predict, adaBoost_predictproba
        else:
            pickle_out = open(file_name, 'wb')
            pickle.dump(adaBoost_model_proba, pickle_out)
            pickle_out.close()
            #return adaBoost_model_proba
    
    def Randomforest(self, features, labels, test_features, file_name):
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_features = test_features
        self.file_name = file_name
        rf = RandomForestClassifier()
        rf_model_one_hot = rf.fit(features, labels) 
        rf_model_proba = rf.fit(features, labels)
        if test_features is not None:
            rf_oneHot_predict = rf_model_one_hot.predict(test_features)
            rf_predictproba = rf_model_proba.predict_proba(test_features)
            return rf_oneHot_predict, rf_predictproba
        else:
            pickle_out = open(file_name, 'wb')
            pickle.dump(rf_model_proba, pickle_out)
            pickle_out.close()
            #return rf_model_proba
        
    


class MetaTraining_data:
    
    def __init__(self, test_label_index1, test_label_index2, test_label_index3):
        self.test_label_index1 = test_label_index1
        self.test_label_index2 = test_label_index2
        self.test_label_index3 = test_label_index3
    
    def meta_training_data(self, test_pred_1, test_pred_2, test_pred_3 ):  
        self.test_pred_1 = test_pred_1
        self.test_pred_2 = test_pred_2
        self.test_pred_3 = test_pred_3
        prediction = np.zeros((434, 4))
        for i, j in zip (test_pred_1, np.array(self.test_label_index1)):
           prediction[j] = i
           
        for i, j in zip (test_pred_2, np.array(self.test_label_index2)):
           prediction[j] = i
        
        for i, j in zip (test_pred_3, np.array(self.test_label_index3)):
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
                
    def XGBoost_predictProba(self, concatenated_data, label, file_name):  
        self.concatenated_data = concatenated_data
        self.label = label
        self.file_name = file_name 
        params = {'max_depth':3, 'eta':0.01,'silent':1,  'num_class':4,'objective':'multi:softprob' } 
        meta_training_data_data =  xgboost.DMatrix(concatenated_data, label)
        model_xg = xgboost.train(params, meta_training_data_data, num_boost_round=100)
        pickle_out = open(file_name, 'wb')
        pickle.dump(lr_model_proba, pickle_out)
        pickle_out.close()
        return model_xg 
    
    def XGBoost_predict(self, concatenated_data, file_name):
         self.concatenated_data = concatenated_data
         self.file_name = file_name
         meta_testing_data =  xgboost.DMatrix(np.concatenate(( lr_prediction, ab_prediction, rf_prediction), axis=1))
         params = {'max_depth':3, 'eta':0.01,'silent':1,  'num_class':4,'objective':'multi:softmax' } 
         dataset = xgboost.DMatrix(concatenated_data, labels )
         model_xg = xgboost.train(params, dataset, num_boost_round=100)
         pickle_out = open(file_name, 'wb')
         pickle.dump(lr_model_proba, pickle_out)
         pickle_out.close()
         
         return model_xg

layer_1A = BaseModels()
metaTrainingData = MetaTraining_data( testIndex_1, testIndex_2, testIndex_3)
meta_xg = Metamodel()

predectionLabelRf_1, predectionProbaRf_1 = layer_1A.Randomforest(features = train_1, labels = trainLabel_1, test_features = test_1)
predectionLabelRf_2, predectionProbaRf_2 = layer_1A.Randomforest(features = train_2, labels = trainLabel_2, test_features = test_2)
predectionLabelRf_3, predectionProbaRf_3 = layer_1A.Randomforest(features = train_3, labels = trainLabel_3, test_features = test_3)

layer_1A.Randomforest(features = train_features, labels = train_labels, test_features = None, file_name = 'D:/Randomforest/figs/randomforest_b1.pickle')

predectionLabelab_1, predectionProbaab_1 = layer_1A.ada_boost(features = train_1, labels = trainLabel_1, test_features = test_1)
predectionLabelab_2, predectionProbaab_2 = layer_1A.ada_boost(features = train_2, labels = trainLabel_2, test_features = test_2)
predectionLabelab_3, predectionProbaab_3 = layer_1A.ada_boost(features = train_3, labels = trainLabel_3, test_features = test_3)

layer_1A.ada_boost(features = train_features, labels = train_labels, test_features = None, file_name = 'abaBoost_b2.pickle')

predectionLabelLr_1, predectionProbaLr_1 = layer_1A.logistic_regression(features = train_1, labels = trainLabel_1, test_features = test_1)
predectionLabelLr_2, predectionProbaLr_2 = layer_1A.logistic_regression(features = train_2, labels = trainLabel_2, test_features = test_2)
predectionLabelLr_3, predectionProbaLr_3 = layer_1A.logistic_regression(features = train_3, labels = trainLabel_3, test_features = test_3)

layer_1A.logistic_regression(features = train_features, labels = train, test_features = None, file_name = 'LogisticRegression_b3')

rf_dat_1 = metaTrainingData.meta_training_data(predectionProbaRf_1, predectionProbaRf_2, predectionProbaRf_3)
rf_dat_2 = metaTrainingData.meta_training_data(predectionProbaab_1, predectionProbaab_2, predectionProbaab_3)
rf_dat_3 = metaTrainingData.meta_training_data(predectionLabelLr_1, predectionLabelLr_2, predectionLabelLr_3)

con_data = metaTrainingData.Concatenate_testData(rf_dat_1, rf_dat_2, rf_dat_3)

model_xg = meta_xg.XGBoost_predict(con_data, train_labelIds, file_name = 'metaclassifier.dat')
