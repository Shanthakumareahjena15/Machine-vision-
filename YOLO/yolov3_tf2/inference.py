# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:44:15 2020

@author: shanthakumar
"""
import pickle 
import numpy as np 
import pandas as pd
import xgboost
from yolov3_tf2.extractTestFeatures  import Inference

class production:
    def __init__(self):
        pass
        
    def load_model(self, file_name):
        self.file_name = file_name
        load_model = open(file_name, 'rb')
        loaded_model = pickle.load(load_model)
        return loaded_model
        
    def validation(self, img):
        self.img = img
        
        rf_classifier = production()
        rf_classifier = rf_classifier.load_model(file_name = 'randomforest_b1.pickle')
        adaboost_classifier = production()
        adaboost_classifier = adaboost_classifier.load_model(file_name = 'abaBoost_b2.pickle')
        lr_classifier = production()
        lr_classifier = lr_classifier.load_model(file_name = 'LogisticRegression_b3.pickle')
        meta_classifier = production()
        meta_classifier = meta_classifier.load_model(file_name = 'metaclassifier.dat')

        
        #validation_features = Inference.VGG16_trained_model(self, layer_name = 'block5_pool')
        validation_features = Inference()
        validation_features = validation_features.extract_feature(img)
        
        rf_pred = rf_classifier.predict_proba(validation_features)
        lr_pred = lr_classifier.predict_proba(validation_features)
        ab_pred = adaboost_classifier.predict_proba(validation_features)
        meta_data =  np.concatenate((rf_pred, ab_pred, lr_pred  ), axis=1)

        #meta_data = pd.DataFrame(meta_data)

        test = xgboost.DMatrix(meta_data)
        print(test)
        final_predictions = meta_classifier.predict(test) 
        final_predictions = final_predictions.tolist()
        final_predictions = int(final_predictions[0])
        return final_predictions
           
       
       


    
    