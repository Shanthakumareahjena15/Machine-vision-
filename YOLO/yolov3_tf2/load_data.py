from yolov3_tf2.save_model import Inference 
from yolov3_tf2 import data_preparation
import xgboost
import numpy as np 
import pandas as pd
#################### model inference ##########################################

rf_classifier = Inference.load_model(file_name = 'rf_classifier_vgg.pickle')
adaboost_classifier = Inference.load_model(file_name = 'adaboost_classifier.pickle')
lr_classifier = Inference.load_model(file_name = 'lr_classifier_vgg.pickle')
meta_classifier = Inference.load_model(file_name = 'meta_classifier_vgg.pickle.dat')


def validation(img):   
    validation_features = data_preparation.extract_feature_for_inference(img)
    rf_pred = rf_classifier.predict_proba(validation_features))
    lr_pred = lr_classifier.predict_proba(validation_features)
    svm_pred = svm_model.predict_proba(validation_features)
    meta_data =  np.concatenate((rf_pred, lr_pred, svm_pred), axis=1)
    meta_data = pd.DataFrame(meta_data)
    test = xgboost.DMatrix(meta_data)
    final_predictions = meta_classifier.predict(test) 
    final_predictions = final_predictions.tolist()
    final_predictions = int(final_predictions[0])
    
    return final_predictions

