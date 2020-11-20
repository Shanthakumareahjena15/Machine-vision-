from yolov3_tf2 import save_model 
from yolov3_tf2 import data_preparation
import xgboost
import numpy as np 
import pandas as pd
#################### model inference ##########################################

#rf_classifier = save_model.load_model(file_name = 'rf_classifier_vgg.pickle')

#adaboost_classifier = save_model.load_model(file_name = 'adaboost_classifier.pickle')

lr_classifier = save_model.load_model(file_name = 'lr_classifier_vgg.pickle')

svm_model = save_model.load_model(file_name = 'svm_classifier_vgg.pickle')

meta_classifier = save_model.load_model(file_name = 'meta_classifier_vgg_2.pickle.dat')


def validation(img):   
    validation_features = data_preparation.extract_feature_for_inference(img)
    #print(validation_features.shape)
    #validation_features = data_preparation.features_in_padas_dataframe(data = validation_features, name = 'val.csv')
    #validation_features = pd.read_csv('val.csv')
    #rf_pred = rf_classifier.predict_proba(validation_features)
    #print(rf_pred)
    #adaboost_pred = adaboost_classifier.predict_proba(validation_features)
    #print(adaboost_pred)
    lr_pred = lr_classifier.predict_proba(validation_features)
    #print(lr_pred)
    svm_pred = svm_model.predict_proba(validation_features)
   # print(svm_pred)
    meta_data =  np.concatenate(( lr_pred, svm_pred), axis=1)
    #print(meta_data[0])
    meta_data = pd.DataFrame(meta_data)
    test = xgboost.DMatrix(meta_data)
    final_predictions = meta_classifier.predict(test) 
    print(final_predictions)
    #print("It's",train_id_to_label[predictions[0]])

