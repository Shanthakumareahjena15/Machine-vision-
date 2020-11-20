# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:44:15 2020

@author: shanthakumar
"""
import pickle 


# import pickle
# pickle_xgb = open('meta_classifier.pickle', 'wb')
# pickle.dump(model_xg, pickle_xgb)
# pickle_xgb.close()

# load_meta_clasifier =open('meta_classifier.pickle', 'rb')
# loaded_meta_classifier = pickle.load(load_meta_clasifier)



def save_model(file_name, model_name):
    pickle_out = open(file_name, 'wb')
    pickle.dump(model_name, pickle_out)
    pickle_out.close()
    
    
def load_model(file_name):
    load_model = open(file_name, 'rb')
    loaded_model = pickle.load(load_model)
    return loaded_model
    
    