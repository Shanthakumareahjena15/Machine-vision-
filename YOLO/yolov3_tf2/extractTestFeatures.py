import os
import glob
import cv2
import numpy as np
from keras.layers import Input
from keras.models import Model
import pandas as pd
from keras.applications.vgg16 import VGG16





class Inference:
    def __init__(self):
        pass
    
    def VGG16_trained_model(self, layer_name = 'block5_pool'):
    
    
        self.layer_name = layer_name
        pre_trained_model = VGG16(input_shape = (224, 224, 3), 
                                    include_top = False, 
                                    weights = 'imagenet')

        FC_layer_model= Model(inputs = pre_trained_model.input, outputs = pre_trained_model.get_layer(layer_name).output)
        return FC_layer_model
    
    def extract_feature(self, img):
        shape = 1
        i = 0
        self.img = img 
        img = cv2.resize(img, (224, 224))
        features = np.zeros((1,25088))
        FC_layer_model = Inference()
        FC_layer_model =FC_layer_model.VGG16_trained_model(layer_name = 'block5_pool')
        img = np.expand_dims(img, axis=0)  
        FC_output = FC_layer_model.predict(img)
        FC_output = FC_output.flatten()   
        features[i]=FC_output 
        
        return features 






    
    
