import os
import glob
import numpy as np 
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Model
import pandas as pd
class data_preparation:

    def __init__(self, base, image_dir, layer_name, feature_name, label_name):
        self.base = base
        self.image_dir = image_dir
        self.layer_name = layer_name
        self.feature_name = feature_name
        self.label_name = label_name
        
    def image_to_array(self):     
        image = []
        lable = []
        for directory_path in glob.glob(self.base + self.image_dir):
            folder_name = directory_path.split("\\")[-1]   
            print(directory_path)             
            for img_path in glob.glob(os.path.join(directory_path, '*.jpg')):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)            
                image.append(img)
                lable.append(folder_name)   
        return np.array(image), np.array(lable) 
        
    def creat_label(self, label):
          
          self.label = label
          label_to_id = {v:i for i,v in enumerate(np.unique(self.label))}
          id_to_label = {v: k for k, v in label_to_id.items()}
          label_ids = np.array([label_to_id[x] for x in self.label])
          pd.DataFrame(label_ids).to_csv(self.label_name, index = False)
          return label_ids, id_to_label
    
    
    def VGG16_trained_model(self, shape):
        self.shape = shape
        features = np.zeros((self.shape,25088))
        pre_trained_model = VGG16(input_shape = (224, 224, 3), 
                                    include_top = False, 
                                    weights = 'imagenet')

        FC_layer_model= Model(inputs = pre_trained_model.input, outputs = pre_trained_model.get_layer(self.layer_name).output)

        i =0
        for directory_path in glob.glob(os.getcwd()+ self.image_dir):
            for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)    
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = np.expand_dims(img, axis=0)  
                FC_output = FC_layer_model.predict(img)
                FC_output = FC_output.flatten()   
                features[i]=FC_output            
                i+=1
        paandas_data_frame = pd.DataFrame(features)
        paandas_data_frame.to_csv(self.feature_name, index = False)
        return features 
