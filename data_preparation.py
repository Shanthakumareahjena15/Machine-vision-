import os
import glob
import cv2
import numpy as np
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16

class data_preparation:
    def __init__(self, base, image_dir):
        self.base = base
        self.image_dir = image_dir
    
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
    
    
