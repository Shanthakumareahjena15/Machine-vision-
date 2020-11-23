
import numpy as np 
import cv2
from keras.applications.vgg16 import VGG16
from keras.models import Model
import pandas as pd


def VGG16_trained_model(shape):
        self.shape = shape
        features = np.zeros((shape,25088))
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
       
        return features 
