import os
import glob
import cv2
import numpy as np
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
import pandas as pd

from keras.applications.vgg16 import VGG16


#data preparation 

def data(base_dir, image_dir):   
    image = []
    lable = []
    for directory_path in glob.glob(base_dir + image_dir):
        folder_name = directory_path.split("\\")[-1]   
        print(directory_path)             
        for img_path in glob.glob(os.path.join(directory_path, '*.jpg')):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            img = cv2.resize(img, (150, 150))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)            
            image.append(img)
            lable.append(folder_name)   
    return np.array(image), np.array(lable)


def creat_id(lable):
    label_to_id = {v:i for i,v in enumerate(np.unique(lable))}
    id_to_label = {v: k for k, v in label_to_id.items()}
    label_ids = np.array([label_to_id[x] for x in lable])
    return label_ids, id_to_label
    
        
        
#Feature extraction using CNN
    
def trained_model(layer_name = 'block5_pool'):
    pre_trained_model = VGG16(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')
                                
 
    FC_layer_model= Model(inputs = pre_trained_model.input, outputs = pre_trained_model.get_layer(layer_name).output)
    #print(pre_trained_model.summary())
    return FC_layer_model



def extract_feature(shape,image_dir):
    features = np.zeros((shape,25088))
    FC_layer_model = trained_model()
    i =0
    for directory_path in glob.glob(os.getcwd()+ image_dir):
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)    
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #img = img -127.5
            #img = img - 1
            img = np.expand_dims(img, axis=0)  
            FC_output = FC_layer_model.predict(img)
            FC_output = FC_output.flatten()   
            features[i]=FC_output            
            i+=1
        #print(features)
    return features 



'''train feature to pandas data frame''' 
    
def features_in_padas_dataframe(data, name):
    #feature_col = []
    #for i in range(37632):
     #   feature_col.append('f_'+str(i))
      #  i+=1
    #np.array(feature_col)
    paandas_data_frame = pd.DataFrame(data)#, columns = feature_col)
    paandas_data_frame = paandas_data_frame.to_csv(name, index = False)
    return paandas_data_frame





def extract_feature_for_inference(img):
    #img = np.float32(img)
    img = cv2.resize(img, (224, 224))
    i = 0
    features = np.zeros((1,25088))
    FC_layer_model = trained_model()
    img = np.expand_dims(img, axis=0)  
    FC_output = FC_layer_model.predict(img)
    FC_output = FC_output.flatten()   
    features[i]=FC_output 
    i +=1
    return features
    















