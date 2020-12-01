import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from data_preparation import data_preparation
import os

skf = StratifiedKFold(n_splits=3,  random_state=None, shuffle=True)

class kFold(data_preparation):
    def __init__(self, base_dir, image_dir, layer_name, feature_name, label_name):
        super().__init__(base_dir, image_dir, layer_name, feature_name, label_name)
        
   
    def index_sorting(self, features, label_ids):
        self.features = features
        self.label_ids = label_ids
        i = 0
        for train_index, test_index in skf.split(features, label_ids):
            if i == 0:
                train_index_1 = train_index 
                test_index_1 = test_index
            elif i == 1:
                train_index_2 = train_index 
                test_index_2 = test_index
            elif i == 2:
                train_index_3 = train_index
                test_index_3 = test_index
        
            i += 1
        return train_index_1, test_index_1, train_index_2, test_index_2, train_index_3, test_index_3
    
    def data_seperation(self, index_1, index_2, index_3, feature):  
        self.feature = feature
        self.index_1 = index_1
        self.index_2 = index_2
        self.index_3 = index_3
        data_one = []
        data_two = []
        data_three = []
        
        for i in index_1:
            data_one.append(feature[i])
        for i in index_2:
            data_two.append(feature[i])
        for i in index_3:
            data_three.append(feature[i])
            
        feature_1 = np.array(data_one)
        feature_2 = np.array(data_two)
        feature_3 = np.array(data_three)
    
        return feature_1, feature_2, feature_3
    
    def label_id(self, label_ids, index_1, index_2, index_3):
        self.label_ids = label_ids
        self.index_1 = index_1
        self.index_2 = index_2
        self.index_3 = index_3
        label_index_1 = []
        label_index_2 = []
        label_index_3 = []
        for i in index_1:
            label_index_1.append(label_ids[i])
        for i in index_2:
            label_index_2.append(label_ids[i])
        for i in index_3:
            label_index_3.append(label_ids[i])
                   
        return label_index_1, label_index_2, label_index_3
    
        

class lable:
     def label(self, labels, index_1, index_2, index_3):
        self.labels = labels
        self.index_1 = index_1
        self.index_2 = index_2
        self.index_3 = index_3
        label_index_1 = []
        label_index_2 = []
        label_index_3 = []
        for i in index_1:
            label_index_1.append(labels[i])
        for i in index_2:
            label_index_2.append(labels[i])
        for i in index_3:
            label_index_3.append(labels[i])
            
        
        return label_index_1, label_index_2, label_index_3

obj_1= kFold( os.getcwd(), '/training_dataset/*', 'block5_pool', 'tarining.csv',  'training_label.csv')

training_image, training_label  = obj_1.image_to_array()
training_feature = obj_1.VGG16_trained_model(training_image.shape[0])
labels_to_ids, ids_to_labels = obj_1.creat_label(training_label)

train_index_1, test_index_1, train_index_2, test_index_2, train_index_3, test_index_3 = obj_1.index_sorting(training_feature, labels_to_ids)

training_feature_1, training_feature_2, training_feature_3 = obj_1.data_seperation(train_index_1, train_index_2, train_index_3, training_feature)
testing_feature_1, testing_feature_2, testing_feature_3 = obj_1.data_seperation(test_index_1, test_index_2, test_index_3, training_feature)

train_label_id1, train_label_id2, train_label_id3 = obj_1.label_id(labels_to_ids, train_index_1, train_index_2, train_index_3)
test_label_id1, test_label_id2, test_label_id3 = obj_1.label_id(labels_to_ids, test_index_1, test_index_2, test_index_3)


label_creation= lable()
train_label_1, train_label_2, train_label_3 = label_creation.label(training_label, train_index_1, train_index_2, train_index_3)
test_label_1, test_label_2, test_label_3 = label_creation.label(training_label, test_index_1, test_index_2, test_index_3)


pd.DataFrame(training_label).to_csv('training_label.csv', index= False)
pd.DataFrame(training_feature).to_csv('Training_feature.csv', index= False)
pd.DataFrame(labels_to_ids).to_csv('training_label_ids.csv', index= False)

pd.DataFrame(training_feature_1).to_csv('training_feature_1.csv', index= False)
pd.DataFrame(training_feature_2).to_csv('training_feature_2.csv', index= False)
pd.DataFrame(training_feature_3).to_csv('training_feature_3.csv', index= False)

pd.DataFrame(testing_feature_1).to_csv('testing_feature_1.csv', index= False)
pd.DataFrame(testing_feature_2).to_csv('testing_feature_2.csv', index= False)
pd.DataFrame(testing_feature_3).to_csv('testing_feature_3.csv', index= False)


pd.DataFrame(train_label_1).to_csv('train_label_1.csv', index= False)
pd.DataFrame(train_label_2).to_csv('train_label_2.csv', index= False)
pd.DataFrame(train_label_3).to_csv('train_label_3.csv', index= False)

pd.DataFrame(train_label_id1).to_csv('train_label_id1.csv', index= False)
pd.DataFrame(train_label_id2).to_csv('train_label_id2.csv', index= False)
pd.DataFrame(train_label_id3).to_csv('train_label_id3.csv', index= False)

pd.DataFrame(train_index_1).to_csv('train_index1.csv', index= False)
pd.DataFrame(train_index_2).to_csv('train_index2.csv', index= False)
pd.DataFrame(train_index_3).to_csv('train_index3.csv', index= False)

pd.DataFrame(test_label_1).to_csv('test_label_1.csv', index= False)
pd.DataFrame(test_label_2).to_csv('test_label_2.csv', index= False)
pd.DataFrame(test_label_3).to_csv('test_label_3.csv', index= False)

pd.DataFrame(test_label_id1).to_csv('test_label_id1.csv', index= False)
pd.DataFrame(test_label_id2).to_csv('test_label_id2.csv', index= False)
pd.DataFrame(test_label_id3).to_csv('test_label_id3.csv', index= False)


pd.DataFrame(test_index_1).to_csv('test_index1.csv', index= False)
pd.DataFrame(test_index_2).to_csv('test_index2.csv', index= False)
pd.DataFrame(test_index_3).to_csv('test_index3.csv', index= False)

