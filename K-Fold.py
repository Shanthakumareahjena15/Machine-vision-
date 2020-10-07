import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

skf = StratifiedKFold(n_splits=3,  random_state=None, shuffle=True)
skf.get_n_splits(train_data_array, train_label_ids)

train_features_pandas = pd.read_csv('cod_her_train_xg.csv')
test_features_pandas = pd.read_csv('cod_her_test_xg.csv')
train_labels = pd.read_csv('train_label.csv')
test_labels = pd.read_csv('test_label.csv')


class k_fold:
  def_init__(self):
    pass
  def index_split(train_features_pandas, train_label_ids):
    i = 0
     for train_index, test_index in skf.split(train_features_pandas, train_label_ids):
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
  
  def k_fold_data_seperation(train_index_1, train_index_2, train_index_3):        
    train_data_one = []
    train_data_two = []
    train_data_three = []
    for i in train_index_1:
        train_data_one.append(train_data_array[i])
    for i in train_index_2:
        train_data_two.append(train_data_array[i])
    for i in train_index_3:
        train_data_three.append(train_data_array[i])
    train_data_one_array = np.array(train_data_one)
    train_data_two_array = np.array(train_data_two)
    train_data_three_array = np.array(train_data_three)
    
    return train_data_one_array, train_data_two_array, train_data_three_array
