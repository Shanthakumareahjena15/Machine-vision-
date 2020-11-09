import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

skf = StratifiedKFold(n_splits=3,  random_state=None, shuffle=True)
skf.get_n_splits(train_data_array, train_label_ids)

train_features_pandas = pd.read_csv('training_data.csv')
test_features_pandas = pd.read_csv('testing_data.csv')
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
  
  
  def k_fold_data_seperation(test_index_1, test_index_2, test_index_3):        
    test_data_one = []
    test_data_two = []
    test_data_three = []
    for i in test_index_1:
        test_data_one.append(train_data_array[i])
    for i in test_index_2:
        test_data_two.append(train_data_array[i])
    for i in test_index_3:
        test_data_three.append(train_data_array[i])
    test_data_one_array = np.array(test_data_one)
    test_data_two_array = np.array(test_data_two)
    test_data_three_array = np.array(test_data_three)
    
    return test_data_one_array, test_data_two_array, test_data_three_array


  def train_label_index(train_label_ids):
    train_label_index_1 = []
    train_label_index_2 = []
    train_label_index_3 = []
    for i in train_index_1:
        train_label_index_1.append(train_label_ids[i])
    for i in train_index_2:
        train_label_index_2.append(train_label_ids[i])
    for i in train_index_3:
        train_label_index_3.append(train_label_ids[i])
    train_label_index_1 = pd.DataFrame(train_label_index_1) 
    train_label_index_1.to_csv( 'train_label_id_1.csv', index = False)
    train_label_index_2 = pd.DataFrame(train_label_index_2)
    train_label_index_2.to_csv('train_label_id_2.csv', index = False)
    train_label_index_3 =pd.DataFrame(train_label_index_3)
    train_label_index_3.to_csv('train_label_id_3.csv', index = False)
    
    return train_label_index_1, train_label_index_2, train_label_index_3



def train_label_k(train_label):
    train_label_1 = []
    train_label_2 = []
    train_label_3 = []
    for i in train_index_1:
        train_label_1.append(train_label[i])
    for i in train_index_2:
        train_label_2.append(train_label[i])
    for i in train_index_3:
        train_label_3.append(train_label[i])
    train_label_index_1 = pd.DataFrame(train_label_1) 
    train_label_index_1.to_csv( 'train_label_1.csv', index = False)
    train_label_index_2 = pd.DataFrame(train_label_2)
    train_label_index_2.to_csv('train_label_2.csv', index = False)
    train_label_index_3 =pd.DataFrame(train_label_3)
    train_label_index_3.to_csv('train_label_3.csv', index = False)
    
    return pd.DataFrame(train_label_1), pd.DataFrame(train_label_2), pd.DataFrame(train_label_3)

train_label_1, train_label_2, train_label_3 = train_label_k(train_label= train_label)



def test_label_index(train_label_ids):
    test_label_index_1 = []
    test_label_index_2 = []
    test_label_index_3 = []
    for i in test_index_1:
        test_label_index_1.append(train_label_ids[i])
    for i in test_index_2:
        test_label_index_2.append(train_label_ids[i])
    for i in test_index_3:
        test_label_index_3.append(train_label_ids[i])
    test_label_index_1 = pd.DataFrame(test_label_index_1) 
    test_label_index_1.to_csv( 'test_label_id_1.csv', index = False)
    test_label_index_2 = pd.DataFrame(test_label_index_2)
    test_label_index_2.to_csv('test_label_id_2.csv', index = False)
    test_label_index_3 =pd.DataFrame(test_label_index_3)
    test_label_index_3.to_csv('test_label_id_3.csv', index = False)
    return test_label_index_1, test_label_index_2, test_label_index_3


test_label_index_k1, test_label_index_k2, test_label_index_k3 = test_label_index(train_label_ids)




def test_label_k(train_label):
    test_label_1 = []
    test_label_2 = []
    test_label_3 = []
    for i in test_index_1:
        test_label_1.append(train_label[i])
    for i in test_index_2:
        test_label_2.append(train_label[i])
    for i in test_index_3:
        test_label_3.append(train_label[i])
    train_label_index_1 = pd.DataFrame(test_label_1) 
    train_label_index_1.to_csv( 'test_label_1.csv', index = False)
    train_label_index_2 = pd.DataFrame(test_label_2)
    train_label_index_2.to_csv('test_label_2.csv', index = False)
    train_label_index_3 =pd.DataFrame(test_label_3)
    train_label_index_3.to_csv('test_label_3.csv', index = False)
    
    return pd.DataFrame(test_label_1), pd.DataFrame(test_label_2), pd.DataFrame(test_label_3)

test_label_1, test_label_2, test_label_3 = test_label_k(train_label= train_label)



