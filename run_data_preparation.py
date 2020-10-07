import data_preparation


train_data= data_preparation(base = 'D:\Randomforest', image_dir = '/training_dataset/*', layer_name = 'block5_pool', feature_name = 'train_features.csv', label_name = 'tarin_label.csv')
image, label = data_preparation.image_to_array(train_data)
label_ids, id_to_label = train_data.creat_label(label = label)
features = train_data.VGG16_trained_model(shape = np.array(image).shape[0])

test_data =  data_preparation(base ='D:\Randomforest',  image_dir = '/Validation-Example1/*', layer_name = 'block5_pool', feature_name = 'test_features.csv', label_name = 'test_label.csv')
image_test, label_test = data_preparation.image_to_array(test_data)
label_ids_test, id_to_label_test = test_data.creat_label(label_test)
test_data.VGG16_trained_model(shape = np.array(image).shape[0])
