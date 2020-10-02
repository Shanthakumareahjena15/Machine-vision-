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
  def __init__(self):
    pass
  def image_to_array():
    image = []
    lables = []
    for directory_path in glob.glob(base_dir):
      
    
