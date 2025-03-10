#Dataframe created 

!pip install keras_preprocessing
from keras.utils import to_categorical
from keras_preprocessing.image import load_img 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np


TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'


def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths,labels

#Initially coded in Google Colab waiting for the dataset 
