import tensorflow as tf
import os,cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_model():
  model = Sequential()
  model.add(Conv2D(32,3,padding='same',activation='relu',input_shape=[150, 150, 3]))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Dropout(0.5))
  model.add(Conv2D(32,3,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Dropout(0.5))
  model.add(Conv2D(64,3,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Conv2D(64,3,padding='same',activation='relu'))
  model.add(Dropout(0.3))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Conv2D(64,3,padding='same',activation='relu'))
  model.add(MaxPool2D(pool_size=2, strides=2, padding='valid'))
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(units=128,activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(units=64,activation='relu'))
  model.add(Dense(units=10,activation='softmax'))

  print(model.summary())
  return model


def plot_accuracy(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

def plot_losses(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()







