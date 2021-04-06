import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

#creazione del modello per la CNN
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

#rissunto accuratezza
def plot_accuracy(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

#riassunto valori persi
def plot_losses(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()







