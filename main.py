import cv2
from glob import glob
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from tqdm import tqdm
import  os
import sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
import dask.array as da
import pandas as pd
import extract_feature
import data_augmentation
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import Cnn_function
from keras.utils.np_utils import to_categorical
import utility



TRAIN_DIR = "C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\training_set"
TEST_DIR = "C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\test_set"
train_labels = os.listdir(TRAIN_DIR)
test_labels = os.listdir(TEST_DIR)
resize = 400
X = []
y = []
x = []
imgsize = 150


for training_name in train_labels:
    label = training_name
    DIR = os.path.join(TRAIN_DIR, training_name)

    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR, img)
        img = cv2.imread(path)
        img1 = cv2.imread(path)
        img = cv2.resize(img, (resize,resize))
        img1 = cv2.resize(img, (imgsize, imgsize))
        glob_features_b = extract_feature.extract(img)
        X.append(np.array(glob_features_b))
        x.append(np.array(img1))
        y.append(str(label))

z = []
z = y

'normalizzazione'
X = np.array(X)
#y = np.array(y)
X = X/255

#Label Encoding
le= LabelEncoder()
y = le.fit_transform(y)

# split il training e il test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#################################### Modello Gaussiano
gnb = GaussianNB()                 #
gnb.fit(X_train, y_train)          #
y_pred = gnb.predict(X_test)       #
print(X_test.shape)
####################################

#creo il modello ExtraTreeClassifier e lo addestro()
extra_clf = ExtraTreesClassifier()
extra_clf.fit(X_train, y_train)
extra_clf_pred = extra_clf.predict(X_test)

#creo il modello RandomForest
rfl = RandomForestClassifier()
rfl.fit(X_train, y_train)
rfl_pred = rfl.predict(X_test)

#modello SVC
#svclassifier = SVC(kernel='linear')
#svclassifier.fit(X_train, y_train)
#svclassifier_pred = svclassifier.predict(X_test)

#KNN
classifierKNN = KNeighborsClassifier(n_neighbors=5)
classifierKNN.fit(X_train, y_train)
KNN_pred = classifierKNN.predict(X_test)


print("Accuracy GausianN:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy RFL:",metrics.accuracy_score(y_test, rfl_pred))
print("Accuracy Extra:",metrics.accuracy_score(y_test, extra_clf_pred))
print("Accuracy KNN:",metrics.accuracy_score(y_test, KNN_pred))
#print("Accuracy SVC:",metrics.accuracy_score(y_test, svclassifier_pred))



#CNN
#Normalizing the data
x = np.array(x)
x = x/255
y = to_categorical(y,10)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1,stratify=y)

#creo modello cnn
model = Cnn_function.load_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.load_weights("h.5")
model.evaluate(x_test,y_test)
predict = model.predict(x).argmax(axis=1)

#visualizzo la matrice di confusione
utility.matrixcfn(z, predict, y)

#visualizzo predizione
pose_name = np.unique(z)
utility.test('C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\test_set\\plank\\File2.jpg',model,pose_name)
utility.test_classificatore('C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\test_set\\plank\\File2.jpg', gnb, pose_name)
utility.test_classificatore('C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\test_set\\plank\\File2.jpg', rfl, pose_name)


utility.test('C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\BRIDGE-POSE-1.jpg',model,pose_name)
utility.test_classificatore("C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\BRIDGE-POSE-1.jpg",rfl,pose_name)
utility.test_classificatore("C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\BRIDGE-POSE-1.jpg",gnb,pose_name)
utility.test_classificatore("C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\BRIDGE-POSE-1.jpg",classifierKNN,pose_name)
utility.test_classificatore("C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\BRIDGE-POSE-1.jpg",extra_clf,pose_name)

utility.test('C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\ChildsPose.jpg',model,pose_name)
utility.test_classificatore("C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\ChildsPose.jpg",rfl,pose_name)
utility.test_classificatore("C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\ChildsPose.jpg",gnb,pose_name)
utility.test_classificatore("C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\ChildsPose.jpg",classifierKNN,pose_name)
utility.test_classificatore("C:\\Users\\Gianpaolo Patierno\\PycharmProjects\\yoga_classifier\\datasets_file\\predizioni\\ChildsPose.jpg",extra_clf,pose_name)