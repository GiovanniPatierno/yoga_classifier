from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import cv2 as cv
import extract_feature


#visualizzo la confusion matrix
def matrixcfn(z, predict,y):
    matrix = confusion_matrix(y.argmax(axis=1), predict)
    target_labels = np.unique(z)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=[target for target in target_labels],
                yticklabels=[target for target in target_labels],
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix RFC')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

#funzione creata per la visualizzazione delle predizioni riguardanti i classificatori
def test(imgpath, model, pose_names):
    img = cv.imread(imgpath)
    imgClean = img
    imgClean = cv.resize(img, (500, 500))
    img = cv.resize(img, (150, 150))
    img = np.array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    pred = np.argmax(pred,-1)
    cv.putText(imgClean, 'Predizione posizione: {}'.format(pose_names[pred]), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv.putText(imgClean, 'predizione del CNN', (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv.imshow("Predizione ", imgClean)
    print("la predizione è': ", pose_names[pred])
    cv.waitKey(0)

#funzione per la visualizzazioni delle predizioni fatte dalla CNN
def test_classificatore(imgpath, classificatore, pose_names):
    img = cv.imread(imgpath)
    img = cv.resize(img, (500, 500))
    glob_fe = extract_feature.extract(img)

    pred =  classificatore.predict(glob_fe.reshape(1, -1))[0]
    cv.putText(img, 'Predizione posizione: {}'.format(pose_names[pred]), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 0), 2)
    cv.putText(img, 'Classificatore: {}'.format(classificatore), (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 0), 2)
    cv.imshow("Predizione ", img)
    print("la predizione è': ", pose_names[pred])
    cv.waitKey(0)


