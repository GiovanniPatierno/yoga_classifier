from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import cv2 as cv

#visualizzo la confuzion matrix
def matrixcfn(z,predict,y):
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

#visualizzo predizione
def test(imgpath, model, pose_names):     #'immagini per test/golden.jpg', Resnet50_model, dog_names)
    img = cv.imread(imgpath)
    imgClean = img
    imgClean = cv.resize(img, (500, 500))
    img = cv.resize(img, (150, 150))
    img = np.array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    pred = np.argmax(pred,-1)
    cv.putText(img, 'Predizione razza: {}'.format(pose_names[pred]), (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 0), 2)
    cv.imshow("Predizione ", imgClean)
    print("la predizione è': ", pose_names[pred])
    cv.waitKey(0)