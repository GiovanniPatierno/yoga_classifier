import cv2
import mahotas
import numpy as np
from matplotlib.pyplot import hist
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix


def extract(img):
    f_moments = fd_hu_moments(img)
    f_haralick = fd_haralick(img)
    f_hog = fd_hog(img, 16)

    glob_features_b = np.hstack([f_hog, f_moments, f_haralick])
    return glob_features_b

# HuMoments è un metodo della libreria open-cv che permette di estrarre la forma degli oggetti presenti nell'immagine
def fd_hu_moments(image_):
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image_)).flatten()
    return feature

# Haralick Texture viene utilizzato per quantificare un'immagine in base alla texture
def fd_haralick(image_):
    # converto l'immagine in una grayscale
    gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
    # estraggo l'haralick features
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

def fd_hog(image_, bins):
    dx = cv2.Sobel(image_, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(image_, cv2.CV_32F, 0, 1)

    # Calcolo la magnitude e l'angolo
    magnitude, angle = cv2.cartToPolar(dx, dy)

    # Quantifico i binvalues in (0..n_bins)
    binvalues = np.int32(bins * angle / (2 * np.pi))

    # Divido l'immagine in 4 parti
    magn_cells = magnitude[:10, :10], magnitude[10:, :10], magnitude[:10, 10:], magnitude[10:, 10:]
    bin_cells = binvalues[:10, :10], binvalues[10:, :10], binvalues[:10, 10:], binvalues[10:, 10:]

    # Con "bincount" possiamo contare il numero di occorrenze di un
    # flat array per creare l'istogramma.
    histogram = [np.bincount(bin_cell.ravel(), magn.ravel(), bins)
                 for bin_cell, magn in zip(bin_cells, magn_cells)]

    return np.hstack(histogram)



