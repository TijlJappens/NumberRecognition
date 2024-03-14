import cv2
import numpy as np
import os
from .ImageLoader import *

class FontDigitImageLoader(ImageLoader):
    labels = []
    images = []

    def __init__(self, path):
        for i in range(10):
            folder_path = os.path.join(os.path.join(path,str(i)))
            for filename in os.listdir(folder_path):
                image = cv2.imread(os.path.join(folder_path,filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image,(28,28))
                image = cv2.bitwise_not(image)
                self.labels.append(i)
                self.images.append(np.array(image))
    def GetImages(self):
        return self.images
    def GetLabels(self):
        return self.labels

