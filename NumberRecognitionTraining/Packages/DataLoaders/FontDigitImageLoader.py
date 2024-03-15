import cv2
import numpy as np
import os
from .ImageLoader import *

class FontDigitImageLoader(ImageLoader):
    labels = []
    images = []
    def rand_scaling():
        pass
    # n indicates how many times it should add the value to the dataset to compete with the fact that the minsc dataset is so much larger
    def __init__(self, path, n=1, width=28):
        self.width=width
        for i in range(10):
            folder_path = os.path.join(os.path.join(path,str(i)))
            for filename in os.listdir(folder_path):
                image = cv2.imread(os.path.join(folder_path,filename))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image,(width,width))
                image = cv2.bitwise_not(image)
                image = self.cropAndFill(image, self.width, self.width)
                if n == 1:
                    self.labels.append(i)
                    self.images.append(np.array(image))
                else:
                    pass
                
    def GetImages(self):
        return self.images
    def GetLabels(self):
        return self.labels

