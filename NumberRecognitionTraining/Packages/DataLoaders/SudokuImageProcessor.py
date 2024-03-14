import cv2
import numpy as np
from .ImageLoader import *

class SudokuImageProcessor(ImageLoader):
    
    def __init__(self,path):
        # Convert the image to grayscale
        image = cv2.imread(path)
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray = self.cropAll(self.gray)
        self.img_np = np.array(self.gray)
        self.gray=cv2.bitwise_not(self.gray)

        # Display the image
        # cv2.imshow('Sudoku: ', self.gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # self.img_np = np.array(self.gray)
        # self.crop()
    def digits(self, inner_margin):
        cropped_images_list = [[None]*9 for _ in range(9)]
        del_x = len(self.img_np[0,:])/9
        del_y = len(self.img_np[:,0])/9
        # Crop the image to the numbers
        for i in range(9):
            for j in range(9):
                cropped_images_list[j][i] = self.gray[int((j+inner_margin)*del_y):int((j+1-inner_margin)*del_y), int((i+inner_margin)*del_x):int((i+1-inner_margin)*del_x)]
        # Downscale the images
        for i in range(9):
            for j in range(9):
                cropped_images_list[j][i] = cv2.resize(cropped_images_list[j][i], (28, 28))
        # Return the list
        return cropped_images_list