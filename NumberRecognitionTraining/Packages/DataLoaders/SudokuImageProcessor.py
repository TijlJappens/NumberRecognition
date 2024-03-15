import cv2
import numpy as np
from .ImageLoader import *
import torch
import os.path
import copy

class SudokuImageProcessor(ImageLoader):
    def __init__(self,image_path,number_path=None):
        # Convert the image to grayscale
        image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.gray = self.cropAll(self.gray)
        self.gray=cv2.bitwise_not(self.gray)
        self.number_path = number_path
        self.cropped_images_list = [[None]*9 for _ in range(9)]
        if number_path != None:
            if os.path.exists(number_path):
                self.number_list = np.load(number_path)

        
    def display(self):
        # Display the image
        cv2.imshow('Sudoku: ', self.gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_digits(self, inner_margin):
        img_np = np.array(self.gray)
        del_x = len(img_np[0,:])/9
        del_y = len(img_np[:,0])/9
        # Crop the image to the numbers
        for i in range(9):
            for j in range(9):
                self.cropped_images_list[j][i] = self.gray[int((j+inner_margin)*del_y):int((j+1-inner_margin)*del_y), int((i+inner_margin)*del_x):int((i+1-inner_margin)*del_x)]
        # Downscale the images and crop and fill
        for i in range(9):
            for j in range(9):
                self.cropped_images_list[j][i] = self.cropAndFill(cv2.resize(self.cropped_images_list[j][i], (28, 28)),28,28)

    def predictions(self,model):
        prediction_list = np.zeros((9,9), dtype=int)
        for i in range(9):
            for j in range(9):
                if np.all(np.array(self.cropped_images_list[j][i])==0):
                    prediction_list[j][i] = 10
                else:
                    n = int(torch.argmax(model(torch.from_numpy(np.array(self.cropped_images_list[j][i])).float())).item())
                    prediction_list[j][i] = n
        return prediction_list

    def save_numbers(self, arr):
        np.save(file=self.number_path,arr=arr)

    def get_numbers(self):
        dummy_number_list = copy.copy(self.number_list)
        for i in range(9):
            for j in range(9):
                if dummy_number_list[j][i] == 0:
                    dummy_number_list[j][i] = 10
        return dummy_number_list

    def to_test_data(self):
        test_data = []
        test_labels = []
        for i in range(9):
            for j in range(9):
                if (self.number_list[j][i] == 0) | (self.number_list[j][i] == 10):
                    pass
                else:
                    test_data.append(self.cropped_images_list[j][i])
                    test_labels.append(self.number_list[j][i])
        return (test_data,test_labels)
