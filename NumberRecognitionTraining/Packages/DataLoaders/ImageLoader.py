import cv2
import numpy as np

class ImageLoader(object):
    def shrinkable(self,np_image):
        return np.all(np_image[:,0]==0) | np.all(np_image[:,-1]) | np.all(np_image[0,:]==0) | np.all(np_image[-1,:]==0)
    def crop(self, image):
        np_image = np.array(image)
        if np.all(np_image==0):
            return np.zeros((28,28))
        #find left most margin
        leftmargin = 0
        for i in range(len(np_image[0,:])):
            if np.any(np_image[:, i] != 0):  # Check if any pixel in the column is non-zero
                leftmargin = i
                break

        rightmargin = len(np_image[0,:])
        for i in range(len(np_image[0,:])):
            if np.any(np_image[:, -i] != 0):  # Check if any pixel in the column is non-zero
                rightmargin = len(np_image[0,:])-i
                break

        uppermargin = 0
        for i in range(len(np_image[:,0])):
            if np.any(np_image[i,:] != 0):  # Check if any pixel in the column is non-zero
                uppermargin = i
                break

        lowermargin = len(np_image[:,0])
        for i in range(len(np_image[:,0])):
            if np.any(np_image[-i,:] != 0):  # Check if any pixel in the column is non-zero
                lowermargin = len(np_image[:,0])-i
                break
        dummyImage = image[uppermargin:lowermargin+1,leftmargin:rightmargin+1]
        return dummyImage
    # This version of crop will work both if the outside layer is all black and when the outside layer is all white
    def cropAll(self,image):
        np_image = np.array(image)
        dummyImage = None
        if self.shrinkable(np_image):
            return self.crop(image)
        else:
            dummyImage = cv2.bitwise_not(image)
            dummyImage = self.crop(dummyImage)
            dummyImage = cv2.bitwise_not(dummyImage)
            return dummyImage
    # First crops the image and then fills with whitespace untill desired resolution is reached
    def addZeros(self, np_image, x_left, x_right, y_left, y_right):
        x_image = len(np_image[0,:])
        y_image = len(np_image[:,0])
        final = np.zeros((y_image+y_left+y_right,x_image+x_left+x_right))
        final[y_left:y_left+y_image,x_left:x_left+x_image]=np_image
        return final

    def cropAndFill(self, image, x, y):
        shrinkable = self.shrinkable(np.array(image))
        cropped_image = self.cropAll(image)
        np_cropped_image = np.array(cropped_image)
        if np_cropped_image.size==0:
            return np.zeros((28,28), dtype=int)
        x_crop = len(np_cropped_image[0,:])
        y_crop = len(np_cropped_image[:,0])
        x_left = int((x-x_crop)/2)
        x_right = x-x_left-x_crop
        y_left = int((y-y_crop)/2)
        y_right = y-y_left-y_crop
        if shrinkable:
            return self.addZeros(np_cropped_image,x_left,x_right,y_left,y_right)
        else:
            cropped_image = cv2.bitwise_not(cropped_image)
            self.addZeros(np_cropped_image,x_left,x_right,y_left,y_right)
            cropped_image = cv2.bitwise_not(cropped_image)
            return cropped_image
        