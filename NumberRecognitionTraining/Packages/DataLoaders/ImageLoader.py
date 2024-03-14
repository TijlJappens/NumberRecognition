import cv2
import numpy as np

class ImageLoader(object):
    def crop(self, image):
        np_image = np.array(image)
        #find left most margin
        leftmargin = 0
        for i in range(len(np_image[0,:])):
            if np.any(np_image[:, i] != 0):  # Check if any pixel in the column is non-zero
                leftmargin = i
                break


        rightmargin = len(np_image[0,:])-1
        for i in range(len(np_image[0,:])):
            if np.any(np_image[:, -i] != 0):  # Check if any pixel in the column is non-zero
                rightmargin = len(np_image[0,:])-1-i
                break

        uppermargin = 0
        for i in range(len(np_image[:,0])):
            if np.any(np_image[i,:] != 0):  # Check if any pixel in the column is non-zero
                uppermargin = i
                break

        lowermargin = len(np_image[:,0])-1
        for i in range(len(np_image[:,0])):
            if np.any(np_image[-i,:] != 0):  # Check if any pixel in the column is non-zero
                lowermargin = len(np_image[:,0])-1-i
                break
        dummyImage = image[uppermargin:lowermargin,leftmargin:rightmargin]
        return dummyImage
    # This version of crop will work both if the outside layer is all black and when the outside layer is all white
    def cropAll(self,image):
        np_image = np.array(image)
        dummyImage = None
        if np.all(np_image[:,0]==0) | np.all(np_image[:,-1]) | np.all(np_image[0,:]==0) | np.all(np_image[-1,:]==0):
            return self.crop(image)
        else:
            dummyImage = cv2.bitwise_not(image)
            dummyImage = self.crop(dummyImage)
            dummyImage = cv2.bitwise_not(dummyImage)
            return dummyImage