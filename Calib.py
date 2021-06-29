#!/usr/bin/python3

import numpy as np

#from libtiff import TIFF
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import tifffile as tf
import cv2
import math
import array
import os
import pdb

#TODO: check libraries to be called in correct functions, clean stuff
#TODO:

class Crop:
    def __init__(self, folder='./Calibration', extension='.tif'):
        self.PATH = folder
        self.file_list = []
        for file in os.listdir(self.PATH):
            if file.endswith(extension):
                self.file_list.append(file)
        #pdb.set_trace()

    def process(self):
        return None

    def ROI(self, i):
        """
        Written to make ROI selection automatic.
        Currently, it does not work as the ROI still includes some white background

        Currently, this uses boundingRectange (https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html),
        but we could use a Sobel filter (https://docs.opencv.org/4.5.2/d5/d0f/tutorial_py_gradients.html)
        """
        img = f'{self.PATH}/{self.file_list[i]}'
        image_16 = cv2.imread(img, -1)#keep img as-is: 16bit tiff
        gray = cv2.cvtColor((image_16/256).astype('uint8'), cv2.COLOR_BGR2GRAY)#Work on 8bit gray image
        gray = cv2.bitwise_not(gray)#invert colors to get boundingBox

        #DEBUG: view gray image
        #cv2.imshow('gray', gray)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #NOTE: Borrowed from https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # Find bounding box and extract ROI
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ROI = image_16[y:y+h, x:x+w]
            break
        cv2.imshow('ROI',ROI)
        cv2.imwrite(f'{self.PATH}/ROI_{i+1}.tif',ROI)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def manual_ROI(self, i):
            img = f'{self.PATH}/{self.file_list[i]}'
            image = cv2.imread(img, -1)#keep img as-is: 16bit tiff
            #ROI manual selection
            x,y,w,h = cv2.selectROI(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ROI = image[y:y+h, x:x+w]
            cv2.imshow(f'ROI {i+1}',ROI)
            cv2.imwrite(f'{self.PATH}/ROI_auto_{i+1}.tif',ROI)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def manual_process(self):
        print(f'This is a manual process, proceed at your own risk\n\
            Create a box around ONE beam dot, do not include the white background.\
            Press Enter after completing the selection')
        for i in range(len(self.file_list)):
            self.manual_ROI(i)
        return f'All ROI extracted'

def OD(img):
    #TODO: add possible modification of image color depth
    #use openCV/cv2
    im = cv2.imread(img, -1)#keep img as-is: 16bit tiff
    print(im.shape)
    if im.dtype != 'uint16':
        return f'Error, image is not 16bit color depth'
    redImage  = im[:,:,2]
    redImage = 65535 - redImage
    OD = -np.log10(redImage/65535.0)
    plt.figure()
    plt.imshow(OD)
    plt.show()


if __name__ == '__main__':
    a = Crop()
    a.manual_process()
