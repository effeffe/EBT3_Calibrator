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

class Calibrate:
    def __init__(self, source='./Calibration', target = f'./Calibration/ROI', extension='.tif'):
        self.PATH_SOURCE = source
        self.PATH_TARGET = target
        self.file_list = []
        for file in os.listdir(self.PATH_SOURCE):
            if file.endswith(extension):
                self.file_list.append(file)
        #pdb.set_trace()
        if not os.path.exists(self.PATH_TARGET):
            os.makedirs(self.PATH_TARGET)
        #else: return f'Error, the target directory must be empty'
        self.ROI_list = []

    def __ROI_automatic(self, i):
        """
        Note: Deprecated as not useful
        Wrsitten to make ROI selection automatic.
        Currently, it does not work as the ROI still includes some white background

        Currently, this uses boundingRectange (https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html),
        but we could use a Sobel filter (https://docs.opencv.org/4.5.2/d5/d0f/tutorial_py_gradients.html)
        """
        img = f'{self.PATH_SOURCE}/{self.file_list[i]}'
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
        cv2.imwrite(f'{self.PATH_TARGET}/ROI_auto_{i+1}.tif',ROI)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ROI_single(self, i):
        """
        Manual ROI selection
        """
        #TODO: need to implement multiple ROI selection within same picture
        #TODO: needs path selection capability and dest folder
        img = f'{self.PATH_SOURCE}/{self.file_list[i]}'
        image = cv2.imread(img, -1)#keep img as-is: 16bit tiff
        #ROI manual selection
        x,y,w,h = cv2.selectROI(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ROI = image[y:y+h, x:x+w]
        cv2.imshow(f'ROI {i}',ROI)
        cv2.imwrite(f'{self.PATH_TARGET}/ROI_{i}.tif',ROI, ((int(cv2.IMWRITE_TIFF_COMPRESSION), 1)))
        self.ROI_list.append(f'{self.PATH_TARGET}/ROI_{i}.tif')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ROI_all(self):
        """
        print(f'This is a manual process, proceed at your own risk\n\
            Create a box around ONE beam dot, do not include the white background.\
            Press Enter after completing the selection')
        #"""
        for i in range(len(self.file_list)):
            self.ROI_single(i)
        return f'All ROI extracted'

    def OD(self, i):
        """
        Return Optical Density of a ROI
        """
        #TODO: add possible modification of image color depth
        #use openCV/cv2
        #DEBUG:
        #pdb.set_trace()
        im = cv2.imread(self.ROI_list[i], -1)#keep img as-is: 16bit tiff
        print(im.shape)
        if im.dtype != 'uint16':
            return f'Error, image is not 16bit color depth'
        redImage  = im[:,:,2]
        redImage = 65535 - redImage
        OD = -np.log10(redImage/65535.0)
        pdb.set_trace()
        return OD
        #Maybe save it instead of show to make the process faster

    def OD_plot(self, OD):
        """
        Just plot the OD
        """
        plt.figure()
        plt.imshow(OD)
        plt.show()
        return None

    def OD_avg(self, OD):
        """
        Averages the OD over the ROI area
        """
        avg = np.sum(OD)/(len(OD)*len(OD[0]))
        return avg

    def calibrate(self, time='1d'):
        """
        Process the OD of a ROI and save it to a dictionary
        """
        #TODO: different acquisition time
        #TODO:
        self.Data = {}
        self.Data['time'] = time
        for i in range(len(self.file_list)):
            self.Data[i] = []
            self.ROI_single(i)
            self.OD(i)
            self.Data[i].append(OD_i)
            #user input: Gy
            dose = input(f'Enter the dose of foil {i}')
            self.Data[i].append(dose)
            pdb.set_trace()
        return self.Data



if __name__ == '__main__':
    a = Crop()
    a.manual_process()
