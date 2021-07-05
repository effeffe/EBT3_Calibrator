#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import math
import array
import os
import pdb
import pickle

#TODO: check if these are needed
from scipy import ndimage
from PIL import Image
#from libtiff import TIFF
import tifffile as tf


#TODO: check libraries to be called in correct functions, clean stuff

class Calibrate():
#class Calibrate(dict):
#Could initialise whole class as dictionary where self = self.Data
    """
    Class to initialise the calibration of the gafchromic films
    """
    def __init__(self, source='./Calibration/1d', target = f'./Calibration/1d/ROI', extension='.tif'):
        """
        Initialises the class

        Paramters
        ---------
        source: str, the folder in where the images of the films are stored
        target: str, forlder in which to save the ROIs
        extension: str, the extension of the images. Please specify the dot in front of it
        """
        self.PATH_SOURCE = source
        self.PATH_TARGET = target
        self.file_list = []
        self.extension = extension
        for file in os.listdir(self.PATH_SOURCE):
            if file.endswith(self.extension):
                self.file_list.append(file)
        #pdb.set_trace()
        if not os.path.exists(self.PATH_TARGET):
            os.makedirs(self.PATH_TARGET)
        #else: return f'Error, the target directory must be empty'
        self.ROI_list = []

    def load_ROIs(self):
        """
        Load all the ROIs from self.PATH_TARGET folder.
        Please keep this folder clean from any file that is not a ROI.
        In theory, other files that are not ending with the specified extension could be placed here...

        Returns a message (could be omitted)
        """
        for file in os.listdir(self.PATH_TARGET):
            if file.endswith(self.extension):
                self.ROI_list.append(file)
        return f'ROIs loaded'

    def __ROI_automatic(self, i):
        """
        Note: Deprecated as not useful (actually, useful but requires much more work)
        Wrsitten to make ROI selection automatic.
        Currently, it does not work as the ROI still includes some white background

        Currently, this uses boundingRectange (https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html),
        but we could use a Sobel filter (https://docs.opencv.org/4.5.2/d5/d0f/tutorial_py_gradients.html)

        Parameters
        ----------
        i: int, the index of the image in the file_list list
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
        cv2.destroyAllWindowCalibrate/s()

    def ROI_single_small(self, i):
        """
        Manual ROI selection
        Uses OpenCV only. Image needs to be smaller than screen size.
        Issues: zoom in; maybe cannot open large files

        Parameters
        ----------
        i: int, the index of the image in the file_list list
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

    def ROI_single(self, i):
        #TODO: should break it in smaller functions
        """
        Manual ROI selection: rectangular selection (need to add circular region)
        Uses matplotlib, but works on large files.
        The image is showed as an 8bit, but the image is processed as a 16bit one.
        The final ROI is shown as a 16bit, saved in original size but shown on-screen as resized

        Parameters
        ----------
        i: int, the index of the image in the file_list list
        """
        #TODO: need to implement multiple ROI selection within same picture
        #TODO: needs path selection capability and dest folder
        img = f'{self.PATH_SOURCE}/{self.file_list[i]}'
        image = cv2.imread(img, -1)#keep img as-is: 16bit tiff
        image_8 = (image/256).astype('uint8')
        #print(image_8.dtype)#To check that it is a 8bit image
        fig, current_ax = plt.subplots()
        plt.imshow(image_8)
        print(f'Select ROI, then press q to close the Window\n\
Activate and deactivate ROI selection using a and d\n\
The program shows a squared selected ROI, but the final image won\'t be like that')
        #BUG: if use the keyboard selectors, self.Data remains empty and the ROI selection crashes

        # drawtype is 'box' or 'line' or 'none'
        toggle_selector.RS = RectangleSelector(current_ax, self.position,
            drawtype='box', useblit=True, button=[1, 3],  # don't use middle button
            minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        plt.connect('key_press_event', self.toggle_selector)
        plt.show()
        #pdb.set_trace()
        #ROI manual selection
        #x,y,w,h = cv2.selectROI(image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        x1,y1,x2,y2 = self.ROI_data
        #DEBUG
        #print(x1,y1,x2,y2)
        #pdb.set_trace()
        ROI = image[y1:y2, x1:x2]
        ROI_small = cv2.resize(ROI, (600, 600))#Resize picture to fit into screen
        cv2.imshow(f'ROI {i}',ROI_small)
        #cv2.imshow(f'ROI {i}',ROI)
        """
        plt.figure()
        plt.imshow(ROI)
        plt.show()
        #"""
        cv2.imwrite(f'{self.PATH_TARGET}/ROI_{i}.tif',ROI, ((int(cv2.IMWRITE_TIFF_COMPRESSION), 1)))
        self.ROI_list.append(f'{self.PATH_TARGET}/ROI_{i}.tif')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def toggle_selector(self, event):
        #DEBUG:
        #print(f'{event.key} pressed')
        #print(event.key)
        #pdb.set_trace()
        if event.key in ['Q', 'q']:# and toggle_selector.RS.active:
            print('RectangleSelector deactivated, exiting...')
            toggle_selector.RS.set_active(False)
        if event.key in ['D', 'd']:# and toggle_selector.RS.active:
            print('RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a']:# and not toggle_selector.RS.active:
            print('RectangleSelector activated.')
            toggle_selector.RS.set_active(True)
        #if event.key in ['enter', 'Enter']:# toggle_selector.RS.active:
        #    toggle_selector.RS.set_active(False)

#MatPlotLib stuff, as in https://matplotlib.org/2.0.2/examples/widgets/rectangle_selector.html
    def position(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        #write to this because toggle_selector.RS does not return anything
        self.ROI_data = [int(x1),int(y1),int(x2),int(y2)]
        return None

    def ROI_all(self):
        """
        Deprecatded function

        Simply processes all the ROIs with a for loop
        """
        #print(f'This is a manual process, proceed at your own risk\n\
        #    Create a box around ONE beam dot, do not include the white background.\
        #    Press Enter after completing the selection')
        for i in range(len(self.file_list)):
            self.ROI_single(i)
        return f'All ROI extracted'

    def OD(self, i):
        """
        Calculate Optical density of a ROI

        Parameter i: int, the index of the image in the file_list list
        Returns Optical Density
        """
        #TODO: add possible modification of image color depth
        im = cv2.imread(f'{self.ROI_list[i]}', -1)#keep img as-is: 16bit tiff
        if im.dtype != 'uint16':
            return f'Error, image is not 16bit color depth'
        #DEBUG:
        #pdb.set_trace()
        #print(im.shape)
        redImage  = im[:,:,2]
        OD = np.log10(65535.0/redImage)
        return OD

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
        return np.sum(OD)/(len(OD)*len(OD[0]))

    def write_comments(self, time='1d', instrument=None, location=None, comments=None):
        """
        Function to write variables into the calibration dictionary
        """
        self.Data = {}
        self.Data['time'] = time
        self.Data['scanning instument'] = instrument
        self.Data['scan location'] = location
        self.Data['comments'] = comments
        return None

    def dose_input(self, index):
        """
        User input

        TODO: add eventual load of stuff from external file or namefile > into separate function
        """
        while True:
            try:
                dose = input(f'Enter the dose of foil {self.file_list[index]}: ')#user input: Gy
                dose = float(dose)
                break
            except ValueError:
                print(f'Need an float value, try again')
        return dose

    def selector(self, i, index):
        selector = input(f'Move to next image? [Y/n]')
        if selector in ['y', 'Y', '']: #The '' is the return key
            if isinstance(i, int):
                i =+ 1
                index =+ 1
            elif isinstance(i, float):
                index = i+1
                i = eval(repr(index))
        elif selector in ['N','n']:
            if isinstance(i, int):
                index = i
            i = round(i+0.001,3)
        print(i,index)
        return [i, index]

    def calibrate(self):
        """
        Process all files, extract ROIs and their OD, then save it to a dictionary
        """
        #TODO: different acquisition time
        self.write_comments()
        i = 0
        index = 0
        while i < len(self.file_list):
            self.Data[i] = []
            self.ROI_single(index)
            self.Data[i].append(self.OD_avg(self.OD(index)))
            self.Data[i].append(float(self.dose_input(index)))

            #DEBUG
            #pdb.set_trace()
            i,index = self.selector(i,index)
        return self.Data

    def calibrate_noroi(self, time='1d', instrument=None, location=None):
        """
        Like calibrate but with existing ROIs
        """
        self.write_comments()
        i = 0
        index = 0
        #for i in range(len(self.file_list)):
        while i < len(self.ROI_list):
            #pdb.set_trace()
            self.Data[i] = []
            self.Data[i].append(self.OD_avg(self.OD(index)))
            self.Data[i].append(float(self.dose_input(index)))

            i,index = self.selector(i,index)
        return self.Data

    def save(self, namefile):
        """
        Save Calibration to pickle file.
        Save as binary
        """
        with open(namefile + '.pkl', 'wb') as f:
            pickle.dump(self.Data, f,  pickle.DEFAULT_PROTOCOL)
            #use default protocol to make it compatible with python 3.4 and followings
        return f'Saved'

    def load(self, namefile):
        """
        Load Calibration from pickle file
        """
        with open(namefile + '.pkl', 'rb') as f:
            self.Data = eval(repr(pickle.load(f)))


def toggle_selector(event):
    #Should be able to just keep it as return None
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


class Fitting():
    def __init__(self, data, path_out='Calibration'):
        OD = []
        Dose = []
        #DEBUG
        #pdb.set_trace()
        for i in data:
            if isinstance(i, int) or isinstance(i, float):
                OD.append(float(data[i][0]))
                Dose.append(float(data[i][1]))
        #TODO: rearrange array as it is by increasing dose
        self.Array = np.array([OD, Dose])
        self.time = data['time']
        self.PATH_out = path_out

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.Array[0], self.Array[1], 'rx', label=f'{self.time} developing time')
        ax.set_title('Dose vs OD')
        ax.set_xlabel('Optical Density')
        ax.set_ylabel('Dose [Gy]')
        ax.legend()
        plt.savefig(f'{self.PATH_out}/{self.time}.png', dpi=600)
        plt.show()

    def fit(self, x_min=0.01, x_max=1.00, samples=1000):
        """
        Fit the data to extract the proper calibration of the EBT3
        NOTE: the calibration is specific to the acquisition instrument used
        """
        #c, stats = np.polynomial.polynomial.polyfit(self.Array[0], self.Array[1], 2, full=True)
        c = np.polyfit(self.Array[0], self.Array[1], 2)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.Array[0], self.Array[1], 'rx', label=f'{self.time} developing time')
        x_fit = np.linspace(x_min, x_max, samples)
        #DEBUG
        #print(c)
        func_fit = np.poly1d(c)
        y_fit = func_fit(x_fit)
        ax.plot(x_fit, y_fit, 'b', label=f'Fitting, c0={c[0]:.2f},\nc1={c[1]:.2f}, c2={c[2]:.2f}')
        ax.set_title('Dose vs OD')
        ax.set_xlabel('Optical Density')
        ax.set_ylabel('Dose [Gy]')
        ax.legend()
        plt.savefig(f'{self.PATH_out}/Fitting_{self.time}.png', dpi=600)
        plt.show()


if __name__ == '__main__':
    #from Calib import Calibrate, Fitting
    """
    a = Calibrate('./','ROI')
    a.ROI_single(0)
    #"""
    """
    d1 = Calibrate()
    d1.load('Calibration/24h')
    d1_fit = Fitting(d1.Data)
    d1_fit.plot()
    d1_fit.fit()

    d7 = Calibrate()
    d7.load('Calibration/7d')
    d7_fit = Fitting(d7.Data)
    d7_fit.plot()
    d7_fit.fit()
    #"""
    #"""
    d = Calibrate(source='Calibration', target='202105_UoB_Microbeam/ROI')
    d.load('Calibration/Calibration_UoB')
    d_fit = Fitting(d.Data)
    d_fit.plot()
    d_fit.fit()

    #"""
