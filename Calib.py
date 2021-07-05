#!/usr/bin/python3

import numpy as np
#from libtiff import TIFF
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy import ndimage
from PIL import Image
import tifffile as tf
import cv2
import math
import array
import os
import pdb
import pickle


#TODO: check libraries to be called in correct functions, clean stuff
#TODO:
class Calibrate():
#class Calibrate(dict):
#Could initialise whole class as dictionary where self = self.Data
    def __init__(self, source='./Calibration/24hours', target = f'./Calibration/24hours/ROI', extension='.tif'):
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
        for file in os.listdir(self.PATH_TARGET):
            if file.endswith(self.extension):
                self.ROI_list.append(file)
        return f'ROIs loaded'

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
        cv2.destroyAllWindowCalibrate/s()

    def ROI_single_small(self, i):
        """
        Manual ROI selection
        Uses OpenCV only. Image needs to be smaller than screen size. Also, it has issues with zoom in
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
        """
        Manual ROI selection
        Works on large files, but requries matplotlib
        """
        #TODO: need to implement multiple ROI selection within same picture
        #TODO: needs path selection capability and dest folder
        img = f'{self.PATH_SOURCE}/{self.file_list[i]}'
        image = cv2.imread(img, -1)#keep img as-is: 16bit tiff
        image_8 = (image/256).astype('uint8')
        #print(image_8.dtype)#To check that it is a 8bit image
        fig, current_ax = plt.subplots()
        plt.imshow(image_8)
        #MatPlotLib stuff, as in https://matplotlib.org/2.0.2/examples/widgets/rectangle_selector.html
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
        print(x1,y1,x2,y2)
        #pdb.set_trace()
        ROI = image[y1:y2, x1:x2]
        ROI_small = cv2.resize(ROI, (600, 600))#Resize picture to fit into screen
        cv2.imshow(f'ROI {i}',ROI_small)#Should use matplotlib for this?
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

    def position(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.ROI_data = [int(x1),int(y1),int(x2),int(y2)]
        #return self.ROI_data

    def ROI_all(self):
        #Deprecatded
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
        im = cv2.imread(f'{self.PATH_TARGET}/{self.ROI_list[i]}', -1)#keep img as-is: 16bit tiff
        #print(im.shape)
        if im.dtype != 'uint16':
            return f'Error, image is not 16bit color depth'
        redImage  = im[:,:,2]
        #redImage = 65535 - redImage
        OD = np.log10(65535.0/redImage)
        #DEBUG
        #pdb.set_trace()
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

    def calibrate(self, time='1d', instrument=None, location=None):
        """
        Process the OD of a ROI and save it to a dictionary
        """
        #TODO: different acquisition time
        #TODO:
        self.Data = {}
        self.Data['time'] = time
        self.Data['scanning instument'] = instrument
        self.Data['scan location'] = location
        i = 0
        index = 0
        #for i in range(len(self.file_list)):
        while i < len(self.file_list):
            self.Data[i] = []
            self.ROI_single(index)
            self.Data[i].append(self.OD_avg(self.OD(index)))
            #user input: Gy
            #dose = input(f'Enter the dose of foil {self.file_list[index]}: ')
            #CORRECT THIS
            #"""
            while True:
                try:
                    dose = input(f'Enter the dose of foil {self.file_list[index]}: ')
                    dose = float(dose)
                    break
                except ValueError:
                    print(f'Need an float value, try again')
            #"""
            self.Data[i].append(float(dose))
            #DEBUG
            #pdb.set_trace()
            """
            selector = input(f'Move to next image? [Y/n]')
            if selector in ['y', 'Y', 'Return', 'return', 'enter', 'Enter']:
                #FIX: return/enter key does not work and goes to N/n
                #Actually, none of the keys that are not the first one in list work
                #pdb.set_trace()
                if isinstance(i, int): i = i+1
                elif isinstance(i, float):
                    index = i+1
                    i = eval(repr(index))
                    print(i,index)
            elif selector in ['N','n']:
                if isinstance(i, int): index = i
                i = round(i+0.01,2)
                print(i,index)
            #"""
            i = i+1
            index = index+1
        return self.Data

    def calibrate_noroi(self, time='1d', instrument=None, location=None):
        """
        function to calibrate using existing ROIs
        """
        self.Data = {}
        self.Data['time'] = time
        self.Data['scanning instument'] = instrument
        self.Data['scan location'] = location
        i = 0
        index = 0
        #for i in range(len(self.file_list)):
        while i < len(self.ROI_list):
            #pdb.set_trace()
            self.Data[i] = []
            self.Data[i].append(self.OD_avg(self.OD(index)))
            #user input: Gy
            #dose = input(f'Enter the dose of foil {self.ROI_list[index]}: ')
            #CORRECT THIS
            #"""
            while True:
                try:
                    dose = input(f'Enter the dose of foil {self.file_list[index]}: ')
                    dose = float(dose)
                    break
                except ValueError:
                    print(f'Need an float value, try again')
            #"""
            self.Data[i].append(float(dose))
            i = i+1
            index = index+1
        return self.Data


    def save(self, namefile):
        """
        Save Calibration to pickle file
        """
        with open(namefile + '.pkl', 'wb') as f:
            pickle.dump(self.Data, f, pickle.HIGHEST_PROTOCOL)
        return f'Saved'

    def load(self, namefile):
        with open(namefile + '.pkl', 'rb') as f:
            self.Data = eval(repr(pickle.load(f)))


def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)

#TODO: fitting of Calibration
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
        ax.plot(self.Array[0], self.Array[1], 'bo', label=f'{self.time} developing time')
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
        ax.plot(self.Array[0], self.Array[1], 'bo', label=f'{self.time} developing time')
        x_fit = np.linspace(x_min, x_max, samples)
        #DEBUG
        #print(c)
        func_fit = np.poly1d(c)
        y_fit = func_fit(x_fit)
        ax.plot(x_fit, y_fit, label=f'Fitting, c0={c[0]:.2f},\nc1={c[1]:.2f}, c2={c[2]:.2f}')
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
    d.calibrate()
    d.save('UoB_Data')
    d_fit = Fitting(d1.Data)
    d_fit.plot()
    d_fit.fit()

    #"""
