#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
@author: Filippo Falezza
<filippo dot falezza at outlook dot it>
<fxf802 at student dot bham dot ac dot uk>

Released under GPLv3 and followings
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import math
import array
import os
import pdb
import pickle

#General stuff
def PATH_set(source='./Calibration/1d', target = f'./Calibration/1d/ROI', extension='.tif'):
    return [source, target, extension]

def plot_image(input):
        plt.figure()
        plt.imshow(input)
        plt.show()
        return None

def plot_map(input):
        plt.figure()
        plt.imshow(input)
        plt.colorbar()
        plt.show()
        return None

def toggle_selector(event):
    """
    Useless function, do not remove as this would break the code (ROI selection)
    """
    return None

class Files:
    def __init__(self, source, target, extension):
        self.PATH = PATH_set(source, target, extension)
        self.file_list = []
        for file in os.listdir(self.PATH[0]):
            if file.endswith(self.PATH[2]):
                self.file_list.append(f'{file}')
        #pdb.set_trace()
        if not os.path.exists(self.PATH[1]):
            os.makedirs(self.PATH[1])
        #else: return f'Error, the target directory must be empty'
        self.ROI_list = []
        self.load_ROIs()

    def __image_open__(self, i, flag=0):
        """
        Open picture
        flag: 0 for file, 1 for ROI
        """
        if flag == 0: img = f'{self.PATH[0]}/{self.file_list[i]}'
        elif flag == 1: img = f'{self.PATH[1]}/{self.ROI_list[i]}'
        return cv2.imread(img, -1)#keep img as-is: 16bit tiff

    def load_ROIs(self):
        for file in os.listdir(self.PATH[1]):
            if file.endswith(self.PATH[2]):
                self.ROI_list.append(f'{file}')

    def save(self, namefile):
        """
        Save Calibration to pickle file.
        """
        with open(namefile + '.pkl', 'wb') as f:
            pickle.dump(self.Data, f,  pickle.DEFAULT_PROTOCOL)
            #use default protocol to make it compatible with python 3.4 and followings

    def load(self, namefile):
        """
        Load Calibration from pickle file
        """
        with open(namefile + '.pkl', 'rb') as f:
            self.Data = eval(repr(pickle.load(f)))

class Analysis:
    def OD(self, i, type=int(2), roi=1):
        """
        Calculate Optical density of a ROI

        Parameters
        ----------
        i: int, the index of the image in the file_list list
        type: int, the color channel to use. 2 is red, 1 is green, 0 is blue
        Returns Optical Density
        """
        #TODO: add possible modification of image color depth
        im = self.__image_open__(i, roi)
        if im.dtype != 'uint16':
            return f'Error, image is not 16bit color depth'
        channelImage  = im[:,:,type]
        OD = np.log10(65535.0/channelImage)
        return OD

    def OD_avg(self, OD):
        """
        Averages the OD over the ROI area
        """
        #Could replace with np.avg?
        return np.sum(OD)/(len(OD)*len(OD[0]))

    def Dose(self, OD):
        return self.coeff[0]*(OD**2)+self.coeff[1]*(np.abs(OD))+self.coeff[2]

class Calibrate(Files,Analysis):
    """
    Class to initialise the calibration of the gafchromic films
    """
    def __init__(self, source='./Calibration/1d', target = f'./Calibration/1d/ROI', extension='.tif'):
        Files.__init__(self, source, target, extension)
        Analysis.__init__(self)

    def __write_comments__(self, time, instrument, location, comments):
        """
        Function to write variables into the calibration dictionary
        """
        self.Data = {}
        self.Data['time'] = time
        self.Data['scanning instument'] = instrument
        self.Data['scan location'] = location
        self.Data['comments'] = comments

    def __dose_input__(self, index):
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

    def __image_selector__(self, i, index):
        """
        Function to analyse multiple ROIs within the same image.
        Supports up 999 images. Tested
        """
        selector = input(f'Move to next image? [Y/n]')
        if selector in ['y', 'Y', '']: #The '' is the return key
            if isinstance(i, int):
                i += 1
                index += 1
            elif isinstance(i, float):
                index += 1
                i = eval(repr(index))
        elif selector in ['N','n']:
            if isinstance(i, int):
                index = i
            i = round(i+0.001,3)
        #print(i,index)
        return [i, index]

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
        image = self.__image_open__(i)
        image_8 = (image/256).astype('uint8')
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

        x1,y1,x2,y2 = self.ROI_data
        ROI = image[y1:y2, x1:x2]
        ROI_small = cv2.resize(ROI, (600, 600))#Resize picture to fit into screen
        cv2.imshow(f'ROI {i}',ROI_small)
        cv2.imwrite(f'{self.PATH[1]}/ROI_{i}.tif',ROI, ((int(cv2.IMWRITE_TIFF_COMPRESSION), 1)))
        self.ROI_list.append(f'{self.PATH[1]}/ROI_{i}.tif')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def toggle_selector(self, event):
        if event.key in ['Q', 'q']:# and toggle_selector.RS.active:
            print('RectangleSelector deactivated, exiting...')
            toggle_selector.RS.set_active(False)
        if event.key in ['D', 'd']:# and toggle_selector.RS.active:
            print('RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a']:# and not toggle_selector.RS.active:
            print('RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

    #MatPlotLib stuff, as in https://matplotlib.org/2.0.2/examples/widgets/rectangle_selector.html
    def position(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        #write to this because toggle_selector.RS does not return anything
        self.ROI_data = [int(x1),int(y1),int(x2),int(y2)]
        return None

    def calibrate(self, ROI=0, time='1d', instrument=None, location=None, comments=None, color=int(2)):
        """
        Process all files, extract ROIs and their OD, then save it to a dictionary
        Parameters
        ----------
        ROI: int, enable use of pre-existing ROIs. 0 to select ROIs, 1 to use pre-existing ROIs

        Returns a dictionary with the calibration info
        """
        self.__write_comments__(time, instrument, location, comments)
        i = 0
        index = 0
        while i < len(self.file_list):
            self.Data[i] = []
            if ROI == 0: self.ROI_single(index)
            self.Data[i].append(self.OD_avg(self.OD(index, color)))
            self.Data[i].append(float(self.__dose_input__(index)))
            i,index = self.__image_selector__(i,index)
        return self.Data


class Fitting(Files,Analysis):
    def __init__(self, data, source='Calibration/1d', ROI='Calibration/ROI', path_out='Calibration/Fit'):
        Files.__init__(self, source, ROI, '.tif')
        Analysis.__init__(self)
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
        self.Data = np.ndarray(3)

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

    def fit_plot(self, samples=1000, x_min=None, x_max=None ):
        """
        Plot the fitting
        """
        #c, stats = np.polynomial.polynomial.polyfit(self.Array[0], self.Array[1], 2, full=True)
        c = self.fit()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(self.Array[0], self.Array[1], 'rx', label=f'{self.time} developing time')
        if x_min is None: x_min = np.min(self.Array[0])
        if x_max is None: x_max = np.max(self.Array[0])
        x_fit = np.linspace(x_min, x_max, samples)
        #DEBUG
        #print(c)
        func_fit = np.poly1d(c)
        y_fit = func_fit(x_fit)
        ax.plot(x_fit, y_fit, 'b', label=f'Fitting ax\u00b2+bx+c, a={c[0]:.2f},\nb={c[1]:.2f}, c={c[2]:.2f}')
        ax.set_title('Dose vs OD')
        ax.set_xlabel('Optical Density')
        ax.set_ylabel('Dose [Gy]')
        ax.legend()
        plt.savefig(f'{self.PATH_out}/Fitting_{self.time}.png', dpi=600)
        plt.show()

    def fit(self):
        """
        Fit the data to extract the proper calibration of the EBT3
        NOTE: the calibration is specific to the acquisition instrument used
        """
        self.Data = np.polyfit(self.Array[0], self.Array[1], 2)
        return self.Data

    def dose(self, i, roi=0):
        OD = self.OD(i, roi)
        Dose = self.Data[0]*(OD**2)+self.Data[1]*(np.abs(OD))+self.Data[2]
        return Dose

    def dose_map(self, i, list=0):
        #need to implement possible selection of area to analyse
        """
        if list == 0: dose = self.dose(i, list)
        elif list == 1: dose = self.dose(i, list)
        #"""
        dose = self.dose(i, list)
        fig = plt.subplots()
        plt.imshow(dose)
        plt.colorbar()
        plt.savefig(f'{self.PATH_out}/Dose_plot_{self.file_list[i]}.png', dpi=600)
        plt.show()

    def dose_hist(self, i, list=0):
        dose = self.dose(i, list)
        fig = plt.subplots()
        n = plt.hist(dose.ravel(), bins=1000)
        plt.savefig(f'{self.PATH_out}/Dose_hist_{self.file_list[i]}.png', dpi=600)
        plt.show()

    #the following replace some inherited functions
    def load(self, namefile):
        self.Data = np.load(namefile + '.npy')

    def save(self, namefile):
        np.save(namefile, self.Data)


if __name__ == '__main__':
    """
    #from EBT3 import File, Analysis, Calibration, Fitting
    #c = Calibrate('202105_UoB_Microbeam/Films/','Calibration/ROI')
    c = Calibrate('Calibration','Calibration/ROI')
    #c.calibrate()
    c.load(f'{c.PATH[0]}/Calibration_UoB')
    f = Fitting(c.Data, source='Calibration')
    f.fit()
    f.save(f'{f.PATH_out}/Fit')
    #f.plot()
    #f.fit_plot()
    #f.dose(0)
    f.dose_map(6, 1)
    f.dose_hist(6, 1)
    #"""

    #"""
    #from EBT3 import File, Analysis, Calibration, Fitting
    c = Calibrate('Calibration','Calibration/ROI')
    c.load(f'{c.PATH[0]}/Calibration_UoB')
    f = Fitting(c.Data, source='Calibration')
    f.load(f'{f.PATH_out}/Fit')
    f.fit_plot()
    f.dose_map(6, 1)
    f.dose_hist(6, 1)
    #"""
