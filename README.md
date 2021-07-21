# README
## 0. Table of Contents

0. [Table of Contents](#0-table-of-contents)
1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Usage](#3-usage)
4. [Example Code](#4-example-code)

## 1. Introduction
This software was written as a part of a summer project organised by Dr Tony Price at thee University of Birmingham. The software is written by Filippo Faleza (© 2021) and released under GNU General Public License v3 and followings.

## 2. Installation
The program is composed of three main parts: 
- EBT3.py, the core library
- CLI (todo)
- GUI (todo)

### 2.1 Requirements
Python 3.4 or later (tested using python 3.9.5)
Matplotlib
NumPy
OpenCV (cv2)
These can simply be run using python 3.4 and later

## 3. Usage
The software is composed of two main classes: Calibrate and Fitting. These two inherit the Analysis and Files classes to respectively import image analsysis and files management tools.

To use the code, all these classes should be called, but only Calibrate and Fitting used directly. If you do otherwise and anything breaks, you had been warned.

### 3.1 Calibrate
Calibrate allows to select ROIs from images (`ROI_single`) and generate a dataset to be used by the Fitting class (`calibrate`).

During a standard calibration, each specified image in the list is analysed, and for each of these multiple ROIs can be selected (maximum 1000). From these, the Optical Density is then extracted and averages, to then be saved together with the corresponding dose.
This procedure then creates a dictionary, which can be saved to be used at a later time: the dictionary stores informations such as time, location of image acquisition, acquisition instrument and eventual comments.

An example code to do this looks like this:
```
from EBT3 import Analysis,Files,Calibrate,Fitting
c = Calibrate()
#generate list of images to analyse: [2,3,7,8,9,10,11,12,13,14,15,16]
range_ = [2,3,*range(7,17)]

#run the calibration
c.calibrate(ROI=0, time='48h', range=range_, instrument='Epson Expression 10000XL', location='NPL', comments='Irradiated at the MC40 cyclotron at the University of Birmingham; Scanned by Sam Flynn at NPL', colour=2)
#save the calibration to Calibration_saved.pkl for future use
c.save('Calibration_saved')
```

### 3.2 Fitting
Fitting takes the data from Calibrate, fits them and then uses the resulting fitting to analyse the given images. The possible analysis that have been implemented are:
- Fit plot (`fit_plot`),
- Dose Histogram (`dose_hist`),
- Dose Map (`dose_map`),
- Dose Profile (`dose_profile`).
Again, the fitting can be saved to be reused later on, whereas all the pictures are saved to `Calibration/Fit` folder.

### 3.3 Folders
By default, all the files are stored within the Calibration folder in the same PATH of the EBT3.py, unless otherwise specified.
The default PATHs are:
- `Calibration/1d` the source files to analyse,
- `Calibration/1d/ROI` the extracted ROIs,
- `Calibration/1d/Fit` the plots saved by the `Fitting` class.
Considering that not everyone could like these folder structure, that the file could be abducted to be run from elsewhere or whatever the case could be, this can be overwritten.

The `Files` class and its children can overwrite these by defining the followings while initialising: `source` refers to, of course, the source folder with all the acquisitions to analyse, `target` refers to the folders in which to save the ROIs.

In addition to these, the extension of the images can be changed. By default this is `.tif`, so that only `.tif` files will be considered as images (hence `.tiff` or otherwise will simply be ignored).

The `Fitting` class has an additional folder that is self-managed: `path_out`, which is where the fittings/profiles/maps/histograms are saved.

__Please__ be consistent among differen classes: extensions and PATHs should be defined coherently for both the two main classes, otherwise your precious files won't be found by the program.

### 3.4 Save and Load
The `Calibration` and `Fitting` classes allow to save respectively the data list and the Fitting values. The former is stored in `Calibration.Data` and the latter in `Fitting.Data`. Both of these are saved by the `.save()` function and loaded by `.load()` and accept just the filename as an argument. The filename has to be in string format and hence can also include the destination PATH.

Note that `Calibration.Data` is a dictionary, saved by the Pickle library; on the other hand `Fitting.Data` is a Numpy array, hence loaded and saved by Numpy itself.

### 4.5 File Management
Both images and ROIs are saved into lists after the initialisation of each of the two main classes. By keeping the PATHs coherent, these two should be equal one another, unless additional ROIs are created using Calibrate.ROI_single().
The images are called using their index in each list:
- `class.file_list` is the list of images in the source folder
- `class.ROI_list` is the list of ROIs, updated after each is created. In case the two lists should go out of sync, please update is using `class.load_ROIs()`.

All the functions that take the image index as an argument also accept the `roi` argument (usually in second position):
```function(i, roi)```
where `i` is the index in the list, and `roi` specifies whether to use `file_list` (`roi=0`) or the `ROI_list` (`roi=1`).

## 4. Example Code
### 4.1 Initial Data collection
```
from EBT3 import Analysis,Files,Calibrate,Fitting
c = Calibrate('Calibration','Calibration/ROI')
#generate list of images to analyse: [2,3,7,8,9,10,11,12,13,14,15,16]
range_ = [2,3,*range(7,17)]

#run the calibration
c.calibrate(ROI=0, time='48h', range=range_, instrument='Epson Expression 10000XL', location='NPL', comments='Irradiated at the MC40 cyclotron at the University of Birmingham; Scanned by Sam Flynn at NPL', colour=2)
#save the calibration to Calibration_saved.pkl for future use
c.save(f'{c.PATH[0]}/Calibration_saved')
```

### 4.2 Dataset recall
```
from EBT3 import File, Analysis, Calibration, Fitting
c = Calibrate('Calibration','Calibration/ROI')
c.load(f'{c.PATH[0]}/Calibration_saved')
```
### 4.3 Fitting Initialisation
reuse code from [Dataset recall](#42-dataset-recall), then add the following:
```
f = Fitting(c.Data, source='Calibration')
#the following both fits and plot the fitting
f.fit_plot()
f.save(f'{f.PATH_out}/Fit_save')
```

### 4.4 Fitting recall
reuse code from [Dataset recall](#42-dataset-recall), then add the following:
```
f = Fitting(c.Data, source='Calibration')
f.load(f'{f.PATH_out}/Fit_save')
```
#### 4.5 Fitting Analysis
reuse code from [Fitting recall](#44-fitting-recall), then add the following:
```
f.dose_map(4)
f.dose_hist(4)
f.dose_profile(4,roi=0,axis_='x')
```
