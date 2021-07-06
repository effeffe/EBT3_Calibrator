# README
## 0. Table of Contents

1. [Table of Contents](##-0.-Table-of-Contents)
2. [Introduction](##-2.-Introduction)
3. [Installation](##-3.-Installation)
4. [Usage](##-4.-Usage)

## 2. Introduction
This software was written as a part of a summer project organised by Dr Tony Price at thee University of Birmingham. The software is written by Filippo Faleza (© 2021) and released under GNU General Public License v3 and followings.

## 3. Installation
The program is composed of three main parts: 
- EBT3.py, the core library
- CLI (todo)
- GUI (todo)

### 3.1 Requirements
Python 3.4 or later (tested using python 3.9.5)
Matplotlib
NumPy
OpenCV (cv2)
These can simply be run using python 3.4 and later

## 4. Usage
The software is composed of two main classes: Calibrate and Fitting. 

### 4.1 Calibrate
Calibrate allows to open images, select ROIs (Regions of Interest) and analyse these. Multiple ROIs can be selected per picture (maximum 1000). From these, the Optical Density is then extracted, averaged and then plotted against the dose of the area.

This procedure then creates a dictionary, which can be saved to be used at a later time: the dictionary stores informations such as time, location of image acquisition, acquisition instrument and eventual comments.

#### 
