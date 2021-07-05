# README
## Table of Contents

0. Table of Contents
1. Introduction
2. Installation
3. Usage

## Introduction
This software was written as a part of a summer project organised by Dr TOny Price at thee University of Birmingham. The software is written by Filippo Faleza (© 2021) and released under GNU General Public License v3 and followings.

## Installation
The program is composed of three main parts: 
- EBT3.py, the core library
- CLI (todo)
- GUI (todo)

### Requirements
Python 3.4 or later (tested using python 3.9.5)
Matplotlib
NumPy
OpenCV (cv2)
These can simply be run using python 3.4 and later

## Usage
The software is composed of two main classes: Calibrate and Fitting. 

### Calibrate
Calibrate allows to open images, select ROIs (Regions of Interest) and analyse these. Multiple ROIs can be selected per picture (maximum 1000). From these, the Optical Density is then extracted, averaged and then plotted against the dose of the area.

This procedure then creates a dictionary, which can be saved to be used at a later time: the dictionary stores informations such as time, location of image acquisition, acquisition instrument and eventual comments.

#### 
