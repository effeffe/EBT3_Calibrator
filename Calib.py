#!/usr/bin/python3

import numpy as np

#from libtiff import TIFF
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import math
import array
import os
import pdb

#TODO: check libraries to be called in correct functions, clean stuff
#TODO:

class Crop:
    def __init__(self, folder='./Calibration', extension='.tif'):
        from PIL import Image
        self.PATH = folder
        self.files_list = []
        for file in os.listdir(self.PATH):
            if file.endswith(extension):
                self.files_list.append(file)
        pdb.set_trace()

    def image_process(self):
        return None

if __name__ == '__main__':
    from PIL import Image
    from scipy import misc
    import tifffile as tf
    import matplotlib.pyplot as plt

    im = Image.open('Calibration/02-10-14_Film001.tif') # uses the Image module (PIL)
    sx = ndimage.sobel(im, axis=0, mode='constant')
    sy = ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    data = np.random.randint(0, 255, (256, 256, 3), 'uint8')
    #imwrite('temp.tif', data, photometric='rgb')

    tf.imread('temp.tif', data, photometric='rgb')
    plt.imshow(im)
    plt.show()
    plt.imshow(data)
    plt.show()

"""
    #Variables
    color_depth = 16 #bits
    image_names = "/home/filippo/Scrivania/Uni/MedicalPhysics_Summer2021/Given/02-10-14/7days/02-10-14_Film007_cropped.tif" #just get calibration filmes#

    f = image_names
    print(f)

    # read in the image
    im = cv2.imread(f, -1)

    print(im.shape)

    # for GaF scan only want the red part
    # take the whole image and split into colors
    redImage  = im[:,:,2]
    greenImage = im[:,:,1]
    blueImage = im[:,:,0]

    # x values are flipped so reverse them
    #redImage = np.flip(redImage,1)

    print(redImage.dtype)

    #inverts the colormap from a tiff for GaFChromic
    color_levels = 2**int(color_depth) - 1
    redImage = color_levels - redImage

    #convert original image to optical density
    OD = -np.log10(redImage/ float(color_levels))

    plt.figure()
    plt.imshow(OD)
    plt.show()
"""
