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

def OD(img):
    #TODO: add possible modification of image color depth
    #use openCV/cv2
    im = cv2.imread(img, -1)#keep img as-is: 16bit tiff
    #print(im.shape)
    if im.dtype =! 'uint16':
        return f'Error, image is not 16bit color depth'
    redImage  = im[:,:,2]
    redImage = 65535 - redImage
    OD = -np.log10(redImage/65535.0)

    plt.figure()
    plt.imshow(OD)
    plt.show()


if __name__ == '__main__':
    sx = ndimage.sobel(im, axis=0, mode='constant')
    from PIL import Image
    from scipy import misc
    import tifffile as tf
    import matplotlib.pyplot as plt
    import cv2

    img = 'Calibration/02-10-14_Film001.tif'
    """
    #Pillow
    im = Image.open(img) # uses the Image module (PIL)
    #sobel filter

    sy = ndimage.sobel(im, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    data = np.random.randint(0, 255, (256, 256, 3), 'uint8')
    #imwrite('temp.tif', data, photometric='rgb')
    #"""
    """
    image = tf.imread(img)
    print(image.shape)
    #plt.imshow(image)
    #plt.show()
    #"""
    #"""

    #"""
