	#!/usr/bin/env python

#from libtiff import TIFF
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


#############################################################################

import array
import os

image_names = "../films/02-10-14/7days/02-10-14_Film007_cropped.tif" #just get calibration filmes#

f = image_names
print( f )

# read in the image
im = cv2.imread(f, -1)

print im.shape

# for GaF scan only want the red part
# take the whole image and split into colors
redImage  = im[:,:,2]
greenImage = im[:,:,1]
blueImage = im[:,:,0]

# x values are flipped so reverse them
#redImage = np.flip(redImage,1)

print redImage.dtype

#inverts the colormap from a tiff for GaFChromic
redImage = 65535 - redImage

#convert original image to optical density
OD = -np.log10(redImage/65535.0)

plt.figure()
plt.imshow(OD)
plt.show()


