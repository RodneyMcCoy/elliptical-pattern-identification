# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:39:31 2021

@author: Malachi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
image = io.imread('greyscale_2DFFT.jfif')
print(np.shape(image))
print(image)
plt.imshow(image)