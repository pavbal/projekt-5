import numpy as np
import matplotlib.pyplot as plt
import skimage
from cffi.setuptools_ext import execfile
from skimage import data
from skimage.color import rgba2rgb, rgb2gray
import skimage.segmentation
import scipy
#from scipy.misc import imread
import matplotlib.image as img
import skimage.io
import os
from os import listdir
from urllib.request import urlopen

import runpy



# exec(open("diversification.py").read())
# exec(open("percentage.py").read())

# runpy.run_path(path_name='diversity.py')
# runpy.run_path(path_name='percentage.py')
# runpy.run_path(path_name='occurrence.py')
# runpy.run_path(path_name='occurrence_strong.py')

# runpy.run_path(path_name='diversity_subplot.py')
# runpy.run_path(path_name='occurence_subplot.py')
# runpy.run_path(path_name='percentage2_subplot.py')




