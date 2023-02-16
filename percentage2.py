import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data
from skimage.color import rgba2rgb, rgb2gray
import skimage.segmentation
import scipy
# from scipy.misc import imread
import matplotlib.image as img
import skimage.io
import os
import fnmatch

classes = ('IGNORE', 'Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agricultural')
classes_numpy = np.arange(len(classes))


def plot_barchart_1(pole, envir):
    performance = pole
    plt.bar(classes_numpy, performance, align='center', alpha=0.5)
    plt.xticks(classes_numpy, classes)
    plt.ylabel('Percentage of pixels')
    plt.xlabel('Classes')
    plt.title('Procenta pixelů v jednotlivých třídách - ' + envir)
    plt.savefig('barchart_percent_' + envir + '.png', bbox_inches='tight')




folder_dir = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/Rural/images_png"
folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/"

global_counter = 0

for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"

#ifelse na count

    pixels_classes = np.zeros((8,), dtype=int)

    folder_dir_2 = folder_dir_1 + "masks_png" + "/"
    count = len(fnmatch.filter(os.listdir(folder_dir_2), '*.*'))
    counter = 0
    for images in os.listdir(folder_dir_2):
        file_name = "LoveDA/Train/" + folder_level_1 + "/masks_png/" + images
        im = skimage.io.imread(file_name)


        pixels_classes += [np.sum(im == 0), np.sum(im == 1), np.sum(im == 2), np.sum(im == 3),
                                           np.sum(im == 4),
                                           np.sum(im == 5), np.sum(im == 6),
                                           np.sum(im == 7)]  # print(np.sum(im[:,:]==1))

        counter += 1
        global_counter += 1

    plot_barchart_1(pixels_classes, folder_level_1)
    if global_counter < len(fnmatch.filter(os.listdir(folder_dir_base + '/Rural/masks_png'), '*.*')) + len(fnmatch.filter(os.listdir(folder_dir_base + '/Urban/masks_png'), '*.*')) - 10:
        pixel_copy = pixels_classes

plot_barchart_1(np.add(pixels_classes, class_occur_copy), 'global')
# diverse_images_global = np.zeros((8,), dtype=int)
# diverse_images_global = [np.sum(diverse_classes_global == 1), np.sum(diverse_classes_global == 2),
#                          np.sum(diverse_classes_global == 3), np.sum(diverse_classes_global == 4),
#                          np.sum(diverse_classes_global == 5), np.sum(diverse_classes_global == 6),
#                          np.sum(diverse_classes_global == 7), np.sum(diverse_classes_global == 8)]
#
# plot_barchart_2(diverse_classes, 'global')