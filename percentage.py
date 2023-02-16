import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data
from skimage.color import rgba2rgb, rgb2gray
import skimage.segmentation
import scipy
#from scipy.misc import imread
import matplotlib.image as img
import skimage.io
import os
import fnmatch
from os import listdir
from urllib.request import urlopen

classes = ('IGNORE', 'Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agricultural')
classes_numpy = np.arange(len(classes))
# This is a Python script for LoveDA dataset statistics.

def plot_barchart_1(pole, envir):
    performance = pole
    plt.bar(classes_numpy, performance, align='center', alpha=0.5)
    plt.xticks(classes_numpy, classes)
    plt.ylabel('Percentage of pixels')
    plt.xlabel('Classes')
    plt.title('Procenta pixelů v jednotlivých třídách - ' + envir)
    plt.savefig('barchart_percent_' + envir + '.png', bbox_inches='tight')
    # plt.show()

# im = skimage.io.imread("LoveDA/Train/Urban/images_png/1400.png")
# plt.figure() #arg?: figsize=(10,5)
# plt.imshow(im[:,:,:])
# plt.show()
#
# im = skimage.io.imread("LoveDA/Train/Urban/masks_png/1400.png")
# plt.figure()
# plt.imshow(im[:,:])
# plt.show()

#folder_dir = "C:/Users/RIJUSHREE/Desktop/Gfg images"

folder_dir = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/Rural/images_png"
folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/"
counter = 0
global_counter = 0

#delka_pole = len(os.listdir(folder_dir))
# pole = np.zeros((8,), dtype= int)
# print("delka seznamu = " + str(len(os.listdir(folder_dir_base))))
# print(os.listdir(folder_dir_base)[1])
for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"
    global_counter
    for folder_level_2 in os.listdir(folder_dir_1):
        folder_dir_2 = folder_dir_1 + folder_level_2 + "/"
        counter = 0
        pixels_classes = np.zeros((8,), dtype = int)

        for images in os.listdir(folder_dir_2):

            # if (images.endswith(".png")):
            # file_name = folder_dir_2 + images
            file_name = "LoveDA/Train/" + folder_level_1 + "/" + folder_level_2 + "/" + images
            # im = img.imread(file_name)
            im = skimage.io.imread(file_name)
            if (folder_level_2 == "masks_png"):
                pixels_classes = pixels_classes + [np.sum(im == 0), np.sum(im == 1), np.sum(im == 2), np.sum(im == 3),
                                                   np.sum(im == 4),
                                                   np.sum(im == 5), np.sum(im == 6),
                                                   np.sum(im == 7)]  # print(np.sum(im[:,:]==1))
                unique = np.unique(im)
                global_counter = global_counter + 1
            counter = counter + 1


        if (folder_level_2 == "masks_png"):
            print("{2}/{3}: Celkovy pocet pixelu = {0}, Soucet prvku pole = {1}".format(counter * 1024 * 1024,
                                                                                        sum(pixels_classes),
                                                                                        folder_level_1,
                                                                                        folder_level_2))
            plt.figure(figsize=(8.5, 6))
            # print(pixels_classes)
            pixels_classes_percentage = (pixels_classes/sum(pixels_classes))*100
            if global_counter < len(fnmatch.filter(os.listdir(folder_dir_base + '/Rural/masks_png'), '*.*')) + len(fnmatch.filter(os.listdir(folder_dir_base + '/Urban/masks_png'), '*.*')) - 10:
                pixels_classes_percentage_copy = pixels_classes_percentage
                pixels_classes_copy = pixels_classes

            print(pixels_classes_percentage)
            plot_barchart_1(pixels_classes_percentage, folder_level_1)



        print("{0}/{1}: pocet snimku: {2}".format(folder_level_1, folder_level_2, counter))
        # print("{2}/{3}: Celkovy pocet pixelu = {0}, Soucet prvku pole = {1}".format(counter*1024*1024,
        #                                            sum(pixels_classes), folder_level_1, folder_level_2))
        # print(pixels_classes)
        print("")

plt.figure(figsize=(8.5, 6))
# plot_barchart(np.multiply(np.divide(np.add(pixels_classes, pixels_classes_copy), np.add(sum(pixels_classes), sum(pixels_classes_copy))), 100), "global") # CHYBAAAAAAAAAAAAAA

count_urban = len(fnmatch.filter(os.listdir(folder_dir_base + '/Urban/masks_png'), '*.*'))
count_rural = len(fnmatch.filter(os.listdir(folder_dir_base + '/Rural/masks_png'), '*.*'))
rural_mpy = np.multiply(pixels_classes_percentage_copy, count_rural)
urban_mpy = np.multiply(pixels_classes_percentage, count_urban)
global_percent = np.divide(np.add(rural_mpy, urban_mpy), (count_rural + count_urban))
plot_barchart(global_percent, 'global')