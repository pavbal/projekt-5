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

classes = ('1', '2', '3', '4', '5', '6', '7', '8')
classes_numpy = np.arange(len(classes))


def plot_barchart_2(pole, envir):
    performance = pole
    plt.figure(figsize=(8.5, 6))
    plt.bar(classes_numpy, performance, align='center', alpha=0.5)
    plt.xticks(classes_numpy, classes)
    plt.ylabel('Number of images')
    plt.xlabel('Number of classes')
    plt.title('Počety tříd na jednotlivých obrázcích - ' + envir)
    plt.savefig('barchart_diverse_' + envir + '.png', bbox_inches='tight')


folder_dir = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/Rural/images_png"
folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/"

diverse_classes_global = np.zeros((1366 + 1156,), dtype=int)
global_counter = 0

for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"
    if folder_level_1 == 'Rural':
        count = 1366
    else:
        count = 1156
    diverse_classes = np.zeros((count,), dtype=int)

    folder_dir_2 = folder_dir_1 + "masks_png" + "/"
    counter = 0
    for images in os.listdir(folder_dir_2):
        file_name = "LoveDA/Train/" + folder_level_1 + "/masks_png/" + images
        im = skimage.io.imread(file_name)
        diverse_classes[counter] = len(np.unique(im))
        diverse_classes_global[global_counter] = len(np.unique(im))
        counter += 1
        global_counter += 1

    # print(pixels_classes)
    diverse_images = np.zeros((8,), dtype=int)
    diverse_images = [np.sum(diverse_classes == 1), np.sum(diverse_classes == 2),
                      np.sum(diverse_classes == 3), np.sum(diverse_classes == 4),
                      np.sum(diverse_classes == 5), np.sum(diverse_classes == 6),
                      np.sum(diverse_classes == 7), np.sum(diverse_classes == 8)]
    if global_counter < 2000:
        diverse_images_copy = diverse_images

    plot_barchart_2(diverse_images, folder_level_1)
diverse_images_global = np.add(diverse_images, diverse_images_copy)
plot_barchart_2(diverse_images_global, 'global')
##diverse_images_global = np.zeros((8,), dtype=int)
# diverse_images_global = [np.sum(diverse_classes_global == 1), np.sum(diverse_classes_global == 2),
#                          np.sum(diverse_classes_global == 3), np.sum(diverse_classes_global == 4),
#                          np.sum(diverse_classes_global == 5), np.sum(diverse_classes_global == 6),
#                          np.sum(diverse_classes_global == 7), np.sum(diverse_classes_global == 8)]
#
# plot_barchart_2(diverse_classes, 'global')
