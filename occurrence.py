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


def plot_barchart_3(pole, envir):
    performance = pole
    plt.figure(figsize=(8.5, 6))
    plt.bar(classes_numpy, performance, align='center', alpha=0.5)
    plt.xticks(classes_numpy, classes)
    plt.ylabel('Number of images')
    plt.xlabel('Classes')
    plt.title('Počty obrázků s výskytem jednotlivých tříd - ' + envir)
    plt.savefig('barchart_occur_' + envir + '.png', bbox_inches='tight')


folder_dir = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/Rural/images_png"
folder_dir_base = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA/Train/"

global_counter = 0

for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"

#ifelse na count

    class_occur = np.zeros((8,), dtype=int)

    folder_dir_2 = folder_dir_1 + "masks_png" + "/"
    count = len(fnmatch.filter(os.listdir(folder_dir_2), '*.*'))
    counter = 0
    for images in os.listdir(folder_dir_2):
        file_name = "LoveDA/Train/" + folder_level_1 + "/masks_png/" + images
        im = skimage.io.imread(file_name)

        # class_occur += [np.sum(im==0)>0, np.sum(im==1)>0, np.sum(im==2)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0, np.sum(im==0)>0]
        class_occur += [np.any(im[:, :] == 0), np.any(im[:, :] == 1), np.any(im[:, :] == 2), np.any(im[:, :] == 3),
                        np.any(im[:, :] == 4), np.any(im[:, :] == 5), np.any(im[:, :] == 6), np.any(im[:, :] == 7)]

        counter += 1
        global_counter += 1

    plot_barchart_3(class_occur, folder_level_1)
    if global_counter < len(fnmatch.filter(os.listdir(folder_dir_base + '/Rural/masks_png'), '*.*')) + len(fnmatch.filter(os.listdir(folder_dir_base + '/Urban/masks_png'), '*.*')) - 10:
        class_occur_copy = class_occur

plot_barchart_3(np.add(class_occur, class_occur_copy), 'global')
# diverse_images_global = np.zeros((8,), dtype=int)
# diverse_images_global = [np.sum(diverse_classes_global == 1), np.sum(diverse_classes_global == 2),
#                          np.sum(diverse_classes_global == 3), np.sum(diverse_classes_global == 4),
#                          np.sum(diverse_classes_global == 5), np.sum(diverse_classes_global == 6),
#                          np.sum(diverse_classes_global == 7), np.sum(diverse_classes_global == 8)]
#
# plot_barchart_2(diverse_classes, 'global')