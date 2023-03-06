import skimage
from skimage import io
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats as st
import statistics as st
import os

# Konstanta (velikost bunky)
CELL_C = 32
# ostatni konstanty
HOG_ORIENT = 9
NUMBER_IMAGES = 16
IMAGE_LEN = 1024
test_img_number = 23
# cesta
folder_dir_base = "./LoveDA_Train_16/"

global_counter_img = 0

# inicializace vystupnich poli
hog_scale = int(IMAGE_LEN/CELL_C)
dataset_hog = np.zeros((hog_scale*hog_scale*NUMBER_IMAGES, HOG_ORIENT), dtype=float)
mask_vect = np.zeros((hog_scale*hog_scale*NUMBER_IMAGES, 1), dtype=int)


# pro
for folder_level_1 in os.listdir(folder_dir_base):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"
    folder_dir_2 = folder_dir_1 + "images_png" + "/"

    for image in os.listdir(folder_dir_2):
        # ziskani obrazku a jeho masky
        file_name_image = "LoveDA_Train_16/" + folder_level_1 + "/images_png/" + image
        file_name_mask = "LoveDA_Train_16/" + folder_level_1 + "/masks_png/" + image
        img = skimage.io.imread(file_name_image)
        mask = skimage.io.imread(file_name_mask)

        # uprava masky pro hog
        scale_mask = int(len(mask) / CELL_C)
        shape_mask = (scale_mask, scale_mask)
        mask_hog = np.zeros(shape_mask, dtype=int)

        for x in range(len(mask_hog)): # jde o délku jedné strany, nikoliv počet prvků
            for y in range(len(mask_hog)):
                mezX1 = x * CELL_C
                mezX2 = mezX1 + CELL_C - 1
                mezY1 = y * CELL_C
                mezY2 = mezY1 + CELL_C - 1
                cell = mask[mezX1:mezX2, mezY1:mezY2]
                mask_hog[x, y] = st.mode(cell.flatten())


        # vytvareni vektoru masek
        # mask_hog_flatten = mask_hog.flatten()
        mask_hog_flatten = np.reshape(mask_hog, (len(mask_hog)*len(mask_hog), 1))
        mez1 = len(mask_hog_flatten)*global_counter_img
        mez2 = len(mask_hog_flatten)*global_counter_img+len(mask_hog_flatten)
        mask_vect[mez1:mez2] = mask_hog_flatten

        # vytvareni datasetu
        fv = hog(img, orientations=HOG_ORIENT, pixels_per_cell=(CELL_C, CELL_C), cells_per_block=(1, 1),
                 visualize=False, channel_axis=2)
        fv = np.reshape(fv, (len(mask_hog_flatten), HOG_ORIENT))
        dataset_hog[mez1:mez2, 0:HOG_ORIENT] = fv

        global_counter_img += 1

print("dataset shape: ", dataset_hog.shape)
print("mask_vect shape: ", mask_vect.shape)
np.save('./saved/dataset_hog', dataset_hog)
np.save('./saved/mask_vect_hog', mask_vect)

## testovaci prvek

img = skimage.io.imread("./LoveDA_Test_16/Rural/images_png/" + str(test_img_number) + ".png")
mask = skimage.io.imread("./LoveDA_Test_16/Rural/masks_png/" + str(test_img_number) + ".png")

# uprava testovaci masky pro hog
scale_mask = int(len(mask) / CELL_C)
shape_mask = (scale_mask, scale_mask)
mask_hog = np.zeros(shape_mask, dtype=int)

for x in range(len(mask_hog)): # jde o délku jedné strany, nikoliv počet prvků
    for y in range(len(mask_hog)):
        mezX1 = x * CELL_C
        mezX2 = mezX1 + CELL_C - 1
        mezY1 = y * CELL_C
        mezY2 = mezY1 + CELL_C - 1
        cell = mask[mezX1:mezX2, mezY1:mezY2]
        mask_hog[x, y] = st.mode(cell.flatten())

# vytvareni vektoru masek
# mask_hog_flatten = mask_hog.flatten()
mask_hog_flatten = np.reshape(mask_hog, (len(mask_hog)*len(mask_hog), 1))
mez1 = len(mask_hog_flatten)*global_counter_img
mez2 = len(mask_hog_flatten)*global_counter_img+len(mask_hog_flatten)

# vytvareni datasetu
fv = hog(img, orientations=HOG_ORIENT, pixels_per_cell=(CELL_C, CELL_C), cells_per_block=(1, 1),
         visualize=False, channel_axis=2)
fv = np.reshape(fv, (len(mask_hog_flatten), HOG_ORIENT))

np.save('./saved/test_img_vect_hog', fv)
np.save('./saved/test_mask_hog', mask_hog_flatten)