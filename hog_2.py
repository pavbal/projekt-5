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
CELL_C = 16
# ostatni konstanty
HOG_ORIENT = 9
NUMBER_IMAGES = 16
NUMBER_IMAGES = 2522 # celý dataset TRAIN
IMAGE_LEN = 1024
test_img_number = 3200
# cesta
folder_dir_base = "./LoveDA_Train_16/"
folder_dir_base = "./LoveDA/Train/" # celý dataset

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
        # file_name_image = "LoveDA_Train_16/" + folder_level_1 + "/images_png/" + image
        # file_name_mask = "LoveDA_Train_16/" + folder_level_1 + "/masks_png/" + image
        file_name_image = folder_dir_base + folder_level_1 + "/images_png/" + image # celý dataset
        file_name_mask = folder_dir_base + folder_level_1 + "/masks_png/" + image # celý dataset
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
        # fv = np.zeros(1024, 1024, 3)
        fv = hog(img, orientations=HOG_ORIENT, pixels_per_cell=(CELL_C, CELL_C), cells_per_block=(1, 1),
                 visualize=False, channel_axis=2)
        fv = np.reshape(fv, (len(mask_hog_flatten), HOG_ORIENT))
        dataset_hog[mez1:mez2, 0:HOG_ORIENT] = fv

        global_counter_img += 1


mode_isnot_zero = mask_vect!=0
np.save('./saved/mode_isnot_zero', mode_isnot_zero)
dataset_hog = dataset_hog[np.reshape(mode_isnot_zero, (len(mode_isnot_zero), )), 0:HOG_ORIENT]
mask_vect = mask_vect[mode_isnot_zero]

# dataset_hog = dataset_hog[mode_isnot_zero[:,0], 0:HOG_ORIENT]

# for i in range(0, len(mask_vect)): #neefektivní, upravit!
#     if mask_vect[i] == 0:
#         mask_vect = np.delete(mask_vect, i, 0)
#         dataset_hog = np.delete(dataset_hog, i, 0)


print("dataset shape: ", dataset_hog.shape)
print("mask_vect shape: ", mask_vect.shape)
# np.save('./saved/dataset_hog', dataset_hog)
# np.save('./saved/mask_vect_hog', mask_vect)
np.save('./saved/dataset_hog_all', dataset_hog) # celý dataset
np.save('./saved/mask_vect_hog_all', mask_vect) # celý dataset

## testovaci prvek

img = skimage.io.imread("./LoveDA/Val/Rural/images_png/" + str(test_img_number) + ".png")
mask = skimage.io.imread("./LoveDA/Val/Rural/masks_png/" + str(test_img_number) + ".png")

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