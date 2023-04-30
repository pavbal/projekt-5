## testovaci prvek
import numpy as np
import skimage
from skimage.feature import hog
import statistics as st

CELL_C = 16
# ostatni konstanty
HOG_ORIENT = 9
test_img_number = 3547

img = skimage.io.imread("./LoveDA/Val/Urban/images_png/" + str(test_img_number) + ".png")
mask = skimage.io.imread("./LoveDA/Val/Urban/masks_png/" + str(test_img_number) + ".png")

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
# mez1 = len(mask_hog_flatten)*global_counter_img
# mez2 = len(mask_hog_flatten)*global_counter_img+len(mask_hog_flatten)

# vytvareni datasetu
fv = hog(img, orientations=HOG_ORIENT, pixels_per_cell=(CELL_C, CELL_C), cells_per_block=(1, 1),
         visualize=False, channel_axis=2)
fv = np.reshape(fv, (len(mask_hog_flatten), HOG_ORIENT))

np.save('./saved/test_img_vect_hog', fv)
np.save('./saved/test_mask_hog', mask_hog_flatten)