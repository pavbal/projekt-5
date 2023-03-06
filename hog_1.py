import skimage
from skimage import io
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
# from scipy import stats as st
import statistics as st

# Konstanty a inicializace proměnných
CELL_C = 16

img =  io.imread('./LoveDA_Train_16/Rural/images_png/0.png')
mask = io.imread('./LoveDA_Train_16/Rural/masks_png/0.png')
img_g = skimage.color.rgb2gray(img)

# vypocet HOG
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(CELL_C, CELL_C),
                    cells_per_block=(1, 1), visualize=True, channel_axis=2)
fv = hog(img, orientations=9, pixels_per_cell=(CELL_C, CELL_C), cells_per_block=(1, 1), visualize=False, channel_axis=2)

# preddefinovani HOG masky
scale_mask = int(len(mask)/CELL_C)
shape_mask = (scale_mask, scale_mask)
mask_hog = np.zeros(shape_mask, dtype=int)
print(mask_hog.shape)

# vytvoreni hog masky
for x in range(len(mask_hog)):
    for y in range(len(mask_hog)):
        mezX1 = x*CELL_C
        mezX2 = mezX1 + CELL_C -1
        mezY1 = y*CELL_C
        mezY2 = mezY1 + CELL_C - 1
        cell = mask[mezX1:mezX2, mezY1:mezY2]
        mask_hog[x,y] = st.mode(cell.flatten())


print("numbers:")
print("delka narovnane mask_hog: ", len(mask_hog.flatten()))
print("delka vektoru priznaku: ", len(fv))
print("vypocet: ", len(fv)/len(mask_hog.flatten())/9)
print()


folder_dir_base = "./LoveDA_Train_16/"

print("hog_image shape", hog_image.shape)
print("fd shape: ", fd.shape)
print("hog_image size: ", np.size(hog_image))
print("fv shape: ", fv.shape)


# Vykresleni
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))#, sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')

ax3.axis('off')
ax3.imshow(mask, cmap=plt.cm.gray)
ax3.set_title('Mask original')

ax4.axis('off')
ax4.imshow(mask_hog, cmap=plt.cm.gray)
ax4.set_title('Mask for HOG')

plt.show()