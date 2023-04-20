import os

import skimage
from sklearn import datasets
from skimage.feature import hog
import numpy as np
import sklearn.model_selection
from sklearn import svm
import sklearn.naive_bayes
import statistics as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# trenovaci data
data_matrix = np.load('./saved/dataset_hog.npy')
target_vect = np.ravel(np.load('./saved/mask_vect_hog.npy'))
data_matrix = np.load('./saved/dataset_hog_all.npy')  # celý dataset
target_vect = np.ravel(np.load('./saved/mask_vect_hog_all.npy'))  # celý dataset

NUMBER_IMAGES_VAL = 1669  # celý dataset TRAIN
HOG_ORIENT = len(data_matrix[1,:])
CELL_C = 32
CELL_C = 16
IMAGE_LEN = 1024
dim = 9


global_counter_img = 0



folder_dir_base = "./LoveDA/Val/"

# svc = svm.SVC()
# svc.fit(data_vect_lbp, target_vect_lbp)

# bayes
gnb = sklearn.naive_bayes.GaussianNB(priors=None)
gnb.fit(data_matrix, target_vect)
print(gnb.n_features_in_)

# VMS
# max_iter=10
svc = svm.SVC(max_iter=10)
svc.fit(data_matrix, target_vect)
print(svc.n_features_in_)


test_data = np.zeros((1024*1024, dim), dtype=int)
test_target = np.zeros((1024*1024, 1), dtype=int)



# y_pred = gnb.predict(test_img_data).reshape(-1, 1) # bayes
# # y_pred = svc.predict(test_img_data).reshape(-1, 1)  # SVM


# inicializace vystupnich poli
hog_scale = int(IMAGE_LEN/CELL_C)
dataset_hog_val = np.zeros((hog_scale*hog_scale*NUMBER_IMAGES_VAL, HOG_ORIENT), dtype=float)
mask_vect_val = np.zeros((hog_scale*hog_scale*NUMBER_IMAGES_VAL, 1), dtype=int)

for folder_level_1 in sorted(os.listdir(folder_dir_base), key=len):
    folder_dir_1 = folder_dir_base + folder_level_1 + "/"
    folder_dir_2 = folder_dir_1 + "images_png" + "/"

    for image in sorted(os.listdir(folder_dir_2), key=len):
        # ziskani obrazku a jeho masky
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
        mask_vect_val[mez1:mez2] = mask_hog_flatten

        # vytvareni datasetu
        # fv = np.zeros(1024, 1024, 3)
        fv = hog(img, orientations=HOG_ORIENT, pixels_per_cell=(CELL_C, CELL_C), cells_per_block=(1, 1),
                 visualize=False, channel_axis=2)
        fv = np.reshape(fv, (len(mask_hog_flatten), HOG_ORIENT))
        dataset_hog_val[mez1:mez2, 0:HOG_ORIENT] = fv

        global_counter_img += 1


mode_isnot_zero = mask_vect_val!=0
np.save('./saved/mode_isnot_zero', mode_isnot_zero)
dataset_hog_val = dataset_hog_val[np.reshape(mode_isnot_zero, (len(mode_isnot_zero), )), 0:HOG_ORIENT]
mask_vect_val = mask_vect_val[mode_isnot_zero]


y_pred = gnb.predict(dataset_hog_val).reshape(-1, 1)


y_pred = y_pred.reshape(-1)
mask_vect_val = mask_vect_val.reshape(-1)

print("Number of mislabeled points: ", np.sum(mask_vect_val != y_pred))
print("Podíl správné klasifikace: ", np.divide(np.sum(mask_vect_val != y_pred), len(y_pred)))
print("y_pred len: ", np.shape(y_pred))
print("mask_vect_val: ", np.shape(mask_vect_val))
print("dataset_hog_val: ", np.shape(dataset_hog_val))

# y_pred_img = np.reshape(y_pred, (int(IMAGE_LEN/CELL_C), int(IMAGE_LEN/CELL_C)))




# # # Vykresleni---------------------------------------------------------------------
# #
# # cmap1 = plt.cm.gray
#
# num_classes = 8
# colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
# myCmap = ListedColormap(colors)
# class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
# bounds = np.arange(len(class_labels) + 1)
# norm = BoundaryNorm(bounds, len(class_labels))
#
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5))
# # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
# ax1.axis('off')
# ax1.imshow(img, cmap=plt.cm.gray)
# ax1.set_title('Input image')
#
# ax2.axis('off')
# ax2.imshow(mask, cmap=myCmap, norm=norm)
# ax2.set_title('Mask')
#
# # im_hybrid = np.hstack((im_m, y_pred_img))
#
# ax3.axis('off')
# ax3.imshow(mask_visual, cmap=myCmap, norm=norm)
# ax3.set_title('Mask reduced')
#
# ax4.axis('off')
# ax4.imshow(y_pred_img, cmap=myCmap, norm=norm)
# ax4.set_title('Predicted reduced mask')
#
# plt.show()