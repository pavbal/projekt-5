import skimage
from skimage.feature import local_binary_pattern
from sklearn import datasets
import numpy as np
import sklearn.model_selection
from sklearn import svm
import sklearn.naive_bayes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data_matrix = np.load('./saved/dataset_rgb_1im.npy')
target_vect = np.ravel(np.load('./saved/mask_vect_rgb_1im.npy'))

print(data_matrix.shape)
print(target_vect.shape)
dim = 3

# svc = svm.SVC()
# svc.fit(data_vect_lbp, target_vect_lbp)

test_data = np.zeros((1024*1024, dim), dtype=int)
test_target = np.zeros((1024*1024, 1), dtype=int)

radius = 1
n_points = 8 * radius
METHOD = 'uniform'
global_counter = 0

# bayes
gnb = sklearn.naive_bayes.GaussianNB(priors=None)
gnb.fit(data_matrix, target_vect)
print(gnb.n_features_in_)

# VMS
# max_iter=10
svc = svm.SVC(max_iter=4)
svc.fit(data_matrix, target_vect)
print(svc.n_features_in_)

im_i = skimage.io.imread("./LoveDA_Train_16/Rural/images_png/6.png")
im_i_g = skimage.color.rgb2gray(im_i)
im_m = skimage.io.imread("./LoveDA_Train_16/Rural/masks_png/6.png")

# plt.imshow(im_i)
# plt.show()


im_lbp_rgb = np.copy(im_i)
im_lbp_rgb[:, :, 0] = local_binary_pattern(im_i[:, :, 0], n_points, radius, METHOD)
im_lbp_rgb[:, :, 1] = local_binary_pattern(im_i[:, :, 1], n_points, radius, METHOD)
im_lbp_rgb[:, :, 2] = local_binary_pattern(im_i[:, :, 2], n_points, radius, METHOD)

lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)



for x in range(np.size(im_m, 0)):
    for y in range(np.size(im_m, 1)):
        if im_m[x, y] != 0:
            test_data[global_counter, 0] = im_lbp_rgb[x, y, 0]
            test_data[global_counter, 1] = im_lbp_rgb[x, y, 1]
            test_data[global_counter, 2] = im_lbp_rgb[x, y, 2]
            if dim == 4:
                test_data[global_counter, 3] = lbp[x, y] * 2
            test_target[global_counter,] = im_m[x, y]
        global_counter += 1

# y_pred = gnb.predict(test_data).reshape(-1, 1) # bayes
y_pred = svc.predict(test_data).reshape(-1, 1)  # SVM


# print(y_pred.shape)
# print("Number of mislabeled points : %d" % np.sum(test_target != y_pred))
# print(test_target != y_pred)
print("Number of mislabeled points : %d" % np.sum(test_target != y_pred))
# test_data_img = np.reshape(skimage.color.rgb2gray(test_data), (1024, 1024))
# test_data_img = np.reshape(test_data[:,1], (1024, 1024))
y_pred_img = np.reshape(y_pred, (1024, 1024))

# Vykresleni---------------------------------------------------------------------

cmap1 = plt.cm.gray


num_classes = 8
colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
myCmap = ListedColormap(colors)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(im_i, cmap=plt.cm.gray)
ax1.set_title('Input image')

ax2.axis('off')
ax2.imshow(im_m)#, cmap=plt.cm.gray)
ax2.set_title('Mask')

im_hybrid = np.hstack((im_m, y_pred_img))

ax3.axis('off')
ax3.imshow(lbp, cmap=plt.cm.gray)
ax3.set_title('LBP')

ax4.axis('off')
ax4.imshow(y_pred_img)#, cmap=plt.cm.gray)
ax4.set_title('Predicted')



plt.show()

