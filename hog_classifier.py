import skimage
from sklearn import datasets
from skimage.feature import hog
import numpy as np
import sklearn.model_selection
from sklearn import svm
import sklearn.naive_bayes
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# trenovaci data
data_matrix = np.load('./saved/dataset_hog.npy')
target_vect = np.ravel(np.load('./saved/mask_vect_hog.npy'))
data_matrix = np.load('./saved/dataset_hog_all.npy') # celý dataset
target_vect = np.ravel(np.load('./saved/mask_vect_hog_all.npy')) # celý dataset

HOG_ORIENT = len(data_matrix[1,:])
CELL_C = 32
CELL_C = 16
IMAGE_LEN = 1024
dim = 9
test_img_number = 3200

# testovaci data
test_img_data = np.load('./saved/test_img_vect_hog.npy')
test_mask = np.load('./saved/test_mask_hog.npy')
mask_visual = np.reshape(test_mask, (int(IMAGE_LEN/CELL_C), int(IMAGE_LEN/CELL_C)))

# originalni maska a obrazek pro vizualizaci
img = skimage.io.imread("./LoveDA/Val/Rural/images_png/" + str(test_img_number) + ".png")
mask = skimage.io.imread("./LoveDA/Val/Rural/masks_png/" + str(test_img_number) + ".png")


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



y_pred = gnb.predict(test_img_data)#.reshape(-1, 1) # bayes
# y_pred = svc.predict(test_img_data).reshape(-1, 1)  # SVM

print("y_pred shape: ", y_pred.shape)
print("test_mask shape: ", test_mask.shape)
print("test_data shape: ", test_img_data.shape)

print("Number of mislabeled points : %d" % np.sum(test_mask != y_pred))

y_pred_img = np.reshape(y_pred, (int(IMAGE_LEN/CELL_C), int(IMAGE_LEN/CELL_C)))


# # Vykresleni---------------------------------------------------------------------
#
# cmap1 = plt.cm.gray

num_classes = 8
colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
myCmap = ListedColormap(colors)
class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
bounds = np.arange(len(class_labels) + 1)
norm = BoundaryNorm(bounds, len(class_labels))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Výchozí snímek')

ax2.axis('off')
ax2.imshow(mask, cmap=myCmap, norm=norm)
ax2.set_title('Maska snímku')

# im_hybrid = np.hstack((im_m, y_pred_img))

ax3.axis('off')
ax3.imshow(mask_visual, cmap=myCmap, norm=norm)
ax3.set_title('Maska pro HOG')

ax4.axis('off')
ax4.imshow(y_pred_img, cmap=myCmap, norm=norm)
ax4.set_title('Predikovaná maska klasifikátorem GNB')

plt.savefig('results_visualisation/hog_result.png')
plt.show()
