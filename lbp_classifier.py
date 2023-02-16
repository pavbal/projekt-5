import skimage
from skimage.feature import local_binary_pattern
from sklearn import datasets
import numpy as np
import sklearn.model_selection
from sklearn import svm
import sklearn.naive_bayes
import matplotlib.pyplot as plt

data_vect_lbp = np.load('./saved/data_vect_lbp_2.npy').reshape(-1, 1)
target_vect_lbp = np.load('./saved/target_vect_lbp_2.npy')

print(data_vect_lbp.shape)
print(target_vect_lbp.shape)

# svc = svm.SVC()
# svc.fit(data_vect_lbp, target_vect_lbp)

test_data = np.zeros((1024*1024, 1), dtype=int)
test_target = np.zeros((1024*1024, 1), dtype=int)

radius = 1
n_points = 8 * radius
METHOD = 'uniform'
global_counter = 0

gnb = sklearn.naive_bayes.GaussianNB()
gnb.fit(data_vect_lbp, target_vect_lbp)

svc = svm.LinearSVC(max_iter=10)
svc.fit(data_vect_lbp, target_vect_lbp)

im_i = skimage.io.imread("./LoveDA_Test_16/Rural/images_png/23.png")
im_i_g = skimage.color.rgb2gray(im_i)
im_m = skimage.io.imread("./LoveDA_Test_16/Rural/masks_png/23.png")

lbp = local_binary_pattern(im_i_g, n_points, radius, METHOD)

for x in range(np.size(im_m, 0)):
    for y in range(np.size(im_m, 1)):
        if im_m[x, y] != 0:
            test_data[global_counter, ] = lbp[x, y]
            test_target[global_counter, ] = im_m[x, y]
        global_counter += 1

# y_pred = gnb.predict(test_data).reshape(-1, 1)
y_pred = svc.predict(test_data).reshape(-1, 1)

# print(y_pred.shape)
# print("Number of mislabeled points : %d" % np.sum(test_target != y_pred))
# print(test_target != y_pred)

test_data_img = np.reshape(test_data, (1024, 1024))
y_pred_img = np.reshape(y_pred, (1024, 1024))



fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(im_i, cmap=plt.cm.gray)
ax1.set_title('Input image')


ax2.axis('off')
ax2.imshow(im_m)#, cmap=plt.cm.gray)
ax2.set_title('Mask')

ax3.axis('off')
ax3.imshow(test_data_img, cmap=plt.cm.gray)
ax3.set_title('LBP')

ax4.axis('off')
ax4.imshow(y_pred_img)#, cmap=plt.cm.gray)
ax4.set_title('Predicted')

print(y_pred_img)
print(im_m)

plt.show()




# fig = plt.figure()
# plt.subplot(1, 3, 1)
# plt.imshow(im_i)
#
# plt.show()
