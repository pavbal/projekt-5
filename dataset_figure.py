import numpy as np
import matplotlib.pyplot as plt
import skimage
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

img = skimage.io.imread("./LoveDA/Train/Rural/images_png/50.png")
mask = skimage.io.imread("./LoveDA/Train/Rural/masks_png/50.png")

num_classes = 8
colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
colors = ['#000000', '#666666', '#d22d04', '#840f8f', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']

#840f8f
myCmap = ListedColormap(colors)
class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
bounds = np.arange(len(class_labels) + 1)
norm = BoundaryNorm(bounds, len(class_labels))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
# fig.suptitle('Dvojice snímek-maska', fontsize=14, )
# # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
# ax1.axis('off')
# ax1.imshow(img, cmap=plt.cm.gray)
# ax1.set_title('Výchozí snímek')
#
# ax2.axis('off')
# im = ax2.imshow(mask, cmap=myCmap, norm=norm)
# ax2.set_title('Maska snímku')
#
# divider = make_axes_locatable(ax2)
# cax = divider.append_axes('right', size='5%', pad=0.1)
# cb = fig.colorbar(im, cax=cax)
#
# ticks = np.arange(len(descriptions)) + 0.5
# cb.set_ticks(ticks)
# cb.set_ticklabels(descriptions)
#
# # Set the title for the colorbar
# cb.ax.set_title('Třídy', fontsize=12)
#
# plt.savefig('images/figure.png', dpi=300, bbox_inches='tight')
# # plt.show()


#svg:
import numpy as np
import matplotlib.pyplot as plt
import skimage
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import generic_filter

from methods.methods import smooth_mask

# def smooth_mask(mask):
#     def replace_if_surrounded(arr):
#         center = arr[len(arr) // 2]
#         if np.sum(arr == center) >= 6:
#             return center
#         else:
#             return arr[0]
#
#     return generic_filter(mask, replace_if_surrounded, size=(3, 3))


img = skimage.io.imread("./LoveDA/Train/Rural/images_png/50.png")
mask = skimage.io.imread("./LoveDA/Train/Rural/masks_png/50.png")

num_classes = 8
colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']

myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=num_classes)
class_labels = [0, 1, 2, 3, 4, 5, 6, 7]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Dvojice snímek-maska', fontsize=14, )
ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Výchozí snímek')

ax2.axis('off')
mask_smoothed = smooth_mask(mask)
im = ax2.imshow(mask_smoothed, cmap=myCmap, vmin=0, vmax=num_classes - 1)
ax2.set_title('Maska snímku')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.1)
cb = fig.colorbar(im, cax=cax, ticks=class_labels)

# navíc
ticks = (np.arange(len(descriptions)) + 0.5)/1.14
cb.set_ticks(ticks)
cb.set_ticklabels(descriptions)
cb.ax.tick_params(labelsize=12)

cb.ax.set_title('Třídy', fontsize=12)
# plt.savefig('images/figure.svg', format='svg', bbox_inches='tight', dpi=300)
# plt.savefig('images/image-mask.png', dpi=300, bbox_inches='tight')
plt.show()