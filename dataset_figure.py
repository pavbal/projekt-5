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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Dvojice snímek-maska', fontsize=14, )
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Výchozí snímek')

ax2.axis('off')
im = ax2.imshow(mask, cmap=myCmap, norm=norm)
ax2.set_title('Maska snímku')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.1)
cb = fig.colorbar(im, cax=cax)

ticks = np.arange(len(descriptions)) + 0.5
cb.set_ticks(ticks)
cb.set_ticklabels(descriptions)

# Set the title for the colorbar
cb.ax.set_title('Třídy', fontsize=12)

plt.savefig('images/figure.png', dpi=300, bbox_inches='tight')
plt.show()