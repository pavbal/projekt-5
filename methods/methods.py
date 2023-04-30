from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import generic_filter
import numpy as np

def smooth_mask(mask):
    def replace_if_surrounded(arr):
        center = arr[len(arr) // 2]
        if np.sum(arr == center) >= 6:
            return center
        else:
            return arr[0]

    return generic_filter(mask, replace_if_surrounded, size=(3, 3))



def plot_two_with_map(im_m, y_pred_img, im_i=None, mask_hog=None, title=None, save_name=None):
    num_classes = 8
    colors = ['#000000', '#666666', '#d22d04', '#ff6bfd', '#0575e6', "#994200", "#1a8f00", "#ffd724"]
    descriptions = ['IGNORUJ', 'Pozadí', 'Budova', 'Silnice', 'Voda', 'Pustina', 'Les', 'Agrikultura']
    hog = False

    myCmap = LinearSegmentedColormap.from_list('myCmap', colors, N=num_classes)
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7]

    if im_i is not None:
        # fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 4))
        if mask_hog is not None:
            fig, (ax0, ax1, ax_h, ax2) = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1, 1.080]})#, 'height_ratios': 1})
            hog = True
            ax_h.axis('off')
            ax_h.imshow(mask_hog, cmap=myCmap, vmin=0, vmax=num_classes - 1)
            ax_h.set_title('Maska pro HOG', fontdict={'fontsize': 20})
        else:
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 6.1), gridspec_kw={'width_ratios': [1, 1, 1.080]})#, 'height_ratios': 1})

        ax0.axis('off')
        ax0.imshow(im_i, cmap=plt.cm.gray)
        ax0.set_title('Výchozí snímek', fontdict={'fontsize': 20})
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    if title is not None:
        fig.suptitle(title, fontsize=22, )
    else:
        fig.suptitle('Výsledek metody LBP', fontsize=20, y=0.98)


    ax1.axis('off')
    ax1.imshow(smooth_mask(im_m), cmap=myCmap, vmin=0, vmax=num_classes - 1)
    ax1.set_title('Originální maska snímku', fontdict={'fontsize': 20})

    ax2.axis('off')
    im = ax2.imshow(y_pred_img, cmap=myCmap, vmin=0, vmax=num_classes - 1)
    ax2.set_title('Predikovaná maska', fontdict={'fontsize': 20})

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cb = fig.colorbar(im, cax=cax, ticks=class_labels)

    # navíc
    ticks = (np.arange(len(descriptions)) + 0.5) / 1.14
    cb.set_ticks(ticks)
    cb.set_ticklabels(descriptions, fontdict={'fontsize': 18})
    cb.ax.tick_params(labelsize=16) #12

    cb.ax.set_title('Třídy', fontsize=18)

    plt.margins(x=0, y=0)

    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


from sklearn.metrics import confusion_matrix
import numpy as np

import numpy as np

def compute_iou(y_pred, mask, num_classes=7):
    # toInt
    y_pred = y_pred.astype(int)
    mask = mask.astype(int)

    # Ignore 0
    mask[mask == 0] = num_classes + 1

    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for c in range(1, num_classes + 1):
        y_pred_c = y_pred == c
        mask_c = mask == c
        intersection[c-1] = np.sum(np.logical_and(y_pred_c, mask_c))
        union[c-1] = np.sum(np.logical_or(y_pred_c, mask_c))

    # iou
    iou_score = np.sum(intersection[1:]) / np.sum(union[1:])

    return iou_score