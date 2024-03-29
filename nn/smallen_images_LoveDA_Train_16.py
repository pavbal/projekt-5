import os
import numpy as np

from PIL import Image

base_path = "C:/Users/pavba/PycharmProjects/projekt-5/LoveDA_Test_16/Rural/"
img_dir = base_path + "images_png/"
mask_dir = base_path + "masks_png/"

img_dir_new = base_path + "images_png_512/"
mask_dir_new = base_path + "masks_png_512/"


img_dir_list_sorted = sorted(os.listdir(img_dir), key=len)
mask_dir_list_sorted = sorted(os.listdir(mask_dir), key=len)

counter = 16

for img_ in img_dir_list_sorted:
    img = np.array(Image.open(img_dir+img_))
    mask = np.array(Image.open(mask_dir+img_))
    for i in [0, 512]:
        for j in [0, 512]:
            img_quarter = img[i:i+512, j:j+512]
            mask_quarter = mask[i:i+512, j:j+512]
            im = Image.fromarray(img_quarter)
            im.save(img_dir_new+str(counter)+".png")
            ma = Image.fromarray(mask_quarter)
            ma.save(mask_dir_new+str(counter) + ".png")
            counter = counter+1
            if (i==0 and j==0):
                width1, height2 = im.size
                width2, height2 = ma.size

                print("mask: " + str(width2))
                print("img: " + str(width1))

# im = Image.fromarray(A)
# im.save("your_file.jpeg")