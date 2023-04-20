import os
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# base_path = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA_Train_16/Rural/"
# img_dir = base_path + "images_png/"
# mask_dir = base_path + "masks_png/"
IMG_LEN = 1024
input_shape = (1024, 1024)
output_shape = (520, 520)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.img_dir_list_sorted = sorted(os.listdir(self.img_dir), key=len)
        self.mask_dir_list_sorted = sorted(os.listdir(self.mask_dir), key=len)
        # self.image_files = sorted(os.listdir(img_dir))
        # self.mask_files = sorted(os.listdir(mask_dir))
        self.len = len(os.listdir(self.img_dir))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_dir_list_sorted[idx]
        mask_path = self.mask_dir + self.mask_dir_list_sorted[idx]
        image = read_image(img_path)
        # image = image / 255

        mask_read = read_image(mask_path)
        mask = torch.from_numpy(np.zeros((7, IMG_LEN, IMG_LEN)))
        for i in range(1, 8):
            mask[i - 1, :, :] = (mask_read == i)
            # if i == 1:
            #     print(mask)

        # image = image.numpy()

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.transform(mask)

        return image, mask


    # def __getitem__(self, index):
    #     # Load the input image and mask
    #     image_path = os.path.join(self.img_dir, self.img_dir_list_sorted[index])
    #     mask_path = os.path.join(self.mask_dir, self.mask_dir_list_sorted[index])
    #     image = read_image(image_path)
    #     mask = Image.open(mask_path)
    #
    #     # Divide the image and mask into four 512x512 sections and pad them
    #     image_sections = []
    #     mask_sections = []
    #     for i in range(0, input_shape[0], 512):
    #         for j in range(0, input_shape[1], 512):
    #             section_image = image[:, i:i + 512, j:j + 512]
    #             section_mask = mask.crop((j, i, j + 512, i + 512))
    #             pad = (0, 0, 512 - section_mask.size[0], 512 - section_mask.size[1])
    #             section_mask = transforms.functional.pad(section_mask, pad)
    #             image_sections.append(section_image)
    #             mask_sections.append(section_mask)
    #
    #     # Apply the transform to the input image and mask
    #     image_sections = torch.stack([self.transform(section_image) for section_image in image_sections])
    #     mask_sections = torch.stack(
    #         [torch.unsqueeze(transforms.functional.to_tensor(section_mask), dim=0) for section_mask in mask_sections])
    #
    #     return image_sections, mask_sections


