from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.io import read_image
import torch
import torchvision.transforms as transforms


input_shape = (1024, 1024)
output_shape = (520, 520)

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __getitem__(self, index):
        # Load the input image and mask
        image_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])
        image = read_image(image_path)
        mask = Image.open(mask_path)

        # Divide the image and mask into four 512x512 sections and pad them
        image_sections = []
        mask_sections = []
        for i in range(0, input_shape[0], 512):
            for j in range(0, input_shape[1], 512):
                section_image = image[:, i:i + 512, j:j + 512]
                section_mask = mask.crop((j, i, j + 512, i + 512))
                pad = (0, 0, 512 - section_mask.size[0], 512 - section_mask.size[1])
                section_mask = transforms.functional.pad(section_mask, pad)
                image_sections.append(section_image)
                mask_sections.append(section_mask)

        # Apply the transform to the input image and mask
        image_sections = torch.stack([self.transform(section_image) for section_image in image_sections])
        mask_sections = torch.stack(
            [torch.unsqueeze(transforms.functional.to_tensor(section_mask), dim=0) for section_mask in mask_sections])

        return image_sections, mask_sections

    def __len__(self):
        return len(self.image_files)
