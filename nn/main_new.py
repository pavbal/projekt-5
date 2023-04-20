import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
import os
from PIL import Image
from CustomDataset import CustomDataset
from CustomDeepLabV3 import CustomDeepLabV3
import torch.optim as optim


# Define the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the parameters of the dataset and the network
input_shape = (1024, 1024)
output_shape = (520, 520)
batch_size = 4
num_epochs = 10
num_classes = 8  # 7 semantic classes + 1 ignore class

# Define the paths to the image and mask directories
image_dir = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA_Train_16/Rural/images_png"
mask_dir = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA_Train_16/Rural/masks_png"

# Define the transforms to apply to the input images and masks
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(output_shape),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_dataset = CustomDataset(image_dir=image_dir, mask_dir=mask_dir,
                              transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# weights = DeepLabV3_ResNet50_Weights#.DEFAULT
model = CustomDeepLabV3(n_classes=7)
# model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define the custom dataset class


