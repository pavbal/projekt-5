import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CustomDeepLabV3(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        resnet = models.resnet50(pretrained=True, progress=True)
        resnet_layers = list(resnet.children())
        self.layer1 = nn.Sequential(*resnet_layers[:5])  # downsample 2
        self.layer2 = nn.Sequential(*resnet_layers[5])  # downsample 4
        self.layer3 = nn.Sequential(*resnet_layers[6])  # downsample 8
        self.layer4 = nn.Sequential(*resnet_layers[7])  # downsample 16
        self.layer5 = nn.Sequential(*resnet_layers[8])  # downsample 32
        self.aspp = ASPP(2048, 256, rates=[6, 12, 18], dropout=0.5)
        self.decoder = Decoder(256, 256, n_classes)

    def forward(self, x):
        # Encode
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # Atrous Spatial Pyramid Pooling
        x = self.aspp(x)
        # Decode
        x = self.decoder(x)
        # Upsample to original image size
        x = F.interpolate(x, size=x.size()[2:],
                          mode='bilinear', align_corners=False)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates, dropout=0.1):
        super().__init__()
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate))
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        atrous_outs = []
        for atrous_conv in self.atrous_convs:
            atrous_out = atrous_conv(x)
            atrous_outs.append(atrous_out)
        concat_out = torch.cat([conv1x1] + atrous_outs, dim=1)
        out = self.bn(concat_out)
        out = self.dropout(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x
