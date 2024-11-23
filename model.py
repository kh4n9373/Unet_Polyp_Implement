from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import cv2
import wandb
import argparse
import os
import random

import pandas as pd
import cv2
import numpy as np
import albumentations as A

def mask2rgb(mask):
    color_dict = {0: torch.tensor([0, 0, 0]),
                  1: torch.tensor([1, 0, 0]),
                  2: torch.tensor([0, 1, 0])}
    output = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3)).long()
    for i in range(mask.shape[0]):
        for k in color_dict.keys():
            output[i][mask[i].long() == k] = color_dict[k]
    return output.to(mask.device)
@torch.no_grad()
def dice_score(
    inputs: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    input_one_hot = mask2rgb(inputs.argmax(dim=1))
    target_one_hot = mask2rgb(targets)
    dims = (2, 3)
    intersection = torch.sum(input_one_hot * target_one_hot, dims)
    cardinality = torch.sum(input_one_hot + target_one_hot, dims)

    dice_score = (2. * intersection + 1e-6) / (cardinality + 1e-6)
    return dice_score.mean()


class DiceLoss(nn.Module):
    def __init__(self, weights=torch.Tensor([[0.4, 0.55, 0.05]])) -> None:
        super(DiceLoss, self).__init__()
        self.eps = 1e-6
        self.weights = weights

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor) -> torch.Tensor:
        input_soft = F.softmax(inputs, dim=1)

        target_one_hot = mask2rgb(targets)

        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)

        dice_score = torch.sum(
            dice_score * self.weights.to(dice_score.device),
            dim=1
        )
        return torch.mean(1. - dice_score)


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        input_soft = F.softmax(inputs, dim=1)
        target_one_hot = mask2rgb(targets)
        input_soft = input_soft.view(-1)
        target_one_hot = target_one_hot.view(-1)
        TP = (input_soft * target_one_hot).sum()
        FP = ((1-target_one_hot) * input_soft).sum()
        FN = (target_one_hot * (1-input_soft)).sum()

        tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        focal_tversky = (1 - tversky)**gamma

        return focal_tversky
def _make_layers(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x+x1)
        return x1


class R2Block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2, max_pool=True):
        super(R2Block, self).__init__()
        self.pool = max_pool
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.pool:
            x = self.max_pool(x)
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class Attention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            _make_layers(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention: bool = False,
        recurrent: bool = True
    ):
        super().__init__()
        self.attention = attention
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        if attention:
            self.attn = Attention(out_channels, out_channels, out_channels//2)
        if recurrent:
            self.conv = R2Block(in_channels, out_channels, max_pool=False)
        else:
            self.conv = _make_layers(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        if self.attention:
            x2 = self.attn(x1, x2)
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        attention: bool = True,
        recurrent: bool = True
    ):
        super().__init__()
        self.attention = attention

        if recurrent:
            self.conv_in = R2Block(in_channels, 64, max_pool=False)
            self.down1 = R2Block(64, 128)
            self.down2 = R2Block(128, 256)
            self.down3 = R2Block(256, 512)
            self.down4 = R2Block(512, 1024)
        else:
            self.conv_in = _make_layers(in_channels, 64)
            self.down1 = DownSample(64, 128)
            self.down2 = DownSample(128, 256)
            self.down3 = DownSample(256, 512)
            self.down4 = DownSample(512, 1024)

        self.up1 = UpSample(1024, 512, attention=attention, recurrent=recurrent)
        self.up2 = UpSample(512, 256, attention=attention, recurrent=recurrent)
        self.up3 = UpSample(256, 128, attention=attention, recurrent=recurrent)
        self.up4 = UpSample(128, 64, attention=attention, recurrent=recurrent)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_out(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class Resnet50Unet(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpsampleBlock(2048, 1024))
        up_blocks.append(UpsampleBlock(1024, 512))
        up_blocks.append(UpsampleBlock(512, 256))
        up_blocks.append(UpsampleBlock(in_channels=128 + 64, out_channels=128,
                                       up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpsampleBlock(in_channels=64 + 3, out_channels=64,
                                       up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools["layer_0"] = x
        x = self.input_block(x)
        pre_pools["layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Resnet50Unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Resnet50Unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
        
class BKAIPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, name: str = "resunet"):
        super().__init__()
        if name == "resunet":
            self.model = Resnet50Unet(n_classes=3)
        else:
            self.model = UNet(in_channels=3)
        self.lr = lr
        self.dice_loss = DiceLoss()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _forward(self, batch, batch_idx, name="train"):
        image, mask = batch['image'].float(), batch['mask'].long()
        logits = self(image)
        loss = self.entropy_loss(logits, mask)
        d_score = dice_score(logits, mask)
        acc = (logits.argmax(dim=1) == mask).float().mean()
        self.log_dict(
            {
                f"{name}_loss": loss,
                f"{name}_dice_score": d_score,
                f"{name}_acc": acc
            },
            on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=5,
            verbose=True,
            factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
        }