import os
from torch.utils.data.dataset import Dataset

import random
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import ToTensor


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            data = np.transpose(data, (1, 0))
            label = np.array(hf.get('label'))
            label = np.transpose(label, (1, 0))
            data, label = augmentation(data.copy() / 255, label.copy() / 255)
        return ToTensor()(data.copy()), ToTensor()(label.copy())

    def __len__(self):
        return self.item_num


def augmentation(data, label):
    if random.random() < 0.5:
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data_hz = np.array(hf.get('data_hz'))
        data_vt = np.array(hf.get('data_vt'))
        data_rf = np.array(hf.get('data_rf'))
        label = np.array(hf.get('label'))
        return data_hz, data_vt, data_rf, label


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class SAI_loss_cal(nn.Module):
    def __init__(self, angRes):
        super(SAI_loss_cal, self).__init__()
        self.angRes = angRes
        if angRes == 5:
            k = torch.Tensor([[.05, .25, .4, .25, .05]])
        if angRes == 7:
            k = torch.Tensor([[.0044, .054, .242, .3991, .2420, .054, .0044]])
        if angRes == 9:
            k = torch.Tensor([[.0001, .0044, .054, .242, .3989, .2420, .054, .0044, .0001]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        # self.tv_cal = TV_Cal()
        # self.ssim_cal = pytorch_ssim.SSIM(window_size=angRes)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

