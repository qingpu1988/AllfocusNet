import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


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


class MacPI_loss_cal(nn.Module):
    def __init__(self, angRes):
        super(MacPI_loss_cal, self).__init__()
        self.angRes = angRes
        if angRes == 5:
            k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

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

