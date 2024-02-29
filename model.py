import torch
import torch.nn as nn
import torch.nn.init as init
import math
from dcn import DeformableConv2d


class AllFocusNet(nn.Module):
    def __init__(self, cfg):
        super(AllFocusNet, self).__init__()
        self.inform_blocks = cfg.inform_blocks
        self.recon_blocks = cfg.recon_blocks
        self.angRes = cfg.angRes
        self.channels = cfg.channels
        self.channels_dcn = cfg.channels_dcn
        self.scale = cfg.upscale_factor
        channels = cfg.channels
        angRes = cfg.angRes
        channels_dcn = cfg.channels_dcn
        res_block = cfg.res_blocks
        reduction = cfg.reduction
        inform_blocks = cfg.inform_blocks
        scale = cfg.upscale_factor
        residual_block = cfg.residual_blocks
        # shallow feature extraction
        self.fe = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False)
        # feature extraction
        self.inform = Information_Fuse_Block(channels, channels_dcn, angRes, residual_block, reduction)
        # feature fusion
        self.agg = nn.Conv2d(inform_blocks * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        # deblur-SR
        self.deblur = Deblur(scale, res_block, channels, reduction)
        # Final convolution
        self.conv_last = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_img):
        fea = self.fe(x_img)
        out_inform = []
        buffer_inform = fea
        for i in range(self.inform_blocks):
            buffer_inform = self.inform(buffer_inform)
            out_inform.append(buffer_inform)
        buffer_inform_concat = torch.cat(out_inform, 1)
        buffer_agg = self.agg(buffer_inform_concat)
        buffer_mac2pi = MacPI2SAI(buffer_agg, self.angRes)
        buffer_deblur = self.deblur(buffer_mac2pi)
        out = self.conv_last(buffer_deblur)
        return out


class Information_Fuse_Block(nn.Module):
    def __init__(self, channels, channels_dcn, angRes, residual_block, reduction):
        super(Information_Fuse_Block, self).__init__()
        self.dconv_1 = DeformableConv2d(channels, channels_dcn, case=1, angRes=angRes, kernel_size=3, stride=1,
                                        padding=1, bias=False)
        self.dconv_2 = DeformableConv2d(channels, channels_dcn, case=2, angRes=angRes, kernel_size=3, stride=1,
                                        padding=int(angRes + 1), bias=False)
        self.dconv_3 = DeformableConv2d(channels, channels_dcn, case=3, angRes=angRes, kernel_size=3, stride=1,
                                        padding=int(angRes - 1), bias=False)
        self.dconv_4 = DeformableConv2d(channels, channels_dcn, case=4, angRes=angRes, kernel_size=3, stride=1,
                                        padding=int(2 * (angRes + 1)), bias=False)
        self.dconv_5 = DeformableConv2d(channels, channels_dcn, case=5, angRes=angRes, kernel_size=3, stride=1,
                                        padding=int(2 * (angRes - 1)), bias=False)
        self.att_block = Attension_Block(residual_block, channels_dcn, reduction)
        self.fe_concat = nn.Conv2d(5 * channels_dcn, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.Relu = nn.PReLU()

    def forward(self, x):
        buffer_f1 = self.dconv_1(x)
        buffer_f1 = self.att_block(buffer_f1)
        buffer_f2 = self.dconv_2(x)
        buffer_f2 = self.att_block(buffer_f2)
        buffer_f3 = self.dconv_3(x)
        buffer_f3 = self.att_block(buffer_f3)
        buffer_f4 = self.dconv_4(x)
        buffer_f4 = self.att_block(buffer_f4)
        buffer_f5 = self.dconv_5(x)
        buffer_f5 = self.att_block(buffer_f5)
        buffer_concat = torch.cat([buffer_f1, buffer_f2, buffer_f3, buffer_f4, buffer_f5], dim=1)
        buffer_fea_concat = self.fe_concat(buffer_concat)
        out = self.Relu(buffer_fea_concat)
        out += x
        return out


class Up_sample(nn.Module):
    def __init__(self, scale, channels, angRes):
        super(Up_sample, self).__init__()
        self.PreConv = nn.Conv2d(channels, channels * scale ** 2, kernel_size=3, stride=1,
                                 dilation=int(angRes), padding=int(angRes), bias=False)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.angRes = angRes

    def forward(self, x):
        buffer = self.PreConv(x)
        bufferSAI_LR = MacPI2SAI(buffer, self.angRes)
        out = self.pixel_shuffle(bufferSAI_LR)
        return out


class Deblur(nn.Module):
    def __init__(self, scale, res_block, channels, reduction):
        super(Deblur, self).__init__()
        self.att = Attension_Block(res_block, channels, reduction)
        self.PreConv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1,
                                 dilation=1, padding=1, bias=False)
        self.PreConv1 = nn.Conv2d(channels, channels * 16, kernel_size=3, stride=1,
                                  dilation=1, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.pixel_shuffle1 = nn.PixelShuffle(4)
        self.agg = nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False)
        self.scale = scale

    def forward(self, x):
        input = x
        buffer_2o = self.PreConv(input)
        buffer_2o = self.pixel_shuffle(buffer_2o)
        buffer_3o = self.PreConv1(input)
        buffer_3o = self.pixel_shuffle1(buffer_3o)
        buffer_1 = self.att(input)
        buffer_1 = self.PreConv(buffer_1)
        buffer_1u = self.pixel_shuffle(buffer_1)
        buffer_2 = torch.cat([buffer_1u, buffer_2o], dim=1)
        buffer_2 = self.agg(buffer_2)
        buffer_2 = self.att(buffer_2)
        buffer_2 = self.PreConv(buffer_2)
        buffer_2u = self.pixel_shuffle(buffer_2)
        buffer_3 = torch.cat([buffer_2u, buffer_3o], dim=1)
        buffer_3 = self.agg(buffer_3)
        out = self.att(buffer_3)
        return out


class Attension_Block(nn.Module):
    def __init__(self, residual_block, channels, reduction):
        super(Attension_Block, self).__init__()
        self.cbam = CBAM(channels)
        body = []
        for i in range(residual_block):
            body.append(ResidualBlock(channels))
            body.append(CBAM(channels))
        self.body = nn.Sequential(*body)
        self.Relu = nn.ReLU()

    def forward(self, x):
        out = self.body(x)
        out += x
        return out


class CALayer(nn.Module):
    def __init__(self, channels, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=bias),
            nn.PReLU()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        # channel attention
        self.fe_cab = Channel_Attention_Block(channels)
        # spatial attention
        self.fe_sab = Spatial_Attention_Block()

    def forward(self, fea_concat):
        out = self.fe_cab(fea_concat) * fea_concat
        out = self.fe_sab(out) * out
        return out


class Channel_Attention_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(Channel_Attention_Block, self).__init__()
        mid_channel = channels // reduction
        self.avg_pool = nn.AvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            # nn.Linear(in_features=channels, out_features=mid_channel),
            nn.Conv2d(channels, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
            # nn.Linear(in_features=mid_channel, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea_concat):
        avg_out = self.shared_MLP(self.avg_pool(fea_concat))
        max_out = self.shared_MLP(self.max_pool(fea_concat))
        return self.sigmoid(avg_out + max_out)


class Spatial_Attention_Block(nn.Module):
    def __init__(self):
        super(Spatial_Attention_Block, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea_rsab):
        avg_out = torch.mean(fea_rsab, dim=1, keepdim=True)
        max_out, _ = torch.max(fea_rsab, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
