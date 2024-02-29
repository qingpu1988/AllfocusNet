import torch
import os
import scipy.io
from model20221008 import AllFocusNet
import argparse
import numpy as np
import scipy.misc
import h5py
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import math
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=7, help="angular resolution")
    parser.add_argument("--inform_blocks", type=int, default=4, help="number of Inter-Groups")
    parser.add_argument("--recon_blocks", type=int, default=4, help="number of Inter-Blocks")
    parser.add_argument("--residual_blocks", type=int, default=4, help="number of Inter-Blocks")
    parser.add_argument("--res_blocks", type=int, default=4, help="number of Inter-Blocks")
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--channels_dcn', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=3000, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--trainset_dir', type=str, default='E:/LF-Dataset/Trainingdata_7x7_4xDeblurSR')
    parser.add_argument('--model_name', type=str, default='AllFocusNet_7x7_4xdeblurSR_epoch20.5.pth.tar')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str,default='log/AllFocusNet.pth.tar')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--reduction', type=int, default=16)

    return parser.parse_args()


def main(cfg):
    device = cfg.device
    crop = 8
    scale = cfg.upsacle_factor
    dir_images = 'E:/LF_set/Test_h5py_7/'
    dir_save_path = 'E:/LF_set/result_final/'
    net = AllFocusNet(cfg).to(device)
    cudnn.benchmark = True
    model = torch.load(cfg.model_path, map_location={'cuda:0': 'cuda:0'})
    net.load_state_dict(model['state_dict'])

    for root, dirs, files in os.walk(dir_images):
        if len(files) == 0:
            break
        for file_name in files:
            file_path = [dir_images + file_name]
            with h5py.File(file_path[0], 'r') as hf:
                data = np.array(hf.get('data'))
                data = np.transpose(data, (1, 0))
                data = np.expand_dims(data, axis=0)
                data = np.expand_dims(data, axis=0)
                data = torch.from_numpy(data.copy())
                data = data.to(device)

            with torch.no_grad():
                if not crop:

                    out = net(data)
                    out = out.cpu().numpy()

                    scipy.io.savemat(dir_save_path + file_name[0:-3] + 'dcn1_beta0.1_ep35.mat', {'LF': out})
                else:
                    # time_start = time.time()
                    length = 64
                    time_start = time.time()
                    lr_l, lr_m, lr_r = CropPatches(data, length // scale, crop // scale, cfg.angRes)

                    out_l = net(lr_l).cpu().numpy()
                    out_m = np.zeros((lr_m.shape[0], 1, lr_m.shape[2] * scale,
                                      lr_m.shape[3] * scale), dtype=np.float32)
                    for i in range(lr_m.shape[0]):
                        out_m[i:i + 1] = net(lr_m[i:i + 1]).cpu().numpy()
                    out_r = net(lr_r).cpu().numpy()
                    time_els = time.time() - time_start
                    print(time_els)

                    out = MergePatches(out_l, out_m, out_r, data.shape[2] * scale, data.shape[3] * scale,
                                       length, crop, cfg.angRes)

                    scipy.io.savemat(
                        dir_save_path + file_name[0:-3] + 'interdcn_' + str(beta) + '_ep' + str(ep) + '.mat',
                        {'LF': out})


def MergePatches(left, middles, right, h, w, len, crop, ang):
    n, a, _, w_l = left.shape[0:4]
    temp = middles[0]
    _, _, w_m = temp.shape[0:4]
    _, _, _, w_r = right.shape[0:4]
    # out = torch.Tensor(n, a, h, w).to(left.device)
    out = np.zeros((n, a, h, w)).astype(left.dtype)
    for idx in range(ang):
        blx = idx * (w // ang)
        elx = blx + len
        bly = idx * (w_l // ang)
        ely = bly + len
        out[:, :, :, blx:elx] = left[:, :, :, bly:ely]
        for i in range(middles.shape[0]):
            bmx = len + i * len + idx * (w // ang)
            emx = bmx + len
            bmy = crop + idx * (w_m // ang)
            emy = bmy + len
            out[:, :, :, bmx:emx] = middles[i:i + 1, :, :, bmy:emy]
        num = middles.shape[0]
        brx = (num + 1) * len + idx * (w // ang)
        erx = (w // ang) * (idx + 1)
        bry = crop + idx * (w_r // ang)
        ery = (w_r // ang) * (idx + 1)
        out[:, :, :, brx:erx] = right[:, :, :, bry:ery]
    return out


def CropPatches(image, len, crop, ang):
    # left [1,an2,h,lw]
    # middles[n,an2,h,mw]
    # right [1,an2,h,rw]
    h, w = image.shape[2:4]
    left = image[:, :, :, 0:(len + crop) * ang]
    num = math.floor((w - (len + crop) * ang) / (len * ang))
    middles = torch.Tensor(num, 1, h, (len + crop * 2) * ang).to(image.device)
    for i in range(num):
        middles[i] = image[:, :, :, ((i + 1) * len - crop) * ang:((i + 2) * len + crop) * ang]
    right = image[:, :, :, ((num + 1) * len - crop) * ang:]
    return left, middles, right


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
