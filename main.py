import torch
import argparse
import os
from model20221008 import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
from math import log10


# Training settings
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
    parser.add_argument('--model_path', type=str, default='log/AllFocusNet_17x7_4xdeblur_SRxdcn1_cbam_1510.0002.pth.tar')
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--reduction', type=int, default=16)

    return parser.parse_args()


def train(train_loader, cfg):
    net = AllFocusNet(cfg).to(cfg.device)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
    cudnn.benchmark = True

    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.load_model))
    else:
        cfg.start_epoch = 1

    SAI_loss = SAI_loss_cal(cfg.angRes).to(cfg.device)
    MacPI_loss = CharbonnierLoss().to(cfg.device)
    Optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': cfg.lr}], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=cfg.epochs,
                                                           eta_min=1e-6, last_epoch=cfg.start_epoch - 1)
    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    for idx_epoch in range(cfg.start_epoch - 1, cfg.epochs):
        for idx_iter, (data, label) in enumerate(train_loader):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out = net(data)
            loss_sai = SAI_loss(out, label)
            label_mac = SAI2MacPI(label, cfg.patch_size, cfg.patch_size)
            out_mac = SAI2MacPI(out, cfg.patch_size, cfg.patch_size)
            loss_macpi = MacPI_loss(out_mac, label_mac)
            loss =  loss_macpi + 0.1 * loss_sai
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

            psnr_epoch.append(cal_psnr(out.data.cpu(), label.data.cpu()))
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print('Epoch----%5d, loss---%f, PSNR---%f' % (
                idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))

        save_ckpt({
            'epoch': idx_epoch + 1,
            'state_dict': net.state_dict(),
            'loss': loss_list,
            'psnr': psnr_list,
        }, save_path='./log/', filename='AllFocusNet_2' + str(cfg.angRes) + 'x' + str(cfg.angRes) + '_' +
                                        str(cfg.upscale_factor) + 'xdeblur' + str(idx_epoch + 1) + str(
            cfg.lr) + '.pth.tar')
        psnr_epoch = []
        loss_epoch = []

    scheduler.step()


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)


def cal_psnr(img1, img2):
    _, _, h, w = img1.size()
    mse = torch.sum((img1 - img2) ** 2) / img1.numel()
    psnr = 10 * log10(1 / mse)
    return psnr


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def main(cfg):
    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
