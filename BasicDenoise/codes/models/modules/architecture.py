import math
import torch
import torch.nn as nn
import torchvision
from . import block as B
#from . import spectral_norm as SN

####################
# Generator
####################


class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]  # ??

        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

class DnCNN(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, norm_type='batch', act_type='relu'):
        super(DnCNN, self).__init__()

        L_CA = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=act_type)

        L_CNA = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        CNA_blocks = [L_CNA for _ in range(nb)]

        L_Conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

        #self.model = B.sequential(L_CA, L_CNA, L_CNA, L_CNA, L_CNA, L_CNA, L_CNA, L_CNA, L_Conv)
        self.model = nn.Sequential(L_CA, *CNA_blocks, L_Conv)

    def forward(self, x):
        x = self.model(x)
        return x

class DnCNN_new(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, norm_type='batch', act_type='relu'):
        super(DnCNN, self).__init__()

        self.nb = nb
        self.norm_type = norm_type

        self.conv1 = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.ReLU = nn.ReLU(True)

        self.conv21 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.BatchNorm21 = nn.BatchNorm2d(nf, affine=True)
        self.InstanceNorm21 = nn.InstanceNorm2d(nf, affine=True)

        self.Conv3 = nn.Conv2d(nf, out_nc, 3, 1, 1)

        print('DnCNN__self.nb0: %d' % self.nb)

    def forward(self, x):
        out = self.ReLU(self.conv1(x))
        if self.norm_type == 'batch':
            for _ in range(self.nb):
                out = self.ReLU(self.BatchNorm21(self.conv12(out)))
        elif self.norm_type == 'instance':
            for _ in range(self.nb):
                out = self.ReLU(self.InstanceNorm21(self.conv12(out)))
        else:
            for _ in range(self.nb):
                out = self.ReLU(self.conv21(out))

        out = self.Conv3(out)
        return out

class minc(nn.Module):
    def __init__(self):
        super(minc, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)

        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


if __name__ == '__main__':
    net = minc()
    net.load_state_dict(torch.load('VGG16minc_53.pth'), strict=True)
    net.eval()
    net = net.cuda()

    import cv2
    import numpy as np
    img = cv2.imread('/mnt/SSD/xtwang/BasicSR_datasets/val_set5/Set5_bicLRx4/butterfly_bicLRx4.png',
                     cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    input = img.unsqueeze(0).cuda()
    out = net.forward(input)
    print(out.float())
    print(out.shape)