import functools
import torch
import torch.nn as nn
from torch.nn import init
import math

import models.modules.architecture as arch
import models.modules.sft_arch as sft_arch

####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    # print('initializing [%s] ...' % classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, std)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1, bGrayDnCNN=False):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # print('initializing [%s] ...' % classname)
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        if False == bGrayDnCNN:
            #Xin Tao code
            init.constant_(m.weight.data, 1.0)
        else:
            #DnCnn-PyTorch SaoYan code (very important for training)
            m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print('initializing [%s] ...' % classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02, bGrayDnCNN=False):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming,
                                                  scale=scale, bGrayDnCNN=bGrayDnCNN)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


####################
# define network
####################

# Generator
def define_G(opt):
    gpu_ids = opt['gpu_ids']
    opt = opt['network_G']
    which_model = opt['which_model_G']

    if which_model == 'sr_resnet':
        netG = arch.SRResNet(in_nc=opt['in_nc'], out_nc=opt['out_nc'], nf=opt['nf'], \
            nb=opt['nb'], upscale=opt['scale'], norm_type=opt['norm_type'], mode=opt['mode'],\
            upsample_mode='pixelshuffle')

    elif which_model == 'dn_cnn':
        netG = arch.DnCNN(in_nc=opt['in_nc'], out_nc=opt['out_nc'], nf=opt['nf'], \
                          nb=opt['nb'], norm_type=opt['norm_type'])

    elif which_model == 'denoise_resnet':
        # add code here ...
        netG = arch.DenoiseResNet(in_nc=opt['in_nc'], out_nc=opt['out_nc'], nf=opt['nf'], \
            nb=opt['nb'], upscale=opt['scale'], norm_type=opt['norm_type'], mode=opt['mode'], \
            upsample_mode='pixelshuffle')

    elif which_model == 'sft_arch':  # SFT-GAN
        netG = sft_arch.SFT_Net()

    elif which_model == 'RRDB_net':  # RRDB
        netG = arch.RRDBNet(in_nc=opt['in_nc'], out_nc=opt['out_nc'], nf=opt['nf'], \
                            nb=opt['nb'], gc=opt['gc'], upscale=opt['scale'], norm_type=opt['norm_type'], \
                            act_type='leakyrelu', mode=opt['mode'], upsample_mode='upconv')

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    if opt['is_train']:
        if which_model == 'dn_cnn' and opt['in_nc'] == 1:
            #init_weights(netG, init_type='kaiming', scale=1, bGrayDnCNN=True)
            init_weights(netG, init_type='kaiming', scale=0.1, bGrayDnCNN=False)
        else:
            init_weights(netG, init_type='kaiming', scale=0.1, bGrayDnCNN=False)

    print("gpu_ids = ", gpu_ids)
    if gpu_ids:
        cuda_a = torch.cuda.is_available()
        print("cuda_a = ", cuda_a)

        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

# Discriminator
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt = opt['network_D']
    which_model = opt['which_model_D']

    if which_model == 'discriminaotr_vgg_128':  #128 is w*h
        netD = arch.Discriminaotr_VGG_128(in_nc=opt['in_nc'], base_nf=opt['nf'], \
            norm_type=opt['norm_type'], mode=opt['mode'], act_type=opt['act_type'])

    elif which_model == 'dis_acd':
        netD = sft_arch.ACD_VGG_BN_96()
    else:
        raise NotImplementedError('Discriminator model [%s] is not recognized' % which_model)

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD

# Feature extractor
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF
