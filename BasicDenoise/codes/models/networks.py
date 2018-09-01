import functools
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch

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

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # print('initializing [%s] ...' % classname)
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant(m.bias.data, 0.0)

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

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
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


    # if which_model != 'sr_resnet':  # need to investigate, the original is better?
    #     init_weights(netG, init_type='orthogonal')
    print("gpu_ids = ", gpu_ids)
    if gpu_ids:
        cuda_a = torch.cuda.is_available()
        print("cuda_a = ", cuda_a)

        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
