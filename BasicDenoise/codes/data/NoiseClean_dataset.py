import os.path
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import data.GlobalVar as GV
import skimage

class NoiseCleanDataset(data.Dataset):
    '''
    Read LR and HR image pair.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(NoiseCleanDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        self.bGenerateResidual = False
        if opt['generate_residual'] is not None:
            self.bGenerateResidual = opt['generate_residual']

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else: # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1]
        #print('~~~~~~init NoiseCleanDataset~~~~~~')

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        sigma = self.opt['sigma']
        HR_size = self.opt['HR_size']
        bGenerateResidual = self.bGenerateResidual;
        #print('~~~~~~getitem index~~~~~~', index)

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)
        # modcrop in validation phase
        if self.opt['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(self.LR_env, LR_path)
        else:  # down-sampling or/and add noise on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_HR.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, HR_size)
                W_s = _mod(W_s, random_scale, scale, HR_size)
                if random_scale == 1:
                    # crop the HR image to want mod size(add by Alex He)
                    img_HR = img_HR[:H_s, :W_s]
                else:
                    #resize the HR image to want scale size
                    img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)

                # force to 3 channels
                # if img_HR.ndim == 2:
                #    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_HR.shape

            # !! just add gaussian noise
            if sigma is not None and sigma > 0:
                seedV = random.randint(1, 10000)
                img_LR = skimage.util.random_noise(img_HR, mode='gaussian', seed=seedV, clip=True, mean=0, var=sigma)
            else:
                # using matlab imresize
                img_LR = util.imresize_np(img_HR, 1 / scale, True)

            #if img_LR.ndim == 2:
            #    img_LR = np.expand_dims(img_LR, axis=2)
            #    img_HR = np.expand_dims(img_HR, axis=2)

        #print('~~~~~~aft add noise index~~~~~~', index)
        if True == self.opt['show_dataset_img']:
            bShow_train = GV.get_t_value()
            if self.opt['phase'] == 'train' and False==bShow_train and 0==index%1000:
                GV.set_t_value(True)
                cv2.namedWindow('im_lr_t', 0)
                cv2.imshow('im_lr_t', img_LR)
                cv2.moveWindow('im_lr_t', 800, 40)
                cv2.waitKey(3*1000)
                cv2.destroyWindow('im_lr_t')

            bShow_val = GV.get_v_value()
            if self.opt['phase'] != 'train' and False==bShow_val and 0==index:
                GV.set_v_value(True)
                cv2.namedWindow('im_lr_v', 0)
                cv2.imshow('im_lr_v', img_LR)
                cv2.moveWindow('im_lr_v', 1200, 40)
                cv2.waitKey(3*1000)
                cv2.destroyWindow('im_lr_v')

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(
                    np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_HR, 1 / scale, True)
                #if img_LR.ndim == 2:
                #    img_LR = np.expand_dims(img_LR, axis=2)
            H, W, C = img_LR.shape

            LR_size = HR_size // scale
            # print("LR_size: ", LR_size) #!!

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

            # augmentation - flip, rotate
            img_LR, img_HR = util.augment([img_LR, img_HR], self.opt['use_flip'], \
                self.opt['use_rot'])

        # channel conversion
        # if self.opt['color']:
        #    img_LR, img_HR = util.channel_convert(C, self.opt['color'], [img_LR, img_HR])

        # BGR to RGB, HWC to CHW, numpy to tensor
        # print('~~~~~~~~img_HR/LR.shape[2]:', img_HR.shape[2], img_LR.shape[2])
        #if img_HR.shape[2] == 3:
        #    img_HR = img_HR[:, :, [2, 1, 0]]
        #    img_LR = img_LR[:, :, [2, 1, 0]]
        #img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        #img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR))).float()
        #print('~~~~~~~~img_HR/LR.shape[2]:', img_HR.shape[2], img_LR.shape[2])

        # !! generate residual image (put here for float type)
        if bGenerateResidual == True:
            img_HR = img_HR - img_LR

        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)
