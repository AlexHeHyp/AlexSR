import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict

import torch

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.logger import Logger, PrintLogger

import numpy as np
import cv2

def main():
    command_mode = True
    if command_mode == True:
        # options
        print('\n********** config option outside and run python command **********\n')
        parser = argparse.ArgumentParser()
        parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
        opt = option.parse(parser.parse_args().opt, is_train=True)
    else:  # run for python file
        # config json in code for debug
        print('\n********** config option in code and debug/run python file **********\n')
        opt_cfg_json = '/home/heyp/code/BasicDenoise/codes/options/train/DnCNN_Paper.json'
        opt = option.parse(opt_cfg_json, is_train=True)

  #  # options
  #  parser = argparse.ArgumentParser()
  #  parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
  #  opt = option.parse(parser.parse_args().opt, is_train=True)

    util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old experiments if exists
    util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and \
        not key == 'pretrain_model_G' and not key == 'pretrain_model_D'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # print to file and std_out simultaneously
    sys.stdout = PrintLogger(opt['path']['log'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epoches = int(math.ceil(total_iters / train_size))
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_dataset_opt = dataset_opt
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
    assert train_loader is not None

    # Create model
    model = create_model(opt)
    # create logger
    logger = Logger(opt)

    current_step = 0
    start_time = time.time()


    print('---------- Start training -------------')
    for epoch in range(total_epoches):
        for i, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break

            #print('epoch, i:', epoch, i)
            #continue  #

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            time_elapsed = time.time() - start_time
            start_time = time.time()

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                for k, v in logs.items():
                    print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt)

            # save models
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                print('Saving the model at the end of iter %d' % (current_step))
                model.save(current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                print('---------- validation -------------')
                start_time = time.time()

                avg_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img_np(visuals['SR'])  # uint8
                    gt_img = util.tensor2img_np(visuals['HR'])  # uint8

                    #for residual mode (model's result is sr_img, but it is residual image,
                    #    so the real sr image need lr image + residual image)
                    if val_dataset_opt['generate_residual'] is not None:
                        sr_img_f = util.tensor2img_np(visuals['SR'], np.float)  # float
                        lr_img_f = util.tensor2img_np(visuals['LR'], np.float)  # float
                        sr_img_f = lr_img_f + sr_img_f
                        sr_img = ((sr_img_f * 255.0).round()).astype(np.uint8)
                     #   print('show sr image')
                     #   cv2.imshow('sr_img', sr_img)
                     #   cv2.waitKey(200)


                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir, '%s_%s.png' % (img_name, current_step))
                    #print('save image as:', save_img_path)
                    util.save_img_np(sr_img.squeeze(), save_img_path)

                    # calculate PSNR
                    crop_size = opt['scale'] + 2
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.psnr(cropped_sr_img, cropped_gt_img)

                avg_psnr = avg_psnr / idx
                time_elapsed = time.time() - start_time
                # Save to log
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = current_step
                print_rlt['time'] = time_elapsed
                print_rlt['psnr'] = avg_psnr
                logger.print_format_results('val', print_rlt)
                print('-----------------------------------')

            # update learning rate
            model.update_learning_rate()

    print('Saving the final model.')
    model.save('latest')
    print('End of Training \t Time taken: %d sec' % (time.time() - start_time))


if __name__ == '__main__':
    # # OpenCV get stuck in transform when used in DataLoader
    # # https://github.com/pytorch/pytorch/issues/1838
    # # However, cause problem reading lmdb
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()
