import os
import os.path
from multiprocessing import Pool
import time
import numpy as np
import cv2
import random
import skimage

def main():
    noise_level = 20   #[0, 255]

    src_dir = '/home/heyp/data/Train400'

    #need create at first(before run)
    dst_dir = '/home/heyp/data/DnCNN/Train/Train400_p40_s4'

    n_thread = 8

    print('Parent process %s.' % os.getpid())
    start = time.time()

    p = Pool(n_thread)
    # read all files to a list
    all_files = []
    for root, _, fnames in sorted(os.walk(GT_dir)):
        full_path = [os.path.join(root, x) for x in fnames]
        all_files.extend(full_path)

    # cut into subtasks
    def chunkify(lst, n):  # for non-continuous chunks
        return [lst[i::n] for i in range(n)]

    sub_lists = chunkify(all_files, n_thread)

    # call workers
    for i in range(n_thread):
        p.apply_async(worker, args=(sub_lists[i], noise_level, dst_dir))

    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    end = time.time()
    print('All subprocesses done. Using time {} sec.'.format(end - start))


def worker(src_paths, noise_s, noise_e, dst_dir):
    thres_sz = 8

    for src_path in src_paths:
        base_name = os.path.basename(src_path)
        print(base_name, os.getpid())
        img_src = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)

        n_channels = len(img_src.shape)
        if n_channels == 2:
            h, w = img_src.shape
        elif n_channels == 3:
            h, w, c = img_src.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))

        img_train = img_src

        # seedV = random.randint(1, 10000)
        # img_LR = skimage.util.random_noise(img_HR, mode='gaussian', seed=seedV, clip=True, mean=0, var=sigma)

        # create noise images
        noiseL_B = [noise_s, noise_e]  # ingnored when opt.mode=='S'
        stdN = noise_s
        if noise_s != noise_e:
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1])

        noise_sigma = stdN/255.
        noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=noise_sigma)


        imgn_train = img_train + noise


        noise_str = '{:03d}'.format(noise_level)
        cv2.imwrite(os.path.join(dst_dir, base_name.replace('.png', \
            '_s'+noise_str+'.png')), noise_add_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    main()


#noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
