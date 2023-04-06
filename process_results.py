import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import csv
import glob
import rawpy

from PyTorch_helpers import print_progress

# define dataset you are going to use; 'test' or 'validate'
dataset = 'test'
if dataset == 'test':
    num = '1'
else:
    num = '2'

# define directories
result_dir = './PyTorch/%s_result_Sony/' % dataset
gt_dir = './dataset/Sony/long/'


# get  IDs
test_fns = glob.glob(result_dir + '%s*.png' % num)
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]
total = len(test_ids)
test_ids = list(set(test_ids))


# set up .csv file
headers = ['image_id', 'psnr', 'ssim']

with open(result_dir + 'results.csv', mode='w', newline='') as results_file:
    writer = csv.writer(results_file)
    writer.writerow(headers)

    count = 0
    for test_id in test_ids:
        in_files = glob.glob(result_dir + '%05d*.png' % test_id)
        gt_file = glob.glob(gt_dir + '%05d*.ARW' % test_id)[0]
        with rawpy.imread(gt_file) as raw:
            gt_img = raw.postprocess()

        sum_psnr = 0
        sum_ssim = 0

        for k in range(len(in_files)):
            count += 1
            in_path = in_files[k]
            in_img = cv2.imread(in_path)

            print_progress(count, total, prefix='Processing images:', suffix='Complete')

            psnr_img = psnr(gt_img, in_img)
            ssim_img = ssim(gt_img, in_img, win_size=7, channel_axis=2)

            # print(in_path, psnr_img, ssim_img)
            sum_psnr += psnr_img
            sum_ssim += ssim_img

        psnr_final = sum_psnr / len(in_files)
        ssim_final = sum_ssim / len(in_files)

        row = [test_id, psnr_final, ssim_final]
        writer.writerow(row)










