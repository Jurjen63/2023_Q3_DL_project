#!/usr/bin/python
# import packages
import os
import time
import numpy as np
import rawpy
import glob
import torch
from torch import optim as optim
from PIL import Image

from PyTorch_helpers import SeeInTheDarkModel, pack_raw

# set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# define directories
input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'

torch_dir = './Pytorch/'
result_dir = './PyTorch/result_Sony/'
model_dir = './PyTorch/model_Sony/'

# fail safes in case the directories do not exist
if not os.path.isdir(torch_dir):
    os.makedirs(torch_dir)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

# set up hyper parameters
patch_size = 512
save_frequency = 100
max_epoch = 4001
learning_rate = 1e-4

DEBUG = 0
if DEBUG == 1:
    save_frequency = 5
    max_epoch = 101
    train_ids = train_ids[:10]

# Raw data takes long to load. Keep them in memory after loaded [Original paper]
gt_images = [None] * 6000
input_images = {'300': [None] * len(train_ids), '250': [None] * len(train_ids), '100': [None] * len(train_ids)}

g_loss = np.zeros((5000, 1))

all_folders = glob.glob(result_dir + '*0')
last_epoch = 0

for folder in all_folders:
    last_epoch = np.maximum(last_epoch, int(folder[-4:]))

model = SeeInTheDarkModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(last_epoch, max_epoch):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    count = 0

    if epoch > (max_epoch/2):
        learning_rate = learning_rate/1e3

    # [code from paper]
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        # in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_path = in_files[np.random.randint(low=0, high=len(in_files))]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        count += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw, patch_size), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        _, H, W, _ = input_images[str(ratio)[0:3]][ind].shape

        xx = np.random.randint(0, W - patch_size)
        yy = np.random.randint(0, H - patch_size)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + patch_size, xx:xx + patch_size, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + patch_size * 2, xx * 2:xx * 2 + patch_size * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        # [end of paper code]

        # tensors are not supported with negative strides
        gt_patch = np.maximum(gt_patch, 0.0)

        input_img = torch.from_numpy(input_patch).permute(0, 3, 1, 2).to(device)
        gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2).to(device)

        #
        model.zero_grad()
        output = model(input_img)
        loss = torch.abs(output - gt_img).mean()
        loss.backward()
        optimizer.step()
        g_loss[ind] = loss.data.cpu()

        print("Epoch: %d, Count: %d,  Loss=%.3f, Time=%.3f" %
              (epoch, count, np.mean(g_loss[np.where(g_loss)]), time.time() - st))

        if epoch % save_frequency == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            output = output.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            Image.fromarray((temp * 255).astype('uint8')).save(
                result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

            torch.save(model.state_dict(), model_dir + 'checkpoint_sony_e%04d.pth' % epoch)

