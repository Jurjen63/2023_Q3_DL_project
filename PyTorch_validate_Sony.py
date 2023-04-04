# import packages
import os
import numpy as np
import rawpy
import glob
import torch
from PIL import Image

from PyTorch_helpers import SeeInTheDarkModel, pack_raw, print_progress


# set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# define directories
input_dir = './dataset/Sony/short/'

torch_dir = './Pytorch/'
result_dir = './PyTorch/validate_result_Sony/'

# fail safes in case the directories do not exist
if not os.path.isdir(torch_dir):
    os.makedirs(torch_dir)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

# connect to trained model
model_dir = './PyTorch/model_Sony/'
model_name = 'checkpoint_sony_e4000.pth'

# get test IDs
test_fns = glob.glob(input_dir + '2*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]
total = len(test_ids)
test_ids = list(set(test_ids))

# set up patch size
patch_size = 512

model = SeeInTheDarkModel()
model.load_state_dict(torch.load(model_dir + model_name, map_location=torch.device(device)))
model = model.to(device)


count = 0
for test_id in test_ids:
    in_files = glob.glob(input_dir + '%05d*.ARW' % test_id)
    for k in range(len(in_files)):
        count += 1
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)

        print_progress(count, total, prefix='Processing image %s:' % in_fn[:-4], suffix='Complete')
        # here the original papers uses the ground truths to determine the ratio between expose time
        # we have chosen not to use the ground truths in any way in our test
        # therefore we use the following code that determines the ratio using the name of the RAW image
        exp_time = in_fn[-8:-4]
        if exp_time == '.04s':
            ratio = 10/0.04
        elif exp_time == '0.1s':
            ratio = 10/0.1
        elif exp_time == '033s':
            ratio = 10 / 0.033
        else:
            ratio = 250

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw, patch_size=patch_size), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)

        input_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)
        output = model(input_img)
        output = output.permute(0, 2, 3, 1).cpu().data.numpy()
        output = np.minimum(np.maximum(output, 0), 1)[0, :, :, :]

        Image.fromarray((output * 255).astype('uint8')).save(result_dir + '%s_%d_out.png' % (in_fn[:-4], ratio))

print("Done with processing images")
