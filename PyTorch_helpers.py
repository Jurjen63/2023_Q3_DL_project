# Helper file with the SeeInTheDarkModel and other functons
# that are used in both the train and test file

import numpy as np
import torch
from torch import nn as nn
import sys


class SeeInTheDarkModel(nn.Module):
    """"
    Create a class for the model using a 10 layered neural network
    """
    def __init__(self):
        super(SeeInTheDarkModel, self).__init__()

        # set up layers
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)

        # initialize weights
        self.init_weights()

    def lrelu(self, x):
        return torch.max(x * 0.2, x)

    def forward(self, x):
        conv1_1 = self.lrelu(self.conv1_1(x))
        conv1_2 = self.lrelu(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.lrelu(self.conv2_1(pool1))
        conv2_2 = self.lrelu(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.lrelu(self.conv3_1(pool2))
        conv3_2 = self.lrelu(self.conv3_2(conv3_1))
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.lrelu(self.conv4_1(pool3))
        conv4_2 = self.lrelu(self.conv4_2(conv4_1))
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.lrelu(self.conv5_1(pool4))
        conv5_2 = self.lrelu(self.conv5_2(conv5_1))

        up6 = torch.cat([self.upv6(conv5_2), conv4_2], 1)
        conv6_1 = self.lrelu(self.conv6_1(up6))
        conv6_2 = self.lrelu(self.conv6_2(conv6_1))

        up7 = torch.cat([self.upv7(conv6_2), conv3_2], 1)
        conv7_1 = self.lrelu(self.conv7_1(up7))
        conv7_2 = self.lrelu(self.conv7_2(conv7_1))

        up8 = torch.cat([self.upv8(conv7_2), conv2_2], 1)
        conv8_1 = self.lrelu(self.conv8_1(up8))
        conv8_2 = self.lrelu(self.conv8_2(conv8_1))

        up9 = torch.cat([self.upv9(conv8_2), conv1_2], 1)
        conv9_1 = self.lrelu(self.conv9_1(up9))
        conv9_2 = self.lrelu(self.conv9_2(conv9_1))

        conv10 = self.conv10_1(conv9_2)

        out = nn.functional.pixel_shuffle(conv10, 2)

        return out

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.normal_(mean=0.0, std=0.02)

            elif isinstance(layer, nn.ConvTranspose2d):
                layer.weight.data.normal_(mean=0.0, std=0.02)



def pack_raw(raw, patch_size):
    """
    pack Bayer image to 4 channels
    """
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - patch_size, 0) / (16383 - patch_size)
    im = np.expand_dims(im, axis=2)
    H, W, _ = im.shape

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def print_progress(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """"
    Prints the progress of processing the images
    """
    percent = "{0:.0f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s (%d / %d)' % (prefix, bar, percent, suffix, iteration, total)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
