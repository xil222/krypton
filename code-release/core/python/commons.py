from __future__ import print_function, division

import copy
import math
import sys
import time

import gc
import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.autograd import Variable
from torchvision import transforms

from matplotlib.figure import Figure

#sys.path.append('../')
sys.path.append('/krypton/code-release')
from core.cuda._ext import inc_conv_lib


def inc_convolution(premat_tensor, in_tensor, weights, biases, out_tensor, locations, padding_y, padding_x, stride_y,
                    stride_x, p_height, p_width, beta):
    temp = inc_conv_lib.inc_convolution(premat_tensor, in_tensor, weights, biases, out_tensor, locations, padding_y,
                                        padding_x, stride_y, stride_x, int(p_height), int(p_width), beta)
    return int(temp / 1000), int(temp % 1000)


def batch_normalization(in_tensor, bn_mean, bn_var, bn_weights, bn_biases, eps=1e-5):
    temp = inc_conv_lib.batch_normalization(in_tensor, bn_mean, bn_var, bn_weights, bn_biases, eps)
    return in_tensor


def inc_add(in_tensor1, locations1, premat_tensor2, in_tensor2, locations2):
    inc_conv_lib.inc_add(in_tensor1, locations1, premat_tensor2, in_tensor2, locations2)
    return in_tensor1


def inc_stack(out_tensor, out_channels, start_channel, out_location, in_tensor, in_location, premat_tensor):
    inc_conv_lib.inc_stack(out_tensor, out_channels, start_channel, out_location, in_tensor, in_location, premat_tensor)
    return out_tensor


def inc_max_pool(premat_tensor, in_tensor, out_tensor, locations, padding_y, padding_x, stride_y, stride_x, k_size_y,
                 k_size_x, p_height, p_width, beta):
    temp = inc_conv_lib.inc_max_pool(premat_tensor, in_tensor, out_tensor, locations, padding_y, padding_x, stride_y,
                                     stride_x,
                                     k_size_y, k_size_x, int(p_height), int(p_width), beta)
    return int(temp / 1000), int(temp % 1000)


def inc_avg_pool(premat_tensor, in_tensor, out_tensor, locations, padding_y, padding_x, stride_y, stride_x, k_size_y,
                 k_size_x, p_height, p_width, beta):
    temp = inc_conv_lib.inc_avg_pool(premat_tensor, in_tensor, out_tensor, locations, padding_y, padding_x, stride_y,
                                     stride_x,
                                     k_size_y, k_size_x, int(p_height), int(p_width), beta)
    return int(temp / 1000), int(temp % 1000)


def full_projection(premat_tensor, in_tensor, out_tensor, locations, p_height, p_width):
    inc_conv_lib.full_projection(premat_tensor, in_tensor, out_tensor, locations, int(p_height), int(p_width))


def calc_bbox_coordinates(batch_size, loc_out_tensor, loc_tensor1, loc_tensor2):
    inc_conv_lib.calc_bbox_coordinates(batch_size, loc_out_tensor, loc_tensor1, loc_tensor2)
    return loc_out_tensor


def inc_inference(model_class, file_path, patch_size, stride, batch_size=128, beta=1.0, x0=0, y0=0, image_size=224,
                      x_size=224, y_size=224, gpu=True, version='v1', n_labels=1000, weights_data=None, loader=None,
                      c=0.0):
    inc_model = model_class(beta=beta, gpu=gpu, n_labels=n_labels, weights_data=weights_data).eval()

    return inc_inference_with_model(inc_model, file_path, patch_size, stride, batch_size=batch_size, beta=beta,
                                        x0=x0, y0=y0, image_size=image_size, x_size=x_size, y_size=y_size, gpu=gpu,
                                        version=version, n_labels=n_labels, weights_data=weights_data, loader=loader,
                                        c=c)


def inc_inference_with_model(inc_model, file_path, patch_size, stride, batch_size=128, beta=1.0, x0=0, y0=0,
                                 image_size=224,
                                 x_size=224, y_size=224, gpu=True, version='v1', n_labels=1000, weights_data=None,
                                 loader=None, c=0.0, g=None):
    if loader == None:
        loader = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])
    orig_image = Image.open(file_path).convert('RGB')
    orig_image = loader(orig_image).unsqueeze(0)

    if gpu:
        orig_image = orig_image.cuda()
    else:
        version = 'v2' #running plain PyTorch

    x_output_width = int(math.ceil((x_size * 1.0 - patch_size) / stride))
    y_output_width = int(math.ceil((y_size * 1.0 - patch_size) / stride))

    if g is None:
        total_number = x_output_width * y_output_width
    else:
        total_number = g

    logit_values = np.zeros((x_output_width, y_output_width), dtype=np.float32)

    image_patches = torch.FloatTensor(3, patch_size, patch_size).fill_(c).repeat(batch_size, 1, 1, 1)

    patch_positions = __generate_positions(x_output_width, y_output_width)

    num_batches = int(math.ceil(total_number * 1.0 / batch_size))

    if gpu:
        inc_model = inc_model.cuda()

    temp = inc_model.forward_materialized(orig_image).cpu().data.numpy()
    logit_index = np.argmax(temp)
    prob = np.max(temp)

    locations = torch.zeros([batch_size, 2], dtype=torch.int32)
    for i in range(num_batches):
        for j in range(batch_size):
            index = j * num_batches + i
            if index >= total_number:
                break

            x, y = patch_positions[index]
            x = x * stride + x0
            y = y * stride + y0
            x, y = int(x), int(y)

            locations[j, 0] = x
            locations[j, 1] = y

        if version == 'v1':
            logits = inc_model.forward_gpu(image_patches, locations, p_height=patch_size, p_width=patch_size)
        else:
            logits = inc_model.forward_pytorch(image_patches, locations, p_height=patch_size, p_width=patch_size)

        logits = logits.cpu().data.numpy()[:, logit_index].flatten().tolist()

        for logit, j in zip(logits, range(batch_size)):
            index = j * num_batches + i
            if index >= total_number:
                break
            x, y = patch_positions[index]
            logit_values[x, y] = logit

    del inc_model
    gc.collect()

    return logit_values, prob, logit_index


def show_heatmap(image_file_path, x, label="", width=224, alpha=1.0, prob=1.0):
    img = Image.open(image_file_path).convert('RGB')
    img = img.resize((width, width), Image.ANTIALIAS)

    stride = int(math.floor(1.0 * np.asarray(img).shape[0] / x.shape[0]))
    start = int((width - stride * x.shape[0]) // 2)
    img = Image.fromarray(np.asarray(img)[start:start + x.shape[0] * stride, start:start + x.shape[0] * stride, :])

    vmax, vmin = min(1.0, prob * 1.25), prob * 0.75

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Predicted Class: " + label, fontsize=12, y=0.9)
    axes[0].imshow(img, extent=(0, 1, 0, 1))
    axes[1].imshow(img, extent=(0, 1, 0, 1))

    im = axes[1].imshow(x, cmap=plt.cm.jet_r, alpha=alpha, interpolation='none', extent=(0, 1, 0, 1))
    im.set_clim(vmin=vmin, vmax=vmax)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1., 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.tight_layout()
    plt.show()


def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        __recursively_save_dict_contents_to_group(h5file, '/', dic)


def __recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            __recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(filename, gpu=True):
    with h5py.File(filename, 'r') as h5file:
        return __recursively_load_dict_contents_from_group(h5file, '/', gpu)


def __recursively_load_dict_contents_from_group(h5file, path, cuda=True):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = torch.from_numpy(item.value)
            if cuda:
                ans[key] = ans[key].cuda()

        elif isinstance(item, h5py._hl.group.Group, cuda):
            ans[key] = __recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def __generate_positions(x_size, y_size):
    m = int(x_size);
    n = int(y_size)
    patch_locations = []
    for i in range(m):
        for j in range(n):
            patch_locations.append((i, j))

    return patch_locations
