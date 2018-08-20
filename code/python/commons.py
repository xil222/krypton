from __future__ import print_function, division

import copy
import math
import sys
import time

import gc
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.autograd import Variable
from torchvision import transforms

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append('../')
from cuda._ext import inc_conv_lib

def inc_convolution(premat_tensor, in_tensor, weights, biases, out_tensor, locations, padding_y, padding_x, stride_y, stride_x, p_height, p_width, beta):
    temp = inc_conv_lib.inc_convolution(premat_tensor, in_tensor, weights, biases, out_tensor, locations, padding_y, padding_x, stride_y, stride_x, int(p_height), int(p_width), beta)
    return int(temp/1000),int(temp%1000)

def batch_normalization(in_tensor, bn_mean, bn_var, bn_weights, bn_biases, eps=1e-5):
    temp = inc_conv_lib.batch_normalization(in_tensor, bn_mean, bn_var, bn_weights, bn_biases, eps)
    return in_tensor

def inc_add(in_tensor1, locations1, premat_tensor2, in_tensor2, locations2):
    inc_conv_lib.inc_add(in_tensor1, locations1, premat_tensor2, in_tensor2, locations2)
    return in_tensor1

def inc_stack(out_tensor, out_channels, start_channel, out_location, in_tensor, in_location, premat_tensor):
    inc_conv_lib.inc_stack(out_tensor, out_channels, start_channel, out_location, in_tensor, in_location, premat_tensor)
    return out_tensor
    
def inc_max_pool(premat_tensor, in_tensor, out_tensor, locations, padding_y, padding_x, stride_y, stride_x, k_size_y, k_size_x, p_height, p_width, beta):
    temp = inc_conv_lib.inc_max_pool(premat_tensor, in_tensor, out_tensor, locations, padding_y, padding_x, stride_y, stride_x,
                                 k_size_y, k_size_x, int(p_height), int(p_width), beta)
    return int(temp/1000),int(temp%1000)

def inc_avg_pool(premat_tensor, in_tensor, out_tensor, locations, padding_y, padding_x, stride_y, stride_x, k_size_y, k_size_x, p_height, p_width, beta):
    temp = inc_conv_lib.inc_avg_pool(premat_tensor, in_tensor, out_tensor, locations, padding_y, padding_x, stride_y, stride_x,
                                 k_size_y, k_size_x, int(p_height), int(p_width), beta)
    return int(temp/1000),int(temp%1000)

def full_projection(premat_tensor, in_tensor, out_tensor, locations, p_height, p_width):    
    inc_conv_lib.full_projection(premat_tensor, in_tensor, out_tensor, locations, int(p_height), int(p_width))
    
def calc_bbox_coordinates(batch_size, loc_out_tensor, loc_tensor1, loc_tensor2):
    inc_conv_lib.calc_bbox_coordinates(batch_size, loc_out_tensor, loc_tensor1, loc_tensor2)
    return loc_out_tensor


def full_inference_e2e(model, file_path, patch_size, stride, batch_size=256, gpu=True, version='v1', image_size=224, x_size=224, y_size=224, n_labels=1000, weights_data=None, loader=None, c=0.0):
    if loader == None:
        loader = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])
    orig_image = Image.open(file_path).convert('RGB')
    orig_image = Variable(loader(orig_image).unsqueeze(0))
     
    if gpu:
        orig_image = orig_image.cuda()
    
    full_model = model(gpu=gpu, n_labels=n_labels, weights_data=weights_data).eval()
    if gpu:
        full_model = full_model.cuda()
    full_model.eval()

    temp = full_model(orig_image).cpu().data.numpy()[0,:]
    logit_index = np.argmax(temp)
    prob = np.max(temp)
    
    output_width = int(math.ceil((x_size*1.0 - patch_size) / stride))
    total_number = output_width * output_width

    logit_values = []
    image_patch = torch.FloatTensor(3, patch_size, patch_size).fill_(c)
    if gpu:
        image_patch = image_patch.cuda()

    for i in range(0, int(math.ceil(total_number * 1.0 / batch_size))):
        start = i * batch_size
        end = min(i * batch_size + batch_size, total_number)

        images_batch = orig_image.repeat(end - start, 1, 1, 1)

        for idx, j in enumerate(range(start, end)):
            x = (j // output_width)*stride
            y = (j % output_width)*stride
            x,y=int(x),int(y)
            images_batch[idx, :, x:x + patch_size, y:y + patch_size] = image_patch

        if version == 'v1':
            x = full_model.forward_fused(images_batch)
        else:
            x = full_model.forward_pytorch(images_batch)
            
        logit_values.extend(x.cpu().data.numpy()[:, logit_index].flatten().tolist())

    x = np.array(logit_values).reshape(output_width, output_width)
    
    return x, prob


def inc_inference_e2e(model, file_path, patch_size, stride, batch_size=64, beta=1.0, x0=0, y0=0, image_size=224,
                      x_size=224, y_size=224, gpu=True, version='v1', n_labels=1000, weights_data=None, loader=None, c=0.0):

    if loader == None:
        loader = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])
    orig_image = Image.open(file_path).convert('RGB')
    orig_image = loader(orig_image).unsqueeze(0)

    if gpu:
        orig_image = orig_image.cuda()

    x_output_width = int(math.ceil((x_size*1.0 - patch_size) / stride))
    y_output_width = int(math.ceil((y_size*1.0 - patch_size) / stride))

    total_number = x_output_width * y_output_width
    logit_values = np.zeros((x_output_width, y_output_width), dtype=np.float32)

    image_patches = torch.FloatTensor(3, patch_size, patch_size).fill_(c).repeat(batch_size, 1, 1, 1)
    
    patch_positions = __generate_positions(x_output_width, y_output_width)
    
    num_batches = int(math.ceil(total_number * 1.0 / batch_size))
    inc_model = model(beta=beta, gpu=gpu, n_labels=n_labels, weights_data=weights_data).eval()
    
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
            x = x*stride + x0
            y = y*stride + y0
            x,y = int(x), int(y)
            
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

    return logit_values, prob



def adaptive_drilldown(model, file_path, patch_size, stride, batch_size=128, image_size=224, beta=1.0, percentile=75, gpu=True, version='v1', n_labels=1000, weights_data=None, loader=None):
    final_out_width = int(math.ceil((image_size*1.0-patch_size)/stride))
    #checking for interested regions
    temp1, prob = inc_inference_e2e(model, file_path, max(16, patch_size), max(8, patch_size/2),
                                    batch_size=batch_size, beta=beta, image_size=image_size, x_size=image_size,
                              y_size=image_size, gpu=gpu, version=version, n_labels=n_labels, weights_data=weights_data, loader=loader)
    temp1 = cv2.resize(temp1, (final_out_width, final_out_width))
    
    threshold = np.percentile(temp1, percentile)
    idx = np.argwhere(temp1 <= threshold)
    
    x0 = int(np.min(idx[:,0])*stride); x1 = int(np.max(idx[:,0])*stride)
    x_size = int(min(image_size-x0, x1-x0 + patch_size))
    y0 = int(np.min(idx[:,1])*stride); y1 = int(np.max(idx[:,1])*stride)
    y_size = int(min(image_size-y0, y1-y0 + patch_size))

    #drilldown into interested regions
    temp2 = inc_inference_e2e(model, file_path, patch_size, stride,
                                    batch_size=batch_size, beta=beta, x0=x0, y0=y0, image_size=image_size,
                              x_size=x_size, y_size=y_size, gpu=gpu, version=version,
                              n_labels=n_labels, weights_data=weights_data, loader=loader)

    temp1[int(x0/stride):int(x1/stride),int(y0/stride):int(y1/stride)] = temp2

    #optional gaussian filter
    #temp1 = ndimage.gaussian_filter(temp1, sigma=.75)
    
    return temp1, prob
    
    
def generate_heatmap(image_file_path, x, show=True, label="", width=224, alpha=1.0, prob=1.0):

    img = Image.open(image_file_path).convert('RGB')
    img = img.resize((width,width), Image.ANTIALIAS)

    stride = int(math.floor(1.0*np.asarray(img).shape[0]/x.shape[0]))
    start = int((width - stride*x.shape[0])//2)
    img = Image.fromarray(np.asarray(img)[start:start+x.shape[0]*stride,start:start+x.shape[0]*stride,:])
    
    vmax, vmin = min(1.0, prob*1.25), prob*0.75
    
    if show:
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.suptitle("Predicted Class: " + label, fontsize=12, y=0.9)
        axes[0].imshow(img, extent=(0,1,0,1))
        axes[1].imshow(img, extent=(0,1,0,1))

        im = axes[1].imshow(x, cmap=plt.cm.jet_r, alpha=alpha, interpolation='none', extent=(0,1,0,1))
        im.set_clim(vmin=vmin, vmax=vmax)
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1., 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        fig.tight_layout()
        #plt.axis('off')

        plt.show()
      
    
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    
    ax.imshow(img, extent=(0,1,0,1))
    im = ax.imshow(x, cmap=plt.cm.jet_r, alpha=1.0, interpolation='none', extent=(0,1,0,1))
    im.set_clim(vmin=vmin, vmax=vmax)
    
    ax.axis('off')
    canvas.draw()  
    
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = data[:, np.where(~np.all(data==255, axis = 0))[0]]
    data = data[np.where(~np.all(data==255, axis = 1,))[0], :]
    data = cv2.resize(data, dsize=(x.shape[0], x.shape[1]), interpolation=cv2.INTER_CUBIC)
    return data

            

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
    m = int(x_size); n = int(y_size)
    patch_locations = []
    for i in range(m):
        for j in range(n):
            patch_locations.append((i, j))
            
    return patch_locations