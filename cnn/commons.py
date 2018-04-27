import math
import sys
import cv2
import h5py
import torch
import numpy as np
from PIL import Image
from scipy import ndimage
from torch.autograd import Variable
from torchvision.transforms import transforms

sys.path.append('../')
from cuda._ext import inc_conv_lib


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
    m = x_size; n = y_size
    k = 0; l = 0
 
    patch_positions = []
    
    while (k < m and l < n) :
        for i in range(l, n) :
            patch_positions.append((k,i))
             
        k += 1
        for i in range(k, m) :
            patch_positions.append((i,n-1))
             
        n -= 1
        if ( k < m) :
            for i in range(n - 1, (l - 1), -1) :
                patch_positions.append((m-1,i))
            m -= 1
         
        if (l < n) :
            for i in range(m - 1, k - 1, -1) :
                patch_positions.append((i,l))
            l += 1
    
    return patch_positions


def __get_position(n, size):
    n += 1

    if size%2 == 0:
        offset = size / 2 - 1
    else:
        offset = size // 2
        
    k = math.ceil((math.sqrt(n) - 1) / 2)
    t = 2 * k + 1
    m = t ** 2
    t -= 1
    if n >= m - t:
        return int(k - (m - n)) + offset, int(-k) + offset
    m -= t
    if n >= m - t:
        return int(-k) + offset, int(-k + (m - n)) + offset
    m -= t
    if n >= m - t:
        return int(-k + (m - n)) + offset, int(k) + offset

    return int(k) + offset, int(k - (m - n - t)) + offset


def inc_convolution(in_tensor, weights, biases, out_tensor, locations, padding, stride, p_height, p_width, beta):
    temp = inc_conv_lib.inc_conv(in_tensor, weights, biases, out_tensor, locations, padding, stride, p_height, p_width, beta)
    return temp/1000,temp%1000


def inc_max_pool(in_tensor, out_tensor, locations, padding, stride, k_size, p_height, p_width, beta):
    temp = inc_conv_lib.inc_max_pool(in_tensor, out_tensor, locations, padding, stride,
                                 k_size, p_height, p_width, beta)
    return temp/1000,temp%1000

def load_dict_from_hdf5(filename, cuda=True):
    with h5py.File(filename, 'r') as h5file:
        return __recursively_load_dict_contents_from_group(h5file, '/', cuda)


def full_inference_e2e(model, file_path, patch_size, stride, logit_index, batch_size=256):
    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    orig_image = Image.open(file_path)
    orig_image = Variable(loader(orig_image).unsqueeze(0), volatile=True).cuda()

    full_model = model()
    full_model.eval()

    output_width = int(math.ceil((224.0 - patch_size) / stride))
    total_number = output_width * output_width

    logit_values = []
    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0)

    for i in range(0, int(math.ceil(total_number * 1.0 / batch_size))):
        start = i * batch_size
        end = min(i * batch_size + batch_size, total_number)

        images_batch = orig_image.repeat(end - start, 1, 1, 1)

        for idx, j in enumerate(range(start, end)):
            x = (j / output_width)*stride
            y = (j % output_width)*stride

            images_batch[idx, :, x:x + patch_size, y:y + patch_size] = image_patch

        x = full_model.forward_fused(images_batch)
        logit_values.extend(x.cpu().data.numpy()[:, logit_index].flatten().tolist())

    return np.array(logit_values).reshape(output_width, output_width)


def inc_inference_e2e(model, file_path, patch_size, stride, logit_index, batch_size=64, beta=1.0, x0=0, y0=0,
                      x_size=224, y_size=224, cuda=True):

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    orig_image = Image.open(file_path)
    orig_image = loader(orig_image).unsqueeze(0)

    if cuda:
        orig_image = orig_image.cuda()

    images_batch = orig_image.repeat(batch_size, 1, 1, 1)

    x_output_width = int(math.ceil((x_size*1.0 - patch_size) / stride))
    y_output_width = int(math.ceil((y_size*1.0 - patch_size) / stride))

    total_number = x_output_width * y_output_width
    logit_values = np.zeros((x_output_width, y_output_width), dtype=np.float32)
    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0)
    num_batches = int(math.ceil(total_number * 1.0 / batch_size))
    
    patch_positions = __generate_positions(x_output_width, y_output_width)
    
    for j in range(batch_size):
        index = j * num_batches
        if index >= total_number:
            break

        x, y = patch_positions[index]
        x = x*stride + x0
        y = y*stride + y0
        images_batch[j, :, x:x + patch_size, y:y + patch_size] = image_patch

    inc_model = model(images_batch, beta=beta)
    inc_model.eval()

    logits = inc_model.initial_result[:, logit_index].flatten().tolist()
    for logit, j in zip(logits, range(batch_size)):
        index = j * num_batches
        if index >= total_number:
            break
        x, y = patch_positions[index]
        logit_values[x, y] = logit

    locations = np.zeros(shape=(batch_size, 2), dtype=np.int32)
    locations = torch.from_numpy(locations).cuda()

    for i in range(1, num_batches):
        images_batch = orig_image.repeat(batch_size, 1, 1, 1)
        for j in range(batch_size):
            index = j * num_batches + i
            if index >= total_number:
                break

            x, y = patch_positions[index]
            x = x*stride + x0
            y = y*stride + y0
            images_batch[j, :, x:x + patch_size, y:y + patch_size] = image_patch

            x_prev, y_prev = patch_positions[index-1]
            x_prev = x_prev*stride + x0
            y_prev = y_prev*stride + y0
            
            x = min(x, x_prev)
            y = min(y, y_prev)

            locations[j, 0] = x
            locations[j, 1] = y

        logits = inc_model(images_batch, locations, p_height=patch_size + stride, p_width=patch_size + stride)
        logits = logits.cpu().data.numpy()[:, logit_index].flatten().tolist()

        for logit, j in zip(logits, range(batch_size)):
            index = j * num_batches + i
            if index >= total_number:
                break
            x, y = patch_positions[index]
            logit_values[x, y] = logit

    return logit_values

def adaptive_drilldown(model, file_path, patch_size, stride, logit_index, batch_size=128, beta=1.0, threshold=0.9):
    final_out_width = int(math.ceil((224.0-patch_size)/stride))
    #checking for interested regions
    temp1 = inc_inference_e2e(model, file_path, patch_size, patch_size/2, logit_index,
                                    batch_size=batch_size, beta=beta)
    temp1 = cv2.resize(temp1, (final_out_width, final_out_width))
    
    idx = np.argwhere(temp1 <= threshold)
    
    x0 = np.min(idx[:,0])*stride; x1 = np.max(idx[:,0])*stride
    x_size = min(224-x0, x1-x0 + patch_size)
    y0 = np.min(idx[:,1])*stride; y1 = np.max(idx[:,1])*stride
    y_size = min(224-y0, y1-y0 + patch_size)

    #drilldown into interested regions
    temp2 = inc_inference_e2e(model, file_path, patch_size, stride, logit_index,
                                    batch_size=batch_size, beta=beta, x0=x0, y0=y0, x_size=x_size, y_size=y_size)

    temp1[x0/stride:x1/stride,y0/stride:y1/stride] = temp2

    #optional gaussian filter
    temp1 = ndimage.gaussian_filter(temp1, sigma=.75)
    
    return temp1