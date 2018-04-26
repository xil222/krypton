import math
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

sys.path.append('../')
from cuda._ext import inc_conv_lib


def _out_patch_size(p_height, p_width, k_size, stride, out_size, beta):
    p_height_temp = min(int(math.ceil((p_height + k_size - 1) * 1.0 / stride)), out_size)
    p_width_temp = min(int(math.ceil((p_width + k_size - 1) * 1.0 / stride)), out_size)

    if p_height_temp > out_size*beta:
        # print(math.ceil(p_height*1.0/stride), math.ceil(p_width*1.0/stride), out_size)
        return int(math.ceil(p_height*1.0/stride)), int(math.ceil(p_width*1.0/stride))

    # print(p_height_temp, p_width_temp, out_size)
    return p_height_temp, p_width_temp


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


def _get_position(n, size):
    n += 1
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
    inc_conv_lib.inc_conv(in_tensor, weights, biases, out_tensor, locations, padding, stride, p_height, p_width, beta)
    out_size = out_tensor.shape[3]
    k_size = weights.shape[3]
    return _out_patch_size(p_height, p_width, k_size, stride, out_size, beta)


def inc_max_pool(in_tensor, out_tensor, locations, padding, stride, k_size, p_height, p_width, beta):
    inc_conv_lib.inc_max_pool(in_tensor, out_tensor, locations, padding, stride,
                                 k_size, p_height, p_width, beta)
    out_size = out_tensor.shape[3]
    return _out_patch_size(p_height, p_width, k_size, stride, out_size, beta)


def load_dict_from_hdf5(filename, cuda=True):
    with h5py.File(filename, 'r') as h5file:
        return __recursively_load_dict_contents_from_group(h5file, '/', cuda)


def full_inference_e2e(model, file_path, patch_size, stride, logit_index, batch_size=256):
    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    orig_image = Image.open(file_path)
    orig_image = Variable(loader(orig_image).unsqueeze(0), volatile=True).cuda()

    full_model = model()
    full_model.eval()

    output_width = (224 - (patch_size - 1)) / stride
    total_number = output_width * output_width

    logit_values = []
    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0)

    for i in range(0, int(math.ceil(total_number * 1.0 / batch_size))):
        start = i * batch_size
        end = min(i * batch_size + batch_size, total_number)

        images_batch = orig_image.repeat(end - start, 1, 1, 1)

        for idx, j in enumerate(range(start, end)):
            x = j / (224 - (patch_size - 1))
            y = j % (224 - (patch_size - 1))

            images_batch[idx, :, x:x + patch_size, y:y + patch_size] = image_patch

        x = full_model.forward_fused(images_batch)
        logit_values.extend(x.cpu().data.numpy()[:, logit_index].flatten().tolist())

    return np.array(logit_values).reshape(((224 - (patch_size - 1)), (224 - (patch_size - 1))))


def inc_inference_e2e(model, file_path, patch_size, stride, logit_index, batch_size=64, beta=1.0, cuda=True):

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    orig_image = Image.open(file_path)
    orig_image = loader(orig_image).unsqueeze(0)

    if cuda:
        orig_image = orig_image.cuda()

    images_batch = orig_image.repeat(batch_size, 1, 1, 1)

    output_width = (224 - (patch_size - 1)) / stride
    total_number = output_width * output_width
    logit_values = np.zeros((output_width, output_width), dtype=np.float32)
    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0)
    num_batches = int(math.ceil(total_number * 1.0 / batch_size))

    for j in range(batch_size):
        index = j * num_batches
        if index >= total_number:
            break

        x, y = _get_position(index, output_width)
        images_batch[j, :, x:x + patch_size, y:y + patch_size] = image_patch

    inc_model = model(images_batch, beta=beta)
    inc_model.eval()

    logits = inc_model.initial_result[:, logit_index].flatten().tolist()
    for logit, j in zip(logits, range(batch_size)):
        index = j * num_batches
        if index >= total_number:
            break
        x, y = _get_position(index, output_width)
        logit_values[x, y] = logit

    locations = np.zeros(shape=(batch_size, 2), dtype=np.int32)
    locations = torch.from_numpy(locations).cuda()

    for i in range(1, num_batches):
        images_batch = orig_image.repeat(batch_size, 1, 1, 1)
        for j in range(batch_size):
            index = j * num_batches + i
            if index >= total_number:
                break

            x, y = _get_position(index, output_width)
            images_batch[j, :, x:x + patch_size, y:y + patch_size] = image_patch

            x_prev, y_prev = _get_position(index - 1, output_width)
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
            x, y = _get_position(index, output_width)
            logit_values[x, y] = logit

    return logit_values