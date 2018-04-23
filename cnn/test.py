import random

import time
import torch
from vgg16 import VGG16
from vgg16_inc import IncrementalVGG16
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from imagenet_classes import class_names
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
import gc


def _get_position(n, size):
    n += 1
    offset = size // 2

    k = math.ceil((math.sqrt(n) - 1) / 2)
    t = 2 * k + 1
    m = t ** 2

    t -= 1

    if (n >= m - t):
        return (int(k - (m - n)) + offset, int(-k) + offset)

    m -= t

    if (n >= m - t):
        return (int(-k) + offset, int(-k + (m - n)) + offset)

    m -= t

    if (n >= m - t):
        return (int(-k + (m - n)) + offset, int(k) + offset)

    return (int(k) + offset, int(k - (m - n - t)) + offset)


def full_inference_e2e(image_file_path, patch_size, stride, interested_logit_index, batch_size=256):
    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    orig_image = Image.open(image_file_path)
    orig_image = Variable(loader(orig_image).unsqueeze(0), volatile=True).cuda()

    full_model = VGG16()
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
        logit_values.extend(x.cpu().data.numpy()[:, interested_logit_index].flatten().tolist())

    return np.array(logit_values).reshape(((224 - (patch_size - 1)), (224 - (patch_size - 1))))


def inc_inference_e2e(image_file_path, patch_size, stride, interested_logit_index, batch_size=64,
                      patch_growth_threshold=1.0):
    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    orig_image = Image.open(image_file_path)
    orig_image = Variable(loader(orig_image).unsqueeze(0), volatile=True, requires_grad=False).cuda()
    images_batch = orig_image.repeat(batch_size, 1, 1, 1)

    full_model = VGG16()
    full_model.eval()

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

    inc_model = IncrementalVGG16(full_model, images_batch, patch_growth_threshold=patch_growth_threshold)
    inc_model.eval()

    logits = inc_model.initial_result[:, interested_logit_index].flatten().tolist()
    for logit, j in zip(logits, range(batch_size)):
        index = j * num_batches
        if index >= total_number:
            break
        x, y = _get_position(index, output_width)
        logit_values[x, y] = logit

    del full_model
    torch.cuda.empty_cache()

    full_model = VGG16()
    
    images_batch = orig_image.repeat(batch_size, 1, 1, 1)
    inc_model = IncrementalVGG16(full_model, images_batch, patch_growth_threshold=patch_growth_threshold)
    del full_model
    torch.cuda.empty_cache()
    
    locations = np.zeros(shape=(batch_size, 2), dtype=np.int32)
    for i in range(1, num_batches):
        images_batch = orig_image.repeat(batch_size, 1, 1, 1)
        for j in range(batch_size):
            index = j * num_batches + i
            if index >= total_number:
                break

            x, y = _get_position(index, output_width)
            images_batch[j, :, x:x + patch_size, y:y + patch_size] = image_patch
            
            x_prev, y_prev = _get_position(index-1, output_width)
            if x == x_prev:
                y = min(y, y_prev)
            else:
                x = min(x, x_prev)
                
            if x + patch_size + stride > 223:
                x -= (x + patch_size + stride - 223)
            if y + patch_size + stride > 223:
                y -= (y + patch_size + stride - 223)
                    
            locations[j, 0] = x
            locations[j, 1] = y

        locations_var = Variable(torch.from_numpy(locations), volatile=True).cuda()
        logits = inc_model(images_batch, locations_var, p_height=patch_size+stride, p_width=patch_size+stride)
        logits = logits.cpu().data.numpy()[:, interested_logit_index].flatten().tolist()
        
        for logit, j in zip(logits, range(batch_size)):
            index = j * num_batches + i
            if index >= total_number:
                break
            x, y = _get_position(index, output_width)
            logit_values[x, y] = logit

    return logit_values


# testing incremental inference for VGG16
if __name__ == "__main__":
    image_file_path = "./dog_resized.jpg"
    interested_logit_index = 208

    # torch.cuda.synchronize()
    # prev_time = time.time()
    # outputs = full_inference_e2e(image_file_path, 4, 1, interested_logit_index)
    # print(outputs[110, 110])
    # torch.cuda.synchronize()
    # full_inference_time = time.time() - prev_time
    # print("Full Inference Time: " + str(full_inference_time))
    #
    # plt.imshow(outputs, cmap=plt.cm.rainbow_r, vmin=.75, vmax=.95, interpolation='none')
    # plt.colorbar()
    # plt.savefig('full_inf_heatmap.png')
    #
    # gc.collect()
    # torch.cuda.empty_cache()
    
    torch.cuda.synchronize()
    prev_time = time.time()
    outputs = inc_inference_e2e(image_file_path, 4, 1, interested_logit_index, patch_growth_threshold=0.4)
    print(outputs[110, 110])
    torch.cuda.synchronize()
    inc_inference_time = time.time() - prev_time
    print("Incremental Inference Time: " + str(inc_inference_time))

    plt.imshow(outputs, cmap=plt.cm.rainbow_r, vmin=.75, vmax=.95, interpolation='none')
    plt.colorbar()
    plt.savefig('inc_inf_heatmap.png')
