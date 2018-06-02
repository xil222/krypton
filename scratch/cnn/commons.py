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
from curvetools import generate_map
from heatmap import Heatmapper
from sklearn.metrics import f1_score

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


# def __generate_positions(x_size, y_size):
#     curve_map = generate_map(x_size, y_size)
    
#     return [(int(x),int(y)) for x,y in curve_map[0]]

def __generate_positions(x_size, y_size):
    m = int(x_size); n = int(y_size)
    patch_locations = []
    for i in range(m):
        for j in range(n):
            patch_locations.append((i, j))
            
    return patch_locations

# def __generate_positions(x_size, y_size):
#     m = int(x_size); n = int(y_size)
#     patch_locations = []

#     temp = 0
#     if m % 2 == 0:
#         for i in range(n):
#             patch_locations.append((0, n-i-1))
#         temp = 1


#     return patch_locations

# def __generate_positions(x_size, y_size):
#     m = int(x_size); n = int(y_size)
#     k = 0; l = 0
#
#     patch_positions = []
#
#     while (k < m and l < n) :
#         for i in range(l, n) :
#             patch_positions.append((k,i))
#
#         k += 1
#         for i in range(k, m) :
#             patch_positions.append((i,n-1))
#
#         n -= 1
#         if ( k < m) :
#             for i in range(n - 1, (l - 1), -1) :
#                 patch_positions.append((m-1,i))
#             m -= 1
#
#         if (l < n) :
#             for i in range(m - 1, k - 1, -1) :
#                 patch_positions.append((i,l))
#             l += 1
#
#     return patch_positions



def inc_convolution(in_tensor, weights, biases, out_tensor, locations, padding, stride, p_height, p_width, beta):
    temp = inc_conv_lib.inc_conv_relu(in_tensor, weights, biases, out_tensor, locations, padding, stride, int(p_height), int(p_width), beta)
    return int(temp/1000),int(temp%1000)

def inc_convolution_bn(in_tensor, weights, bn_mean, bn_var, bn_weights, bn_biases, out_tensor, locations, padding, stride, p_height, p_width, beta, relu=True, eps=1e-5):
    temp = inc_conv_lib.inc_conv_bn(in_tensor, weights, bn_mean, bn_var, bn_weights, bn_biases, out_tensor, locations, padding, padding, stride, int(p_height), int(p_width), beta, relu, eps)
    return int(temp/1000),int(temp%1000)

def inc_convolution_bn2(in_tensor, weights, bn_mean, bn_var, bn_weights, bn_biases, out_tensor, locations, padding_y, padding_x, stride, p_height, p_width, beta, relu=True, eps=1e-5):
    temp = inc_conv_lib.inc_conv_bn(in_tensor, weights, bn_mean, bn_var, bn_weights, bn_biases, out_tensor, locations, padding_y, padding_x, stride, int(p_height), int(p_width), beta, relu, eps)
    return int(temp/1000),int(temp%1000)

def inc_add(in_tensor1, in_tensor2, out_tensor, locations, p_height, p_width, relu=True):
    temp = inc_conv_lib.inc_add(in_tensor1, in_tensor2, out_tensor, locations, p_height, p_width, relu)
    return int(p_height),int(p_width)

def inc_max_pool(in_tensor, out_tensor, locations, padding, stride, k_size, p_height, p_width, beta):
    temp = inc_conv_lib.inc_max_pool(in_tensor, out_tensor, locations, padding, stride,
                                 k_size, int(p_height), int(p_width), beta)
    return int(temp/1000),int(temp%1000)

def inc_avg_pool(in_tensor, out_tensor, locations, padding, stride, k_size, p_height, p_width, beta):
    temp = inc_conv_lib.inc_avg_pool(in_tensor, out_tensor, locations, padding, stride,
                                 k_size, int(p_height), int(p_width), beta)
    return int(temp/1000),int(temp%1000)

def update_output_locations(locations, padding_y, padding_x, stride, k_size_y, k_size_x, p_height, p_width, in_size, out_size, beta):
    temp = inc_conv_lib.update_output_locations(locations, padding_y, padding_x, stride,
                                 k_size_y, k_size_x, int(p_height), int(p_width), in_size, out_size, beta)
    return int(temp/1000),int(temp%1000)


def load_dict_from_hdf5(filename, cuda=True):
    with h5py.File(filename, 'r') as h5file:
        return __recursively_load_dict_contents_from_group(h5file, '/', cuda)


def full_inference_e2e(model, file_path, patch_size, stride, logit_index, batch_size=256, cuda=True, image_size=224, x_size=224, y_size=224):
    loader = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])
    orig_image = Image.open(file_path)
    orig_image = Variable(loader(orig_image).unsqueeze(0), volatile=True)
     
    if cuda:
        orig_image = orig_image.cuda()
    
    full_model = model(cuda)
    full_model.eval()

    output_width = int(math.ceil((x_size*1.0 - patch_size) / stride))
    total_number = output_width * output_width

    logit_values = []
    image_patch = torch.FloatTensor(3, patch_size, patch_size).fill_(128.0/255)
    if cuda:
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

        x = full_model.forward_fused(images_batch)
        logit_values.extend(x.cpu().data.numpy()[:, logit_index].flatten().tolist())

    return np.array(logit_values).reshape(output_width, output_width)


def inc_inference_e2e(model, file_path, patch_size, stride, logit_index, batch_size=64, beta=1.0, x0=0, y0=0, image_size=224,
                      x_size=224, y_size=224, cuda=True, num_batches=32):

    loader = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])
    orig_image = Image.open(file_path)
    orig_image = loader(orig_image).unsqueeze(0)

    if cuda:
        orig_image = orig_image.cuda()

    x_output_width = int(math.ceil((x_size*1.0 - patch_size) / stride))
    y_output_width = int(math.ceil((y_size*1.0 - patch_size) / stride))

    total_number = x_output_width * y_output_width
    logit_values = np.zeros((x_output_width, y_output_width), dtype=np.float32)
    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(128.0/255)    
    patch_positions = __generate_positions(x_output_width, y_output_width)
    
    if num_batches > 0:
        num_batches = min(num_batches, int(math.ceil(total_number * 1.0 / batch_size)))
    else:
        num_batches = int(math.ceil(total_number * 1.0 / batch_size))
        
    num_rounds = int(math.ceil(total_number*1.0/batch_size/num_batches))

    for n in range(num_rounds):
        images_batch = orig_image.repeat(batch_size, 1, 1, 1)

        offset = n*batch_size*num_batches
        
        for j in range(batch_size):
            index = offset + j * num_batches
            if index >= total_number:
                break

            x, y = patch_positions[index]
            x = x*stride + x0
            y = y*stride + y0
            x,y = int(x), int(y)
            images_batch[j, :, x:x + patch_size, y:y + patch_size] = image_patch

        inc_model = model(images_batch, beta=beta)
        inc_model.eval()

        logits = inc_model.initial_result[:, logit_index].flatten().tolist()
        for logit, j in zip(logits, range(batch_size)):
            index = offset + j * num_batches
            if index >= total_number:
                break
            x, y = patch_positions[index]
            logit_values[x, y] = logit


        locations = torch.zeros([batch_size, 2], dtype=torch.int32)
        for i in range(1, num_batches):
            images_batch = orig_image.repeat(batch_size, 1, 1, 1)

            for j in range(batch_size):
                index = offset + j * num_batches + i
                if index >= total_number:
                    break

                x, y = patch_positions[index]
                x = x*stride + x0
                y = y*stride + y0
                x,y = int(x), int(y)
                images_batch[j, :, x:x + patch_size, y:y + patch_size] = image_patch

                x_prev, y_prev = patch_positions[index-1]
                x_prev = x_prev*stride + x0
                y_prev = y_prev*stride + y0
                x_prev, y_prev = int(x_prev), int(y_prev)
                x = min(x, x_prev)
                y = min(y, y_prev)

                locations[j, 0] = x
                locations[j, 1] = y

            locations_final = locations
            if cuda: locations_final = locations_final.cuda()
            logits = inc_model(images_batch, locations_final, p_height=patch_size+stride, p_width=patch_size+stride)
            logits = logits.cpu().data.numpy()[:, logit_index].flatten().tolist()

            for logit, j in zip(logits, range(batch_size)):
                index = offset + j * num_batches + i
                if index >= total_number:
                    break
                x, y = patch_positions[index]
                logit_values[x, y] = logit

        del inc_model
        gc.collect()
        
    return logit_values


def inc_inference_e2e2(model, file_path, patch_size, stride, logit_index, batch_size=64, beta=1.0, x0=0, y0=0, image_size=224,
                      x_size=224, y_size=224, cuda=True):

    loader = transforms.Compose([transforms.Resize([image_size, image_size]), transforms.ToTensor()])
    orig_image = Image.open(file_path)
    orig_image = loader(orig_image).unsqueeze(0)

    if cuda:
        orig_image = orig_image.cuda()

    x_output_width = int(math.ceil((x_size*1.0 - patch_size) / stride))
    y_output_width = int(math.ceil((y_size*1.0 - patch_size) / stride))

    total_number = x_output_width * y_output_width
    logit_values = np.zeros((x_output_width, y_output_width), dtype=np.float32)
    image_patches = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(128.0/255).repeat(batch_size, 1, 1, 1)
    patch_positions = __generate_positions(x_output_width, y_output_width)
    
    num_batches = int(math.ceil(total_number * 1.0 / batch_size))
    inc_model = model(orig_image, beta=beta)
    inc_model.eval()    
 
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

        locations_final = locations
        if cuda: locations_final = locations_final.cuda()
        logits = inc_model.forward_gpu2(image_patches, locations_final, p_height=patch_size, p_width=patch_size)
        logits = logits.cpu().data.numpy()[:, logit_index].flatten().tolist()

        for logit, j in zip(logits, range(batch_size)):
            index = j * num_batches + i
            if index >= total_number:
                break
            x, y = patch_positions[index]
            logit_values[x, y] = logit

    del inc_model
    gc.collect()

    return logit_values



def adaptive_drilldown(model, file_path, patch_size, stride, logit_index, batch_size=128, image_size=224, beta=1.0, percentile=75, num_batches=32):
    final_out_width = int(math.ceil((image_size*1.0-patch_size)/stride))
    #checking for interested regions
    temp1 = inc_inference_e2e(model, file_path, max(16, patch_size), max(8, patch_size/2), logit_index,
                                    batch_size=batch_size, beta=beta, image_size=image_size, x_size=image_size,
                              y_size=image_size, num_batches=num_batches)
    temp1 = cv2.resize(temp1, (final_out_width, final_out_width))
    
    threshold = np.percentile(temp1, percentile)
    idx = np.argwhere(temp1 <= threshold)
    
    x0 = int(np.min(idx[:,0])*stride); x1 = int(np.max(idx[:,0])*stride)
    x_size = int(min(image_size-x0, x1-x0 + patch_size))
    y0 = int(np.min(idx[:,1])*stride); y1 = int(np.max(idx[:,1])*stride)
    y_size = int(min(image_size-y0, y1-y0 + patch_size))

    #drilldown into interested regions
    temp2 = inc_inference_e2e(model, file_path, patch_size, stride, logit_index,
                                    batch_size=batch_size, beta=beta, x0=x0, y0=y0, image_size=image_size,
                              x_size=x_size, y_size=y_size, num_batches=num_batches)

    temp1[int(x0/stride):int(x1/stride),int(y0/stride):int(y1/stride)] = temp2

    #optional gaussian filter
    #temp1 = ndimage.gaussian_filter(temp1, sigma=.75)
    
    return temp1

def show_heatmap_on_image(image_file_path, heatmap, patch_size, stride, image_size=224, percentile=80):
    original_image = Image.open(image_file_path)
    original_image = original_image.resize((image_size,image_size), Image.ANTIALIAS)
    width, height = heatmap.shape
    width, height = (width-1)*stride, (height-1)*stride
    original_image = original_image.crop((patch_size//2, patch_size//2, patch_size//2+width, patch_size//2+height))
    original_image = original_image.resize((image_size,image_size), Image.ANTIALIAS)
    
    heatmap = cv2.resize(heatmap, (image_size, image_size))

    threshold = np.percentile(heatmap, percentile)
    coordinates = np.where(heatmap > threshold)
    coordinates = zip(coordinates[1].tolist(),coordinates[0].tolist())
    
    heatmapper = Heatmapper(opacity=0.9, point_diameter=2, point_strength=1.0)
    heatmap = heatmapper.heatmap_on_img(coordinates, original_image)
    
    plt.imshow(heatmap)    
    plt.show()
    

def heatmap_f1_score(heatmap1, heatmap2, percentile=80):
    threshold1 = np.percentile(heatmap1, percentile)
    threshold2 = np.percentile(heatmap2, percentile)
    
    mask1 = (heatmap1>threshold1)*1
    mask2 = (heatmap2>threshold2)*1

    return f1_score(mask1.flatten(), mask2.flatten())

def show_images(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

    
def ft_train_model(model, criterion, optimizer, dataloaders,device, dataset_sizes, class_names,  num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, dataloaders, class_names, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        

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

            
#############Experimental###############            
            
def inc_convolution2(premat_tensor, in_tensor, weights, biases, out_tensor, locations, out_size, padding, stride, p_height, p_width, beta):
    temp = inc_conv_lib.inc_conv_relu2(premat_tensor, in_tensor, weights, biases, out_tensor, locations, out_size, padding, stride, int(p_height), int(p_width), beta)
    return int(temp/1000),int(temp%1000) 

def inc_max_pool2(premat_tensor, in_tensor, out_tensor, locations, out_size, padding, stride, k_size, p_height, p_width, beta):
    temp = inc_conv_lib.inc_max_pool2(premat_tensor, in_tensor, out_tensor, locations, out_size, padding, stride,
                                 k_size, int(p_height), int(p_width), beta)
    return int(temp/1000),int(temp%1000)


def final_full_projection(premat_tensor, in_tensor, out_tensor, locations, p_height, p_width):    
    inc_conv_lib.final_full_projection(premat_tensor, in_tensor, out_tensor, locations, int(p_height), int(p_width))