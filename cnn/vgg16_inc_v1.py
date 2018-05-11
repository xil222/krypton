#!/usr/bin/env python
# coding=utf-8

import random
import math
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from commons import load_dict_from_hdf5
from torch.autograd import Variable
from torchvision.transforms import transforms

from commons import inc_convolution, inc_max_pool
from imagenet_classes import class_names
from vgg16 import VGG16


class IncrementalVGG16V1(nn.Module):

    def __init__(self, in_tensor, cuda=True):
        super(IncrementalVGG16V1, self).__init__()

        # performing initial full inference
        full_model = VGG16(cuda)
        full_model.eval()
        self.cuda = cuda
        
        if self.cuda:
            in_tensor = in_tensor.cuda()
            
        full_model.forward_materialized(Variable(in_tensor, volatile=True))
        self.full_model = full_model
        
        self.conv1_1_op = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv1_2_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool1_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv2_2_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool2_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv3_2_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv3_3_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool3_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv4_2_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv4_3_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool4_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv5_2_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv5_3_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool5_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1)
        )
        
        self.__initialize_weights(cuda)

        self.cache = {}
        
        
    def forward(self, image, patch, in_locations, patch_size):
        if self.cuda:
            patch = patch.cuda()
            image = image.cuda()
        
        b = len(in_locations)
        
        patches = patch.repeat(b, 1, 1, 1)
        
        layers = [
                  self.conv1_1_op, self.conv1_2_op, self.pool1_op,
                  self.conv2_1_op, self.conv2_2_op, self.pool2_op,
                  self.conv3_1_op, self.conv3_2_op, self.conv3_3_op, self.pool3_op,
                  self.conv4_1_op, self.conv4_2_op, self.conv4_3_op, self.pool4_op,
                  self.conv5_1_op, self.conv5_2_op, self.conv5_3_op, self.pool5_op
                 ]
        premat_data = [
                  image, self.full_model.conv1_1.data, self.full_model.conv1_2.data, self.full_model.pool1.data,
                  self.full_model.conv2_1.data, self.full_model.conv2_2.data, self.full_model.pool2.data,
                  self.full_model.conv3_1.data, self.full_model.conv3_2.data, self.full_model.conv3_3.data, self.full_model.pool3.data,
                  self.full_model.conv4_1.data, self.full_model.conv4_2.data, self.full_model.conv4_3.data, self.full_model.pool4.data,
                  self.full_model.conv5_1.data, self.full_model.conv5_2.data, self.full_model.conv5_3.data
                ]
        C = [3, 64, 64, 64, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]
        sizes = [224, 224, 112, 112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 14, 14, 14, 7]
        S = [1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2]
        P = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
        K = [3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2]

        prev_size = 224
        for layer, data, c, size, s, p, k in zip(layers, premat_data, C, sizes, S, P, K):
            out_p_size = int(min(math.ceil((patch_size + k - 1.0)/s), size))
            in_p_size = k + (out_p_size-1)*s
            out_locations = self.__get_output_locations(in_locations, out_p_size, s, p, k, size)
            if layer in self.cache:
                x = self.cache[layer]#.fill_(0.0)
            else:
                x = torch.FloatTensor(b, c, in_p_size, in_p_size).fill_(0.0)
                self.cache[layer] = x

            if self.cuda:
                x = x.cuda()
            
            for i in range(b):
                x0 = 0 if s*out_locations[i][0]-p >= 0 else -1*(s*out_locations[i][0]-p)
                x1 = min(prev_size - s*out_locations[i][0]+p, in_p_size)
                y0 = 0 if s*out_locations[i][1]-p >= 0 else -1*(s*out_locations[i][1]-p)
                y1 = min(prev_size - s*out_locations[i][1]+p, in_p_size)
                
                x[i,:,x0:x1, y0:y1] = \
                    data[0,:,max(s*out_locations[i][0]-p,0):max(0, s*out_locations[i][0]-p)+x1-x0,
                     max(0, s*out_locations[i][1]-p):max(0, s*out_locations[i][1]-p)+y1-y0]

                rx = x0 + in_locations[i][0] - max(s*out_locations[i][0]-p,0)
                ry = y0 + in_locations[i][1] - max(s*out_locations[i][1]-p,0)

                x[i,:,rx:rx+patch_size,ry:ry+patch_size] = patches[i,:,:,:]
                
            patches = layer(Variable(x, volatile=True)).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
        output = self.full_model.pool5.data.repeat(b, 1, 1, 1)
        for i, (x, y) in enumerate(out_locations):
            output[i,:,x:x+out_p_size,y:y+out_p_size] = patches[i,:,:,:]
        
        x = Variable(output, volatile=True)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
        
    def __get_output_locations(self, in_locations, out_p_size, stride, padding, ksize, out_size):
        out_locations = []
        
        for x,y in in_locations:
            x_out = int(max(math.ceil((padding + x - ksize + 1.0)/stride), 0))
            y_out = int(max(math.ceil((padding + y - ksize + 1.0)/stride), 0))
            
            if x_out + out_p_size > out_size:
                x_out = out_size - out_p_size
            if y_out + out_p_size > out_size:
                y_out = out_size - out_p_size
                
            out_locations.append((x_out, y_out))
            
        return out_locations
    
    def __initialize_weights(self, cuda):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_data = load_dict_from_hdf5(dir_path + "/vgg16_weights_ptch.h5", cuda)

        self.conv1_1_op[0].weight.data = weights_data['conv1_1_W:0']
        self.conv1_1_op[0].bias.data = weights_data['conv1_1_b:0']
        self.conv1_2_op[0].weight.data = weights_data['conv1_2_W:0']
        self.conv1_2_op[0].bias.data = weights_data['conv1_2_b:0']

        self.conv2_1_op[0].weight.data = weights_data['conv2_1_W:0']
        self.conv2_1_op[0].bias.data = weights_data['conv2_1_b:0']
        self.conv2_2_op[0].weight.data = weights_data['conv2_2_W:0']
        self.conv2_2_op[0].bias.data = weights_data['conv2_2_b:0']

        self.conv3_1_op[0].weight.data = weights_data['conv3_1_W:0']
        self.conv3_1_op[0].bias.data = weights_data['conv3_1_b:0']
        self.conv3_2_op[0].weight.data = weights_data['conv3_2_W:0']
        self.conv3_2_op[0].bias.data = weights_data['conv3_2_b:0']
        self.conv3_3_op[0].weight.data = weights_data['conv3_3_W:0']
        self.conv3_3_op[0].bias.data = weights_data['conv3_3_b:0']

        self.conv4_1_op[0].weight.data = weights_data['conv4_1_W:0']
        self.conv4_1_op[0].bias.data = weights_data['conv4_1_b:0']
        self.conv4_2_op[0].weight.data = weights_data['conv4_2_W:0']
        self.conv4_2_op[0].bias.data = weights_data['conv4_2_b:0']
        self.conv4_3_op[0].weight.data = weights_data['conv4_3_W:0']
        self.conv4_3_op[0].bias.data = weights_data['conv4_3_b:0']

        self.conv5_1_op[0].weight.data = weights_data['conv5_1_W:0']
        self.conv5_1_op[0].bias.data = weights_data['conv5_1_b:0']
        self.conv5_2_op[0].weight.data = weights_data['conv5_2_W:0']
        self.conv5_2_op[0].bias.data = weights_data['conv5_2_b:0']
        self.conv5_3_op[0].weight.data = weights_data['conv5_3_W:0']
        self.conv5_3_op[0].bias.data = weights_data['conv5_3_b:0']

        self.classifier[0].weight.data = weights_data['fc6_W:0']
        self.classifier[0].bias.data = weights_data['fc6_b:0']

        self.classifier[2].weight.data = weights_data['fc7_W:0']
        self.classifier[2].bias.data = weights_data['fc7_b:0']

        self.classifier[4].weight.data = weights_data['fc8_W:0']
        self.classifier[4].bias.data = weights_data['fc8_b:0']

        
if __name__ == '__main__':
    batch_size = 1
    patch_size = 16
    input_size = 224

    random.seed(0)
    
    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0)

    x_loc = random.sample(range(0, input_size - patch_size), batch_size)
    y_loc = random.sample(range(0, input_size - patch_size), batch_size)
    patch_locations = zip(x_loc, y_loc)
    
    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    image = Image.open('./dog_resized.jpg')
    image = loader(image)
    image = image.unsqueeze(0)

    inc_model = IncrementalVGG16V1(image)

    inc_model.eval()
    x = inc_model(image, image_patch, patch_locations, patch_size)
    print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])

    #temp = y - x
    #print(np.max(np.abs(temp.cpu().data.numpy())))
