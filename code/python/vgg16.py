#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch.nn as nn
from PIL import Image
import os
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms

from commons import inc_convolution, inc_max_pool, full_projection
from commons import load_dict_from_hdf5
from imagenet_classes import class_names


class VGG16(nn.Module):

    def __init__(self, beta=1.0, gpu=True, n_labels=1000, weights_data=None):
        super(VGG16, self).__init__()
        self.initialized = False
        self.tensor_cache = {}
        
        self.conv1_1_op = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv1_1_inc_op = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv1_2_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv1_2_inc_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.ReLU(inplace=True))        
        self.pool1_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv2_1_inc_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv2_2_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv2_2_inc_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool2_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv3_1_inc_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv3_2_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv3_2_inc_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv3_3_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv3_3_inc_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool3_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv4_1_inc_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv4_2_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv4_2_inc_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv4_3_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv4_3_inc_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool4_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv5_1_inc_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.conv5_2_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv5_2_inc_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))        
        self.conv5_3_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv5_3_inc_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=0), nn.ReLU(inplace=True))
        self.pool5_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, n_labels),
            nn.Softmax(dim=1)
        )

        self.weights_data = weights_data
        self.__initialize_weights(gpu)
        self.gpu = gpu
        self.beta = beta

        #used for pytorch based impl.
        self.cache = {}
        
    def forward(self, x):
        return self.forward_fused(x)

    def forward_fused(self, x):
        if self.gpu:
            x = x.cuda()
            
        x = self.conv1_1_op(x)
        x = self.conv1_2_op(x)
        x = self.pool1_op(x)     
    
        x = self.conv2_1_op(x)        
        x = self.conv2_2_op(x)
        x = self.pool2_op(x)          
        
        x = self.conv3_1_op(x)        
        x = self.conv3_2_op(x)         
        x = self.conv3_3_op(x)        
        x = self.pool3_op(x)
    
        x = self.conv4_1_op(x)
        x = self.conv4_2_op(x)
        x = self.conv4_3_op(x)
        x = self.pool4_op(x)
          
        x = self.conv5_1_op(x)
        x = self.conv5_2_op(x)               
        x = self.conv5_3_op(x)
        x = self.pool5_op(x)
    
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

    def forward_materialized(self, x):
        self.initialized = True
        if self.gpu:
            x = x.cuda()
        
        self.image = x
        self.conv1_1 = self.conv1_1_op(x)
        self.conv1_2 = self.conv1_2_op(self.conv1_1)
        self.pool1 = self.pool1_op(self.conv1_2)

        self.conv2_1 = self.conv2_1_op(self.pool1)
        self.conv2_2 = self.conv2_2_op(self.conv2_1)
        self.pool2 = self.pool2_op(self.conv2_2)

        self.conv3_1 = self.conv3_1_op(self.pool2)
        self.conv3_2 = self.conv3_2_op(self.conv3_1)
        self.conv3_3 = self.conv3_3_op(self.conv3_2)
        self.pool3 = self.pool3_op(self.conv3_3)

        self.conv4_1 = self.conv4_1_op(self.pool3)
        self.conv4_2 = self.conv4_2_op(self.conv4_1)
        self.conv4_3 = self.conv4_3_op(self.conv4_2)
        self.pool4 = self.pool4_op(self.conv4_3)

        self.conv5_1 = self.conv5_1_op(self.pool4)
        self.conv5_2 = self.conv5_2_op(self.conv5_1)
        self.conv5_3 = self.conv5_3_op(self.conv5_2)
        self.pool5 = self.pool5_op(self.conv5_3)

        x = self.pool5.view(self.pool5.size(0), -1)
        x = self.classifier(x)

        return x

    def forward_gpu(self, x, locations, p_height, p_width, beta=None):
        if not self.initialized:
            raise Exception("Not initialized...")
        
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            
        batch_size = x.shape[0]
        
        debug = False
        
        if self.gpu:
            x = x.cuda()
            locations = locations.cuda()

        # conv1_1
        out = self.__get_tensor('conv1_1', batch_size, 64, p_height, p_width, 3, 1, 224, 224)
        p_height, p_width = inc_convolution(self.image.data, x, self.conv1_1_op[0].weight.data, self.conv1_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)

        # conv1_2
        out = self.__get_tensor('conv1_2', batch_size, 64, p_height, p_width, 3, 1, 224, 224)
        p_height, p_width = inc_convolution(self.conv1_1.data, x, self.conv1_2_op[0].weight.data, self.conv1_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
    
        # pool1
        out = self.__get_tensor('pool1', batch_size, 64, p_height, p_width, 2, 2, 224, 112)
        p_height, p_width = inc_max_pool(self.conv1_2.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out                                            
        if debug: print(locations, p_height, x.shape)
        
        # conv2_1
        out = self.__get_tensor('conv2_1', batch_size, 128, p_height, p_width, 3, 1, 112, 112)
        p_height, p_width = inc_convolution(self.pool1.data, x, self.conv2_1_op[0].weight.data, self.conv2_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
        
        # conv2_2
        out = self.__get_tensor('conv2_2', batch_size, 128, p_height, p_width, 3, 1, 112, 112)
        p_height, p_width = inc_convolution(self.conv2_1.data, x, self.conv2_2_op[0].weight.data, self.conv2_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
        
        # pool2
        out = self.__get_tensor('pool2', batch_size, 128, p_height, p_width, 2, 2, 112, 56, 56)
        p_height, p_width = inc_max_pool(self.conv2_2.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out
        if debug: print(locations, p_height, x.shape)
        
        # conv3_1
        out = self.__get_tensor('conv3_1', batch_size, 256, p_height, p_width, 3, 1, 56, 56)
        p_height, p_width = inc_convolution(self.pool2.data, x, self.conv3_1_op[0].weight.data, self.conv3_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
        
        # conv3_2
        out = self.__get_tensor('conv3_2', batch_size, 256, p_height, p_width, 3, 1, 56, 56)
        p_height, p_width = inc_convolution(self.conv3_1.data, x, self.conv3_2_op[0].weight.data, self.conv3_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)    
        if debug: print(locations, p_height, x.shape)
        
        # conv3_3
        out = self.__get_tensor('conv3_3', batch_size, 256, p_height, p_width, 3, 1, 56, 56)
        p_height, p_width = inc_convolution(self.conv3_2.data, x, self.conv3_3_op[0].weight.data, self.conv3_3_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)                             
        if debug: print(locations, p_height, x.shape)
        
        # pool3
        out = self.__get_tensor('pool3', batch_size, 256, p_height, p_width, 2, 2, 56, 28, 28)
        p_height, p_width = inc_max_pool(self.conv3_3.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out        
        if debug: print(locations, p_height, x.shape)
    
        # conv4_1
        out = self.__get_tensor('conv4_1', batch_size, 512, p_height, p_width, 3, 1, 28, 28)
        p_height, p_width = inc_convolution(self.pool3.data, x, self.conv4_1_op[0].weight.data, self.conv4_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
        
        # conv4_2
        out = self.__get_tensor('conv4_2', batch_size, 512, p_height, p_width, 3, 1, 28, 28)
        p_height, p_width = inc_convolution(self.conv4_1.data, x, self.conv4_2_op[0].weight.data, self.conv4_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
        
        # conv4_3
        out = self.__get_tensor('conv4_3', batch_size, 512, p_height, p_width, 3, 1, 28, 28)
        p_height, p_width = inc_convolution(self.conv4_2.data, x, self.conv4_3_op[0].weight.data, self.conv4_3_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
        
        # pool4
        out = self.__get_tensor('pool4', batch_size, 512, p_height, p_width, 2, 2, 28, 14, 14)
        p_height, p_width = inc_max_pool(self.conv4_3.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out                
        if debug: print(locations, p_height, x.shape)
    
        # conv5_1
        out = self.__get_tensor('conv5_1', batch_size, 512, p_height, p_width, 3, 1, 14, 14)
        p_height, p_width = inc_convolution(self.pool4.data, x, self.conv5_1_op[0].weight.data, self.conv5_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)        
        if debug: print(locations, p_height, x.shape)
            
        # conv5_2
        out = self.__get_tensor('conv5_2', batch_size, 512, p_height, p_width, 3, 1, 14, 14)
        p_height, p_width = inc_convolution(self.conv5_1.data, x, self.conv5_2_op[0].weight.data, self.conv5_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
            
        # conv5_3
        out = self.__get_tensor('conv5_3', batch_size, 512, p_height, p_width, 3, 1, 14, 14)
        p_height, p_width = inc_convolution(self.conv5_2.data, x, self.conv5_3_op[0].weight.data, self.conv5_3_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)

        # pool5
        out = self.__get_tensor('pool5', batch_size, 512, p_height, p_width, 2, 2, 14, 7, 7)
        p_height, p_width = inc_max_pool(self.conv5_3.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out                      
        if debug: print(locations, p_height, x.shape)
        
        #final full-projection
        out = self.__get_tensor('pool5-full', batch_size, 512, 7, 7, 1, 1, 7, 7, truncate=False)
        full_projection(self.pool5.data, x, out, locations, p_height, p_width)
        x = out
        
        return x
                
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def forward_pytorch(self, patches, locations, p_height, p_width, beta=None):
        if not self.initialized:
            raise Exception("Not initialized...")
        
        image = self.image

        if beta is None:
            beta = self.beta
        
        batch_size = patches.shape[0]
        
        if self.gpu:
            patches = patches.cuda()
    
        in_locations = locations.cpu().data.numpy().tolist()

        #FIXME
        patch_size = p_height
          
        layers = [
                  self.conv1_1_inc_op, self.conv1_2_inc_op, self.pool1_op,
                  self.conv2_1_inc_op, self.conv2_2_inc_op, self.pool2_op,
                  self.conv3_1_inc_op, self.conv3_2_inc_op, self.conv3_3_inc_op, self.pool3_op,
                  self.conv4_1_inc_op, self.conv4_2_inc_op, self.conv4_3_inc_op, self.pool4_op,
                  self.conv5_1_inc_op, self.conv5_2_inc_op, self.conv5_3_inc_op, self.pool5_op
                 ]
        
        premat_data = [
                  image, self.conv1_1.data, self.conv1_2.data, self.pool1.data,
                  self.conv2_1.data, self.conv2_2.data, self.pool2.data,
                  self.conv3_1.data, self.conv3_2.data, self.conv3_3.data, self.pool3.data,
                  self.conv4_1.data, self.conv4_2.data, self.conv4_3.data, self.pool4.data,
                  self.conv5_1.data, self.conv5_2.data, self.conv5_3.data
                ]
        
        C = [3, 64, 64, 64, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512]
        sizes = [224, 224, 112, 112, 112, 56, 56, 56, 56, 28, 28, 28, 28, 14, 14, 14, 14, 7]
        S = [1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2]
        P = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
        K = [3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2]

        prev_size = 224
        for layer, data, c, size, s, p, k in zip(layers, premat_data, C, sizes, S, P, K):
            
            remove = 0
            orig_patch_size = patch_size
            if patch_size > round(prev_size*beta):
                temp_patch_size = int(round(prev_size*beta))
                if (patch_size-temp_patch_size)%2 != 0:
                    temp_patch_size -= 1
                remove = patch_size - temp_patch_size
                patch_size = temp_patch_size

                
            out_p_size = int(min(math.ceil((patch_size + k - 1.0)/s), size))
            in_p_size = k + (out_p_size-1)*s
            
            out_locations = self.__get_output_locations(in_locations, out_p_size, s, p, k, prev_size, size, remove=remove)
            
            if layer in self.cache:
                x = self.cache[layer].fill_(0.0)
            else:
                x = torch.FloatTensor(batch_size, c, in_p_size, in_p_size).fill_(0.0)
                self.cache[layer] = x

            if self.gpu:
                x = x.cuda()
            
            for i in range(batch_size):
                x0 = 0 if s*out_locations[i][0]-p >= 0 else -1*(s*out_locations[i][0]-p)
                x1 = min(prev_size - s*out_locations[i][0]+p, in_p_size)
                y0 = 0 if s*out_locations[i][1]-p >= 0 else -1*(s*out_locations[i][1]-p)
                y1 = min(prev_size - s*out_locations[i][1]+p, in_p_size)
        
                temp = data[0,:,:,:].clone()               
                temp[:,in_locations[i][0]:in_locations[i][0]+patch_size,in_locations[i][1]:in_locations[i][1]+patch_size] = patches[i,:,:,:]
      
                x[i,:,x0:x1,y0:y1] = temp[:,max(s*out_locations[i][0]-p,0):max(0, s*out_locations[i][0]-p)+x1-x0,
                     max(0, s*out_locations[i][1]-p):max(0, s*out_locations[i][1]-p)+y1-y0]
                
            patches = layer(x).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
        output = self.pool5.data.repeat(batch_size, 1, 1, 1)
        for i, (x, y) in enumerate(out_locations):
            output[i,:,x:x+out_p_size,y:y+out_p_size] = patches[i,:,:,:]
        
        x = output
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
    def __initialize_weights(self, gpu):
        if self.weights_data is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            weights_data = load_dict_from_hdf5(dir_path + "/vgg16_weights_ptch.h5", gpu)
        else:
            weights_data = self.weights_data
            
        self.conv1_1_op[0].weight.data = weights_data['conv1_1_W:0']
        self.conv1_1_op[0].bias.data = weights_data['conv1_1_b:0']
        self.conv1_1_inc_op[0].weight.data = weights_data['conv1_1_W:0']
        self.conv1_1_inc_op[0].bias.data = weights_data['conv1_1_b:0']
        self.conv1_2_op[0].weight.data = weights_data['conv1_2_W:0']
        self.conv1_2_op[0].bias.data = weights_data['conv1_2_b:0']
        self.conv1_2_inc_op[0].weight.data = weights_data['conv1_2_W:0']
        self.conv1_2_inc_op[0].bias.data = weights_data['conv1_2_b:0']

        self.conv2_1_op[0].weight.data = weights_data['conv2_1_W:0']
        self.conv2_1_op[0].bias.data = weights_data['conv2_1_b:0']
        self.conv2_1_inc_op[0].weight.data = weights_data['conv2_1_W:0']
        self.conv2_1_inc_op[0].bias.data = weights_data['conv2_1_b:0']
        self.conv2_2_op[0].weight.data = weights_data['conv2_2_W:0']
        self.conv2_2_op[0].bias.data = weights_data['conv2_2_b:0']
        self.conv2_2_inc_op[0].weight.data = weights_data['conv2_2_W:0']
        self.conv2_2_inc_op[0].bias.data = weights_data['conv2_2_b:0']

        self.conv3_1_op[0].weight.data = weights_data['conv3_1_W:0']
        self.conv3_1_op[0].bias.data = weights_data['conv3_1_b:0']
        self.conv3_1_inc_op[0].weight.data = weights_data['conv3_1_W:0']
        self.conv3_1_inc_op[0].bias.data = weights_data['conv3_1_b:0']
        self.conv3_2_op[0].weight.data = weights_data['conv3_2_W:0']
        self.conv3_2_op[0].bias.data = weights_data['conv3_2_b:0']
        self.conv3_2_inc_op[0].weight.data = weights_data['conv3_2_W:0']
        self.conv3_2_inc_op[0].bias.data = weights_data['conv3_2_b:0']
        self.conv3_3_op[0].weight.data = weights_data['conv3_3_W:0']
        self.conv3_3_op[0].bias.data = weights_data['conv3_3_b:0']
        self.conv3_3_inc_op[0].weight.data = weights_data['conv3_3_W:0']
        self.conv3_3_inc_op[0].bias.data = weights_data['conv3_3_b:0']

        self.conv4_1_op[0].weight.data = weights_data['conv4_1_W:0']
        self.conv4_1_op[0].bias.data = weights_data['conv4_1_b:0']
        self.conv4_1_inc_op[0].weight.data = weights_data['conv4_1_W:0']
        self.conv4_1_inc_op[0].bias.data = weights_data['conv4_1_b:0']
        self.conv4_2_op[0].weight.data = weights_data['conv4_2_W:0']
        self.conv4_2_op[0].bias.data = weights_data['conv4_2_b:0']
        self.conv4_2_inc_op[0].weight.data = weights_data['conv4_2_W:0']
        self.conv4_2_inc_op[0].bias.data = weights_data['conv4_2_b:0']
        self.conv4_3_op[0].weight.data = weights_data['conv4_3_W:0']
        self.conv4_3_op[0].bias.data = weights_data['conv4_3_b:0']
        self.conv4_3_inc_op[0].weight.data = weights_data['conv4_3_W:0']
        self.conv4_3_inc_op[0].bias.data = weights_data['conv4_3_b:0']

        self.conv5_1_op[0].weight.data = weights_data['conv5_1_W:0']
        self.conv5_1_op[0].bias.data = weights_data['conv5_1_b:0']
        self.conv5_1_inc_op[0].weight.data = weights_data['conv5_1_W:0']
        self.conv5_1_inc_op[0].bias.data = weights_data['conv5_1_b:0']
        self.conv5_2_op[0].weight.data = weights_data['conv5_2_W:0']
        self.conv5_2_op[0].bias.data = weights_data['conv5_2_b:0']
        self.conv5_2_inc_op[0].weight.data = weights_data['conv5_2_W:0']
        self.conv5_2_inc_op[0].bias.data = weights_data['conv5_2_b:0']
        self.conv5_3_op[0].weight.data = weights_data['conv5_3_W:0']
        self.conv5_3_op[0].bias.data = weights_data['conv5_3_b:0']
        self.conv5_3_inc_op[0].weight.data = weights_data['conv5_3_W:0']
        self.conv5_3_inc_op[0].bias.data = weights_data['conv5_3_b:0']

        self.classifier[0].weight.data = weights_data['fc6_W:0']
        self.classifier[0].bias.data = weights_data['fc6_b:0']

        self.classifier[2].weight.data = weights_data['fc7_W:0']
        self.classifier[2].bias.data = weights_data['fc7_b:0']

        self.classifier[4].weight.data = weights_data['fc8_W:0']
        self.classifier[4].bias.data = weights_data['fc8_b:0']

        
    def __get_tensor(self, name, batch_size, channels, p_height, p_width, k_size, stride, in_size, out_size, truncate=True):
        if name in self.tensor_cache:
            return self.tensor_cache[name]
        else:
            tensor = torch.FloatTensor(batch_size, channels, *self.__get_output_shape(p_height, p_width, k_size, stride, in_size, out_size, truncate))
            if self.gpu:
                tensor = tensor.cuda()
            self.tensor_cache[name] = tensor
            return tensor


    def __get_output_shape(self, p_height, p_width, k_size, stride, in_size, out_size, truncate):

        if truncate and (p_height > round(in_size*self.beta)):
            temp_p_height = round(in_size*self.beta)
            
            if ((p_height-temp_p_height) % 2) != 0:
                temp_p_height -= 1
                
            p_height = temp_p_height
                
        new_p_height = min(int(math.ceil((p_height+k_size-1)*1.0/stride)), out_size)
        
        return (new_p_height,new_p_height)
    
    
    def __get_output_locations(self, in_locations, out_p_size, stride, padding, ksize, in_size, out_size, remove=0):
        out_locations = []
        
        for x,y in in_locations:
            x_out = int(max(math.ceil((padding + x + remove/2 - ksize + 1.0)/stride), 0))
            y_out = int(max(math.ceil((padding + y + remove/2 - ksize + 1.0)/stride), 0))
            
            if x_out + out_p_size > out_size:
                x_out = out_size - out_p_size
            if y_out + out_p_size > out_size:
                y_out = out_size - out_p_size
                
            out_locations.append((x_out, y_out))

        return out_locations
    

if __name__ == "__main__":
    batch_size = 64

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')

    images = loader(images)
    images = Variable(images.unsqueeze(0), requires_grad=False, volatile=True).cuda()

    images = images.repeat(batch_size, 1, 1, 1)

    model = VGG16()

    model.eval()
    x = model(images)
    print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])
