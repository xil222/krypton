#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch.nn as nn
from PIL import Image
import random as rand
import os
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms

from commons import inc_convolution, inc_max_pool, inc_add, batch_normalization, full_projection
from commons import load_dict_from_hdf5
from imagenet_classes import class_names

class ResNet18(nn.Module):

    def __init__(self, dataset, beta=1.0, gpu=True, n_labels=1000, weights_data=None):
        super(ResNet18, self).__init__()
        self.initialized = False
        
        #layer1
        self.conv1_op = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1_inc_op = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1_op = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool1_inc_op = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        #layer2
        self.conv2_1_a_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_1_a_inc_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_1_b_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))
        self.conv2_1_b_inc_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(64))
        
        self.conv2_2_a_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_2_a_inc_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_2_b_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))
        self.conv2_2_b_inc_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(64))
        
        
        #layer3
        self.conv3_1_a_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3_1_a_inc_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3_1_b_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128))
        self.conv3_1_b_inc_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(128))
        
        self.residual_3_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128))
        self.residual_3_inc_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2), nn.BatchNorm2d(128))
        
        self.conv3_2_a_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3_2_a_inc_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3_2_b_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128))
        self.conv3_2_b_inc_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(128))
        
              
        #layer4
        self.conv4_1_a_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv4_1_a_inc_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv4_1_b_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256))
        self.conv4_1_b_inc_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(256))

        self.residual_4_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2), nn.BatchNorm2d(256))
        self.residual_4_inc_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2), nn.BatchNorm2d(256))

        
        self.conv4_2_a_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv4_2_a_inc_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv4_2_b_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256))
        self.conv4_2_b_inc_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(256))
          
        #layer5
        self.conv5_1_a_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv5_1_a_inc_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv5_1_b_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512))
        self.conv5_1_b_inc_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(512))

        self.residual_5_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2), nn.BatchNorm2d(512))
        self.residual_5_inc_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2), nn.BatchNorm2d(512))
        
        self.conv5_2_a_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv5_2_a_inc_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv5_2_b_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512))
        self.conv5_2_b_inc_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0), nn.BatchNorm2d(512))
               
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, n_labels)
        self.classifier = nn.Softmax(dim=1)
        

        self.dataset = dataset
        self.gpu = gpu
        self.beta = beta
        self.weights_data = weights_data
        self.__initialize_weights(gpu)
        
        #used for pytorch based impl.
        self.cache = {}
        #used for cuda impl.
        self.tensor_cache = {}
 

    def forward(self, x):
        return self.forward_fused(x)

    
    def forward_fused(self, x):
        if self.gpu:
            x = x.cuda()
        
        x = self.conv1_op(x)
        x = self.pool1_op(x)
        
        residual = x
        x = self.conv2_1_a_op(x)
        x = self.conv2_1_b_op(x)
        x = F.relu(x + residual)
        residual = x
        x = self.conv2_2_a_op(x)
        x = self.conv2_2_b_op(x)        
        x = F.relu(x + residual)
        
        residual = self.residual_3_op(x)
        x = self.conv3_1_a_op(x)   
        x = self.conv3_1_b_op(x)           
        x = F.relu(x + residual)   
        residual = x
        x = self.conv3_2_a_op(x)
        x = self.conv3_2_b_op(x)
        x = F.relu(x + residual)
        
        residual = self.residual_4_op(x)
        x = self.conv4_1_a_op(x)
        x = self.conv4_1_b_op(x)
        x = F.relu(x + residual)
        residual = x
        x = self.conv4_2_a_op(x)
        x = self.conv4_2_b_op(x)
        x = F.relu(x + residual)        
    
        residual = self.residual_5_op(x)
        x = self.conv5_1_a_op(x)
        x = self.conv5_1_b_op(x)
        x = F.relu(x + residual) 
        residual = x
        x = self.conv5_2_a_op(x)
        x = self.conv5_2_b_op(x)
        x = F.relu(x + residual)        
        #return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x

    
    def forward_materialized(self, x):
        self.initialized = True
        if self.gpu:
            x = x.cuda()

        self.image = x
        self.conv1 = self.conv1_op(x)
        self.pool1 = self.pool1_op(self.conv1)
        
        self.conv2_1_a = self.conv2_1_a_op(self.pool1)
        self.conv2_1_b = self.conv2_1_b_op(self.conv2_1_a)
        self.merge_2_1 = F.relu(self.conv2_1_b + self.pool1)
        self.conv2_2_a = self.conv2_2_a_op(self.merge_2_1)
        self.conv2_2_b = self.conv2_2_b_op(self.conv2_2_a)
        self.merge_2_2 = F.relu(self.conv2_2_b + self.merge_2_1)
        
        self.residual_3 = self.residual_3_op(self.merge_2_2)
        self.conv3_1_a = self.conv3_1_a_op(self.merge_2_2)
        self.conv3_1_b = self.conv3_1_b_op(self.conv3_1_a)
        self.merge_3_1 = F.relu(self.conv3_1_b + self.residual_3)
        self.conv3_2_a = self.conv3_2_a_op(self.merge_3_1)
        self.conv3_2_b = self.conv3_2_b_op(self.conv3_2_a)
        self.merge_3_2 = F.relu(self.conv3_2_b + self.merge_3_1)

        self.residual_4 = self.residual_4_op(self.merge_3_2)
        self.conv4_1_a = self.conv4_1_a_op(self.merge_3_2)
        self.conv4_1_b = self.conv4_1_b_op(self.conv4_1_a)
        self.merge_4_1 = F.relu(self.conv4_1_b + self.residual_4)
        self.conv4_2_a = self.conv4_2_a_op(self.merge_4_1)
        self.conv4_2_b = self.conv4_2_b_op(self.conv4_2_a)
        self.merge_4_2 = F.relu(self.conv4_2_b + self.merge_4_1)

        self.residual_5 = self.residual_5_op(self.merge_4_2)
        self.conv5_1_a = self.conv5_1_a_op(self.merge_4_2)
        self.conv5_1_b = self.conv5_1_b_op(self.conv5_1_a)
        self.merge_5_1 = F.relu(self.conv5_1_b + self.residual_5)
        self.conv5_2_a = self.conv5_2_a_op(self.merge_5_1)
        self.conv5_2_b = self.conv5_2_b_op(self.conv5_2_a)
        self.merge_5_2 = F.relu(self.conv5_2_b + self.merge_5_1)
        
        x = self.avgpool(self.merge_5_2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
        
        layers = [self.conv1_inc_op, self.pool1_inc_op]
        premat_data = [image, self.conv1]
        C = [3, 64]
        sizes = [112, 56]
        S = [2, 2]
        P = [3, 1]
        K = [7, 3]
                
        prev_size = 224
        for layer, data, c, size, s, p, k in zip(layers, premat_data, C, sizes, S, P, K):
            remove = 0
            orig_patch_size = patch_size
            out_p_size = int(min(math.ceil((patch_size + k - 1.0)/s), size))

            #in_p_size = k + (out_p_size-1)*s
            
            #if out_p_size > round(size*beta):
            #    temp_out_p_size = int(round(size*beta))
            #    remove = (out_p_size-temp_out_p_size)*s
            #    in_p_size -= remove
            #    out_p_size = temp_out_p_size
                
            if out_p_size > round(size*beta):
                temp_out_p_size = int(round(size*beta))
                new_patch_size = temp_out_p_size*s-k+1
                remove = patch_size - new_patch_size
                out_p_size = temp_out_p_size
                
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
            
        layers = [
            [self.conv2_1_a_inc_op, self.conv2_1_b_inc_op, None, self.conv2_2_a_inc_op, self.conv2_2_b_inc_op, "merge_2_2"],
            [self.conv3_1_a_inc_op, self.conv3_1_b_inc_op, self.residual_3_inc_op, self.conv3_2_a_inc_op, self.conv3_2_b_inc_op, "merge_3_2"],
            [self.conv4_1_a_inc_op, self.conv4_1_b_inc_op, self.residual_4_inc_op, self.conv4_2_a_inc_op, self.conv4_2_b_inc_op, "merge_4_2"],
            [self.conv5_1_a_inc_op, self.conv5_1_b_inc_op, self.residual_5_inc_op, self.conv5_2_a_inc_op, self.conv5_2_b_inc_op, "merge_5_2"]
        ]
        
        premat_data = [
            [self.pool1, self.conv2_1_a, self.conv2_1_b, self.merge_2_1, self.conv2_2_a, self.conv2_2_b],
            [self.merge_2_2, self.conv3_1_a, self.conv3_1_b, self.merge_3_1, self.conv3_2_a, self.conv3_2_b],
            [self.merge_3_2, self.conv4_1_a, self.conv4_1_b, self.merge_4_1, self.conv4_2_a, self.conv4_2_b],
            [self.merge_4_2, self.conv5_1_a, self.conv5_1_b, self.merge_5_1, self.conv5_2_a, self.conv5_2_b]
        ]
        
        C = [64, 128, 256, 512]
        sizes = [56, 28, 14, 7]
        
        first_layer = True
        
        for sub_layers, data, c, size in zip(layers, premat_data, C, sizes):
            r_in_locations = in_locations            
            r_patches = patches
            residual_prev_size = prev_size
            r_patch_size = patch_size
            
            if first_layer:
                first_layer = False
                c1 = c
                s1 = 1
            else:
                c1 = c/2
                s1 = 2
               
            #1
            x, out_p_size, in_p_size, out_locations = self.__get_patch_sizes(sub_layers[0], patch_size, prev_size, 3, s1, 1, size, beta, in_locations, batch_size, c1)
            
            for i in range(batch_size):
                x0, x1, y0, y1 = self.__get_patch_coordinates(i, out_locations, s1, 1, in_p_size, prev_size)
                temp = data[0][0,:,:,:].clone()        
                temp[:,in_locations[i][0]:in_locations[i][0]+patch_size,in_locations[i][1]:in_locations[i][1]+patch_size] = patches[i,:,:,:]
                x[i,:,x0:x1,y0:y1] = temp[:,max(s1*out_locations[i][0]-1,0):max(0, s1*out_locations[i][0]-1)+x1-x0,
                     max(0, s1*out_locations[i][1]-1):max(0, s1*out_locations[i][1]-1)+y1-y0]
                    
        
            patches = sub_layers[0](x).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
            #2
            x, out_p_size, in_p_size, out_locations = self.__get_patch_sizes(sub_layers[1], patch_size, prev_size, 3, 1, 1, size, beta, in_locations, batch_size, c)
            
            for i in range(batch_size):
                x0, x1, y0, y1 = self.__get_patch_coordinates(i, out_locations, 1, 1, in_p_size, prev_size)
                temp = data[1][0,:,:,:].clone()  
                temp[:,in_locations[i][0]:in_locations[i][0]+patch_size,in_locations[i][1]:in_locations[i][1]+patch_size] = patches[i,:,:,:]
                
                x[i,:,x0:x1,y0:y1] = temp[:,max(out_locations[i][0]-1,0):max(0, out_locations[i][0]-1)+x1-x0,
                     max(0, out_locations[i][1]-1):max(0, out_locations[i][1]-1)+y1-y0]
                    
        
            patches = sub_layers[1](x).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
            #residual
            if sub_layers[2] is None:
                if sub_layers[2]  in self.cache:
                    x = self.cache[sub_layers[2]].fill_(0.0)
                else:
                    x = torch.FloatTensor(batch_size, c, patch_size, patch_size).fill_(0.0)
                    if self.gpu: x = x.cuda()
                    self.cache[sub_layers[2]] = x  
                    
                for i in range(batch_size):
                    temp = data[0][0,:,:,:].clone()
                    temp[:,r_in_locations[i][0]:r_in_locations[i][0]+r_patch_size,r_in_locations[i][1]:+r_in_locations[i][1]+r_patch_size] = r_patches[i,:,:,:]                    
                    x[i,:,:,:] = temp[:,out_locations[i][0]:out_locations[i][0]+patch_size,out_locations[i][1]:out_locations[i][1]+patch_size]
            else:
                if sub_layers[2] in self.cache:
                    x = self.cache[sub_layers[2]].fill_(0.0)
                else:
                    x = torch.FloatTensor(batch_size, c/2, patch_size*2, patch_size*2).fill_(0.0)
                    if self.gpu: x = x.cuda()
                    self.cache[sub_layers[2]] = x  
                    
                for i in range(batch_size):
                    temp = data[0][0,:,:,:].clone()
                    
                    temp[:,r_in_locations[i][0]:r_in_locations[i][0]+r_patch_size,r_in_locations[i][1]:r_in_locations[i][1]+r_patch_size] = r_patches[i,:,:,:]

                    x[i,:,:,:] = temp[:,2*out_locations[i][0]:2*out_locations[i][0]+patch_size*2,2*out_locations[i][1]:2*out_locations[i][1]+2*patch_size]
                    
                x = sub_layers[2](x)
                        
            patches = F.relu(patches + x)
            
            r_in_locations = in_locations
            r_patches = patches
            r_patch_size = patch_size
            residual_prev_size = prev_size
            
            x, out_p_size, in_p_size, out_locations = self.__get_patch_sizes(sub_layers[3], patch_size, prev_size, 3, 1, 1, size, beta, in_locations, batch_size, c)
            
            for i in range(batch_size):
                x0, x1, y0, y1 = self.__get_patch_coordinates(i, out_locations, 1, 1, in_p_size, prev_size)
                temp = data[3][0,:,:,:].clone()        
                temp[:,in_locations[i][0]:in_locations[i][0]+patch_size,in_locations[i][1]:in_locations[i][1]+patch_size] = patches[i,:,:,:]
                x[i,:,x0:x1,y0:y1] = temp[:,max(out_locations[i][0]-1,0):max(0, out_locations[i][0]-1)+x1-x0,
                     max(0, out_locations[i][1]-1):max(0, out_locations[i][1]-1)+y1-y0]
                    
        
            patches = sub_layers[3](x).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
            
            x, out_p_size, in_p_size, out_locations = self.__get_patch_sizes(sub_layers[4], patch_size, prev_size, 3, 1, 1, size, beta, in_locations, batch_size, c)
            
            for i in range(batch_size):
                x0, x1, y0, y1 = self.__get_patch_coordinates(i, out_locations, 1, 1, in_p_size, prev_size)
                temp = data[4][0,:,:,:].clone()  
                temp[:,in_locations[i][0]:in_locations[i][0]+patch_size,in_locations[i][1]:in_locations[i][1]+patch_size] = patches[i,:,:,:]
                x[i,:,x0:x1,y0:y1] = temp[:,max(out_locations[i][0]-1,0):max(0, out_locations[i][0]-1)+x1-x0,
                     max(0, out_locations[i][1]-1):max(0, out_locations[i][1]-1)+y1-y0]
                    
            patches = sub_layers[4](x).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size

            
            if sub_layers[5]  in self.cache:
                x = self.cache[sub_layers[5]].fill_(0.0)
            else:
                x = torch.FloatTensor(batch_size, c, patch_size, patch_size).fill_(0.0)
                if self.gpu: x = x.cuda()
                self.cache[sub_layers[5]] = x  

            for i in range(batch_size):
                temp = data[3][0,:,:,:].clone()
                temp[:,r_in_locations[i][0]:r_in_locations[i][0]+r_patch_size,r_in_locations[i][1]:r_in_locations[i][1]+r_patch_size] = r_patches[i,:,:,:]
                x[i,:,:,:] = temp[:,out_locations[i][0]:out_locations[i][0]+patch_size,out_locations[i][1]:out_locations[i][1]+patch_size]
            
            patches = F.relu(patches + x)
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
            
        merge_5_2 = self.merge_5_2.data.repeat(batch_size, 1, 1, 1)
        for i, (x, y) in enumerate(out_locations):
            merge_5_2[i,:,x:x+out_p_size,y:y+out_p_size] = patches[i,:,:,:]
                    
        x = self.avgpool(merge_5_2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)      
        
        return x
        
        
    def forward_gpu(self, x, locations, p_height, p_width, beta=None):
        
        if not self.initialized:
            raise Exception("Not initialized...")
            
        m = self

        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            
        batch_size = x.shape[0]
        debug = False
        
        if self.gpu:
            x = x.cuda()
            locations = locations.cuda()
        
        # conv1
        out = self.__get_tensor('conv1', batch_size, 64, p_height, p_width, 7, 2, 224, 112)
        p_height, p_width = inc_convolution(self.image.data, x, self.conv1_op[0].weight.data, self.conv1_op[0].bias.data,
                                            out.data, locations.data, 3, 3, 2, 2, p_height, p_width, beta)
        out = batch_normalization(out, self.conv1_op[1].running_mean.data, self.conv1_op[1].running_var.data, self.conv1_op[1].weight.data, self.conv1_op[1].bias.data)
        x = F.relu(out)

        if debug: print(locations, p_height, x.shape)
  
        # pool1
        out = self.__get_tensor('pool1', batch_size, 64, p_height, p_width, 3, 2, 112, 56)
        p_height, p_width = inc_max_pool(self.conv1.data, x,
                                            out, locations, 1, 1, 2, 2, 3, 3, p_height, p_width, beta)
        x = out
        r = x
        r_locations = locations.clone()
        if debug: print(locations, p_height, x.shape)
               
        # conv2
        out = self.__get_tensor('conv2_1_a', batch_size, 64, p_height, p_width, 3, 1, 56, 56)
        p_height, p_width = inc_convolution(self.pool1.data, x, self.conv2_1_a_op[0].weight.data, 
                                            self.conv2_1_a_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv2_1_a_op[1].running_mean.data, self.conv2_1_a_op[1].running_var.data, 
                            self.conv2_1_a_op[1].weight.data, self.conv2_1_a_op[1].bias.data)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)


        out = self.__get_tensor('conv2_1_b', batch_size, 64, p_height, p_width, 3, 1, 56, 56)
        p_height, p_width = inc_convolution(self.conv2_1_a.data, x, self.conv2_1_b_op[0].weight.data, 
                                            self.conv2_1_b_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv2_1_b_op[1].running_mean.data, self.conv2_1_b_op[1].running_var.data, 
                            self.conv2_1_b_op[1].weight.data, self.conv2_1_b_op[1].bias.data)
        x = out
        if debug: print(locations, p_height, x.shape)
        
        
        x = inc_add(x, locations, self.pool1.data, r, r_locations)    
        x = F.relu(x)
        r = x
        r_locations = locations.clone()      
           
        
        out = self.__get_tensor('conv2_2_a', batch_size, 64, p_height, p_width, 3, 1, 56, 56)
        p_height, p_width = inc_convolution(self.merge_2_1.data, x, self.conv2_2_a_op[0].weight.data, 
                                            self.conv2_2_a_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv2_2_a_op[1].running_mean.data, self.conv2_2_a_op[1].running_var.data, 
                            self.conv2_2_a_op[1].weight.data, self.conv2_2_a_op[1].bias.data)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)


        out = self.__get_tensor('conv2_2_b', batch_size, 64, p_height, p_width, 3, 1, 56, 56)
        p_height, p_width = inc_convolution(self.conv2_2_a.data, x, self.conv2_2_b_op[0].weight.data, 
                                            self.conv2_2_b_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv2_2_b_op[1].running_mean.data, self.conv2_2_b_op[1].running_var.data, 
                            self.conv2_2_b_op[1].weight.data, self.conv2_2_b_op[1].bias.data)
        x = out
        if debug: print(locations, p_height, x.shape)                      
            
        x = inc_add(x, locations, self.merge_2_1.data, r, r_locations)    
        x = F.relu(x)
        r = x
        r_locations = locations.clone()   
        r_p_height, r_p_width = p_height, p_width   
        
        
        # conv3
        out = self.__get_tensor('conv3_1_a', batch_size, 128, p_height, p_width, 3, 2, 56, 28)
        p_height, p_width = inc_convolution(self.merge_2_2.data, x, self.conv3_1_a_op[0].weight.data, 
                                            self.conv3_1_a_op[0].bias.data, out.data, locations.data, 1, 1, 2, 2, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv3_1_a_op[1].running_mean.data, self.conv3_1_a_op[1].running_var.data, 
                            self.conv3_1_a_op[1].weight.data, self.conv3_1_a_op[1].bias.data)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
        

        out = self.__get_tensor('conv3_1_b', batch_size, 128, p_height, p_width, 3, 1, 28, 28)
        p_height, p_width = inc_convolution(self.conv3_1_a.data, x, self.conv3_1_b_op[0].weight.data, 
                                            self.conv3_1_b_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv3_1_b_op[1].running_mean.data, self.conv3_1_b_op[1].running_var.data, 
                            self.conv3_1_b_op[1].weight.data, self.conv3_1_b_op[1].bias.data)
        x = out
        if debug: print(locations, p_height, x.shape)            
            
        out = self.__get_tensor('residual3', batch_size, 128, r_p_height, r_p_width, 1, 2, 56, 28)
        _, _ = inc_convolution(self.merge_2_2.data, r, self.residual_3_op[0].weight.data, 
                                            self.residual_3_op[0].bias.data, out.data, r_locations.data, 0, 0, 2, 2, r_p_height, 
                                            r_p_width, beta)
        out = batch_normalization(out, self.residual_3_op[1].running_mean.data, self.residual_3_op[1].running_var.data, self.residual_3_op[1].weight.data, self.residual_3_op[1].bias.data)
        
        r = out 
        
        x = inc_add(x, locations, self.residual_3.data, r, r_locations)    
        x = F.relu(x)
        r = x
        r_locations = locations.clone()    
        
        out = self.__get_tensor('conv3_2_a', batch_size, 128, p_height, p_width, 3, 1, 28, 28)
        p_height, p_width = inc_convolution(self.merge_3_1.data, x, self.conv3_2_a_op[0].weight.data, 
                                            self.conv3_2_a_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv3_2_a_op[1].running_mean.data, self.conv3_2_a_op[1].running_var.data, 
                            self.conv3_2_a_op[1].weight.data, self.conv3_2_a_op[1].bias.data)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)


        out = self.__get_tensor('conv3_2_b', batch_size, 128, p_height, p_width, 3, 1, 28, 28)
        p_height, p_width = inc_convolution(self.conv3_2_a.data, x, self.conv3_2_b_op[0].weight.data, 
                                            self.conv3_2_b_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv3_2_b_op[1].running_mean.data, self.conv3_2_b_op[1].running_var.data, 
                            self.conv3_2_b_op[1].weight.data, self.conv3_2_b_op[1].bias.data)
        x = out
        if debug: print(locations, p_height, x.shape)
            
            
        x = inc_add(x, locations, self.merge_3_1.data, r, r_locations)    
        x = F.relu(x)
        r = x
        r_locations = locations.clone()   
        r_p_height, r_p_width = p_height, p_width
        
        # conv4

        out = self.__get_tensor('conv4_1_a', batch_size, 256, p_height, p_width, 3, 2, 28, 14)
        p_height, p_width = inc_convolution(self.merge_3_2.data, x, self.conv4_1_a_op[0].weight.data, 
                                            self.conv4_1_a_op[0].bias.data, out.data, locations.data, 1, 1, 2, 2, p_height, 
                                            p_width, beta)
        
        out = batch_normalization(out, self.conv4_1_a_op[1].running_mean.data, self.conv4_1_a_op[1].running_var.data, 
                            self.conv4_1_a_op[1].weight.data, self.conv4_1_a_op[1].bias.data)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)
        
        out = self.__get_tensor('conv4_1_b', batch_size, 256, p_height, p_width, 3, 1, 14, 14)
        p_height, p_width = inc_convolution(self.conv4_1_a.data, x, self.conv4_1_b_op[0].weight.data, 
                                            self.conv4_1_b_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv4_1_b_op[1].running_mean.data, self.conv4_1_b_op[1].running_var.data, 
                            self.conv4_1_b_op[1].weight.data, self.conv4_1_b_op[1].bias.data)
        x = out
        if debug: print(locations, p_height, x.shape)

        out = self.__get_tensor('residual4', batch_size, 256, r_p_height, r_p_width, 1, 2, 28, 14)
        _, _ = inc_convolution(self.merge_3_2.data, r, self.residual_4_op[0].weight.data, 
                                            self.residual_4_op[0].bias.data, out.data, r_locations.data, 0, 0, 2, 2, r_p_height, 
                                            r_p_width, beta)
        out = batch_normalization(out, self.residual_4_op[1].running_mean.data, self.residual_4_op[1].running_var.data, self.residual_4_op[1].weight.data, self.residual_4_op[1].bias.data)
            
        r = out
        
        x = inc_add(x, locations, self.residual_4.data, r, r_locations)    
        x = F.relu(x)
        r = x
        r_locations = locations.clone()
        
        out = self.__get_tensor('conv4_2_a', batch_size, 256, p_height, p_width, 3, 1, 14, 14)
        p_height, p_width = inc_convolution(self.merge_4_1.data, x, self.conv4_2_a_op[0].weight.data, 
                                            self.conv4_2_a_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv4_2_a_op[1].running_mean.data, self.conv4_2_a_op[1].running_var.data, 
                            self.conv4_2_a_op[1].weight.data, self.conv4_2_a_op[1].bias.data)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)


        out = self.__get_tensor('conv4_2_b', batch_size, 256, p_height, p_width, 3, 1, 14, 14)
        p_height, p_width = inc_convolution(self.conv4_2_a.data, x, self.conv4_2_b_op[0].weight.data, 
                                            self.conv4_2_b_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv4_2_b_op[1].running_mean.data, self.conv4_2_b_op[1].running_var.data, 
                            self.conv4_2_b_op[1].weight.data, self.conv4_2_b_op[1].bias.data)
        x = out
        if debug: print(locations, p_height, x.shape)
            
            
        x = inc_add(x, locations, self.merge_4_1.data, r, r_locations)    
        x = F.relu(x)
        r = x
        r_locations = locations.clone()   
        r_p_height, r_p_width = p_height, p_width

        
        # conv5
        out = self.__get_tensor('conv5_1_a', batch_size, 512, p_height, p_width, 3, 2, 14, 7)
        p_height, p_width = inc_convolution(self.merge_4_2.data, x, self.conv5_1_a_op[0].weight.data, 
                                            self.conv5_1_a_op[0].bias.data, out.data, locations.data, 1, 1, 2, 2, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv5_1_a_op[1].running_mean.data, self.conv5_1_a_op[1].running_var.data, 
                            self.conv5_1_a_op[1].weight.data, self.conv5_1_a_op[1].bias.data)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)


        out = self.__get_tensor('conv5_1_b', batch_size, 512, p_height, p_width, 3, 1, 7, 7)
        p_height, p_width = inc_convolution(self.conv5_1_a.data, x, self.conv5_1_b_op[0].weight.data, 
                                            self.conv5_1_b_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv5_1_b_op[1].running_mean.data, self.conv5_1_b_op[1].running_var.data, 
                            self.conv5_1_b_op[1].weight.data, self.conv5_1_b_op[1].bias.data)
    
        x = out
        if debug: print(locations, p_height, x.shape)
            
        out = self.__get_tensor('residual5', batch_size, 512, r_p_height, r_p_width, 1, 2, 14, 7)
        _, _ = inc_convolution(self.merge_4_2.data, r, self.residual_5_op[0].weight.data, 
                                            self.residual_5_op[0].bias.data, out.data, r_locations.data, 0, 0, 2, 2, r_p_height, 
                                            r_p_width, beta)
        out = batch_normalization(out, self.residual_5_op[1].running_mean.data, self.residual_5_op[1].running_var.data, self.residual_5_op[1].weight.data, self.residual_5_op[1].bias.data)
        
        r = out
                
        x = inc_add(x, locations, self.residual_5.data, r, r_locations)    
        x = F.relu(x)
        r = x
        r_locations = locations.clone()
        
        
        out = self.__get_tensor('conv5_2_a', batch_size, 512, p_height, p_width, 3, 1, 7, 7)
        p_height, p_width = inc_convolution(self.merge_5_1.data, x, self.conv5_2_a_op[0].weight.data, 
                                            self.conv5_2_a_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv5_2_a_op[1].running_mean.data, self.conv5_2_a_op[1].running_var.data, 
                            self.conv5_2_a_op[1].weight.data, self.conv5_2_a_op[1].bias.data)
        x = F.relu(out)
        if debug: print(locations, p_height, x.shape)


        out = self.__get_tensor('conv5_2_b', batch_size, 512, p_height, p_width, 3, 1, 7, 7)
        p_height, p_width = inc_convolution(self.conv5_2_a.data, x, self.conv5_2_b_op[0].weight.data, 
                                            self.conv5_2_b_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height, 
                                            p_width, beta)
        out = batch_normalization(out, self.conv5_2_b_op[1].running_mean.data, self.conv5_2_b_op[1].running_var.data, 
                            self.conv5_2_b_op[1].weight.data, self.conv5_2_b_op[1].bias.data)
        x = out
        if debug: print(locations, p_height, x.shape)
            
            
        x = inc_add(x, locations, self.merge_5_1.data, r, r_locations)    
        x = F.relu(x)
        r = x
        r_locations = locations.clone()   
        r_p_height, r_p_width = p_height, p_width
        
        
        #final full-projection
        out = self.__get_tensor('merge_5_2-full', batch_size, 512, 7, 7, 1, 1, 7, 7, truncate=False)
        full_projection(self.merge_5_2.data, x, out, locations, p_height, p_width)
        x = out
        #return x
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
    
        return x
      
        
    def __get_patch_coordinates(self, i, out_locations, s, p, in_p_size, prev_size):
        x0 = 0 if s*out_locations[i][0]-p >= 0 else -1*(s*out_locations[i][0]-p)
        x1 = min(prev_size - s*out_locations[i][0]+p, in_p_size)
        y0 = 0 if s*out_locations[i][1]-p >= 0 else -1*(s*out_locations[i][1]-p)
        y1 = min(prev_size - s*out_locations[i][1]+p, in_p_size)
        
        return x0, x1, y0, y1
        
        
    def __get_patch_sizes(self, layer, patch_size, prev_size, k, s, p, size, beta, in_locations, b, c):
        remove = 0
        orig_patch_size = patch_size
        out_p_size = int(min(math.ceil((patch_size + k - 1.0)/s), size))
        
        #in_p_size = k + (out_p_size-1)*s

        #if out_p_size > round(size*beta):
        #    temp_out_p_size = int(round(size*beta))
        #    remove = (out_p_size-temp_out_p_size)*s
        #    in_p_size -= remove
        #    out_p_size = temp_out_p_size
            
        if out_p_size > round(size*beta):
            temp_out_p_size = int(round(size*beta))
            new_patch_size = temp_out_p_size*s-k+1
            remove = patch_size - new_patch_size
            out_p_size = temp_out_p_size

        in_p_size = k + (out_p_size-1)*s            

        
        out_locations = self.__get_output_locations(in_locations, out_p_size, s, p, k, prev_size, size, remove=remove)

        if layer in self.cache:
            x = self.cache[layer].fill_(0.0)
        else:
            x = torch.FloatTensor(b, c, in_p_size, in_p_size).fill_(0.0)
            self.cache[layer] = x
        
        if self.gpu:
            x = x.cuda()
        
        return x, out_p_size, in_p_size, out_locations
    
        
    def __get_output_locations(self, in_locations, out_p_size, stride, padding, ksize, in_size, out_size, remove=0):
        out_locations = []
        
        for x,y in in_locations:
            x_out = int(max(math.ceil((padding + x + remove//2 - ksize + 1.0)/stride), 0))
            y_out = int(max(math.ceil((padding + y + remove//2 - ksize + 1.0)/stride), 0))
            
            if x_out + out_p_size > out_size:
                x_out = out_size - out_p_size
            if y_out + out_p_size > out_size:
                y_out = out_size - out_p_size
                
            out_locations.append((x_out, y_out))
            
        return out_locations
    
    
    def __get_tensor(self, name, batch_size, channels, p_height, p_width, k_size, stride, in_size, out_size, truncate=True):
        if name in self.tensor_cache:
            return self.tensor_cache[name]
        else:
            tensor = torch.FloatTensor(batch_size, channels, *self.__get_output_shape(p_height, p_width, k_size, stride, in_size, out_size, truncate))
            if self.gpu:
                tensor = tensor.cuda()
            self.tensor_cache[name] = tensor
            return tensor
        
    def reset_tensor_cache(self):
        self.tensor_cache = {}

    
    def __get_output_shape(self, p_height, p_width, k_size, stride, in_size, out_size, truncate):            
        new_p_height = min(int(math.ceil((p_height+k_size-1)*1.0/stride)), out_size)
        if truncate and (new_p_height > round(out_size*self.beta)):
            temp_new_p_height = round(out_size*self.beta)
            #if (new_p_height-temp_new_p_height)%2 != 0:
            #    temp_new_p_height -= 1
            new_p_height = temp_new_p_height
    
        return new_p_height, new_p_height
    
    
    def __initialize_weights(self, gpu):
        if self.weights_data is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            weights_data = load_dict_from_hdf5(dir_path + "/resnet18_weights_ptch.h5", gpu)
        else:
            weights_data = self.weights_data

        count = 0
        column_names = [
'conv1_a:w','conv1_a:bn_m','conv1_a:bn_v','conv1_a:bn_w','conv1_a:bn_b',
'conv2_1_a:w','conv2_1_a:bn_m','conv2_1_a:bn_v','conv2_1_a:bn_w','conv2_1_a:bn_b','conv2_1_b:w','conv2_1_b:bn_m','conv2_1_b:bn_v','conv2_1_b:bn_w','conv2_1_b:bn_b',
'conv2_2_a:w','conv2_2_a:bn_m','conv2_2_a:bn_v','conv2_2_a:bn_w','conv2_2_a:bn_b','conv2_2_b:w','conv2_2_b:bn_m','conv2_2_b:bn_v','conv2_2_b:bn_w','conv2_2_b:bn_b',
'conv3_1_a:w','conv3_1_a:bn_m','conv3_1_a:bn_v','conv3_1_a:bn_w','conv3_1_a:bn_b','conv3_1_b:w','conv3_1_b:bn_m','conv3_1_b:bn_v','conv3_1_b:bn_w','conv3_1_b:bn_b',
'residual2:w','residual2:bn_m','residual2:bn_v','residual2:bn_w','residual2:bn_b',
'conv3_2_a:w','conv3_2_a:bn_m','conv3_2_a:bn_v','conv3_2_a:bn_w','conv3_2_a:bn_b','conv3_2_b:w','conv3_2_b:bn_m','conv3_2_b:bn_v','conv3_2_b:bn_w','conv3_2_b:bn_b',
'conv4_1_a:w','conv4_1_a:bn_m','conv4_1_a:bn_v','conv4_1_a:bn_w','conv4_1_a:bn_b','conv4_1_b:w','conv4_1_b:bn_m','conv4_1_b:bn_v','conv4_1_b:bn_w','conv4_1_b:bn_b',
'residual3:w','residual3:bn_m','residual3:bn_v','residual3:bn_w','residual3:bn_b',
'conv4_2_a:w','conv4_2_a:bn_m','conv4_2_a:bn_v','conv4_2_a:bn_w','conv4_2_a:bn_b','conv4_2_b:w','conv4_2_b:bn_m','conv4_2_b:bn_v','conv4_2_b:bn_w','conv4_2_b:bn_b',
'conv5_1_a:w','conv5_1_a:bn_m','conv5_1_a:bn_v','conv5_1_a:bn_w','conv5_1_a:bn_b','conv5_1_b:w','conv5_1_b:bn_m','conv5_1_b:bn_v','conv5_1_b:bn_w','conv5_1_b:bn_b',
'residual4:w','residual4:bn_m','residual4:bn_v','residual4:bn_w','residual4:bn_b',
'conv5_2_a:w','conv5_2_a:bn_m','conv5_2_a:bn_v','conv5_2_a:bn_w','conv5_2_a:bn_b','conv5_2_b:w','conv5_2_b:bn_m','conv5_2_b:bn_v','conv5_2_b:bn_w','conv5_2_b:bn_b',

     'fc:w',
     'fc:b'
   ]
        values = []
        for col_name in column_names:
            values.append(weights_data[col_name])
            
        modules = list(self.children())
        i = 0
        while i < len(modules):
            if isinstance(modules[i], nn.Sequential):
                modules[i][0].weight.data = values[count]
                modules[i+1][0].weight.data = values[count]
                modules[i][0].bias.data.fill_(0.0)
                modules[i+1][0].bias.data.fill_(0.0)
                if gpu:
                    modules[i][0].bias.data = modules[i][0].bias.data.cuda()
                    modules[i+1][0].bias.data = modules[i+1][0].bias.data.cuda()
                count += 1
                
                modules[i][1].running_mean.data = values[count]
                modules[i+1][1].running_mean.data = values[count]               
                count += 1
                
                modules[i][1].running_var.data = values[count]
                modules[i+1][1].running_var.data = values[count]
                count += 1
                
                modules[i][1].weight.data = values[count]
                modules[i+1][1].weight.data = values[count]            
                count += 1
                
                modules[i][1].bias.data = values[count]
                modules[i+1][1].bias.data = values[count]
                count += 1
                
                i += 2
            elif isinstance(modules[i], nn.MaxPool2d):
                i += 2
            elif isinstance(modules[i], nn.Linear):
                modules[i].weight.data = values[count]
                count += 1
                
                modules[i].bias.data = values[count]
                count += 1
                
                i += 1
            else:
                i += 1
        
        if self.dataset == 'oct':
            weights_data = load_dict_from_hdf5(dir_path + "/oct_resnet18_ptch.h5", gpu)
            self.fc.weight.data = weights_data['fc:w']
            self.fc.bias.data = weights_data['fc:b']
                
                
if __name__ == '__main__':
    batch_size = 1
    patch_size = 64
    input_size = 224
    
    patch_locations = torch.cuda.IntTensor(1, 2)
    patch_locations[0, 0] = (224-patch_size)//2
    patch_locations[0, 1] = (224-patch_size)//2
    
    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    image = Image.open('./dog_resized.jpg')
    image = loader(image)
    image = image.unsqueeze(0).cuda()
    
    image_patch = image[:, :, (224-patch_size)//2:(224-patch_size)//2+patch_size, (224-patch_size)//2:(224-patch_size)//2+patch_size]
    
    model1 = ResNet18().eval().cuda()
    model2 = ResNet18().eval().cuda()
    
    model1.forward_materialized(image)
    x = model1.forward_gpu(image_patch, patch_locations, patch_size, patch_size, beta=1.0)

    y = model2(image)
    
    #print(class_names[np.argmax(y.data.cpu().numpy()[0, :])])

    temp = y - x
    print(np.max(np.abs(temp.cpu().data.numpy())))
                
