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

from commons import inc_convolution, inc_max_pool, inc_avg_pool, full_projection, batch_normalization, inc_stack
from commons import load_dict_from_hdf5
from imagenet_classes import class_names

class Inception3(nn.Module):

    def __init__(    self, beta=1.0, gpu=True, n_labels=1000, weights_data=None):
        super(Inception3, self).__init__()

        self.gpu = gpu
        self.beta = beta

        # layer1
        self.conv1_op = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2),
                                      nn.BatchNorm2d(32, eps=0.001), nn.ReLU(inplace=True))
        self.conv1_inc_op = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2),
                                      nn.BatchNorm2d(32, eps=0.001), nn.ReLU(inplace=True))
        # layer2
        self.conv2_a_op = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1),
                                        nn.BatchNorm2d(32, eps=0.001), nn.ReLU(inplace=True))
        self.conv2_a_inc_op = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1),
                                        nn.BatchNorm2d(32, eps=0.001), nn.ReLU(inplace=True))
        self.conv2_b_op = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.conv2_b_inc_op = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1),
                                        nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.pool2_op = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool2_inc_op = nn.MaxPool2d(kernel_size=3, stride=2)        

        # layer3
        self.conv3_op = nn.Sequential(nn.Conv2d(64, 80, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(80, eps=0.001), nn.ReLU(inplace=True))
        self.conv3_inc_op = nn.Sequential(nn.Conv2d(64, 80, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(80, eps=0.001), nn.ReLU(inplace=True))

        # layer4
        self.conv4_op = nn.Sequential(nn.Conv2d(80, 192, kernel_size=3, stride=1),
                                      nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.conv4_inc_op = nn.Sequential(nn.Conv2d(80, 192, kernel_size=3, stride=1),
                                      nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.pool4_op = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool4_inc_op = nn.MaxPool2d(kernel_size=3, stride=2)        

        # layer5
        self.mixed_5a = InceptionA(192, 32, beta=beta, gpu=gpu)
        self.mixed_5b = InceptionA(256, 64, beta=beta, gpu=gpu)
        self.mixed_5c = InceptionA(288, 64, beta=beta, gpu=gpu)

        # layer6
        self.mixed_6a = InceptionB(288, beta=beta, gpu=gpu)
        self.mixed_6b = InceptionC(768, c7=128, beta=beta, gpu=gpu)
        self.mixed_6c = InceptionC(768, c7=160, beta=beta, gpu=gpu)
        self.mixed_6d = InceptionC(768, c7=160, beta=beta, gpu=gpu)
        self.mixed_6e = InceptionC(768, c7=192, beta=beta, gpu=gpu)
        
        # layer 7
        self.mixed_7a = InceptionD(768)
        self.mixed_7b = InceptionE(1280)
        self.mixed_7c = InceptionE(2048)

        self.fc = nn.Linear(2048, n_labels)
        self.classifier = nn.Softmax(dim=1)
        self.weights_data = weights_data
        self.__initialize_weights(gpu)
        
        self.tensor_cache = {}        

    def forward(self, x):
        return self.forward_fused(x)

    def forward_fused(self, x):
        x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        x = self.conv1_op(x)
        x = self.conv2_a_op(x)
        x = self.conv2_b_op(x)
        x = self.pool2_op(x)       
        x = self.conv3_op(x)
        x = self.conv4_op(x)      
        x = self.pool4_op(x)       
        
        x = self.mixed_5a(x)
        x = self.mixed_5b(x)   
        x = self.mixed_5c(x)
    
        x = self.mixed_6a(x)        
        x = self.mixed_6b(x)                  
        x = self.mixed_6c(x)
        x = self.mixed_6d(x)
        x = self.mixed_6e(x)
    
        x = self.mixed_7a(x)
        x = self.mixed_7b(x)   
        x = self.mixed_7c(x)
    
        x = F.avg_pool2d(x, kernel_size=8)        
    
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        
        return x

    def forward_materialized(self, x):
        self.initialized = True
        
        x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        if self.gpu:
            x = x.cuda()
        
        self.image = x
        self.conv1 = self.conv1_op(x)
        self.conv2_a = self.conv2_a_op(self.conv1)
        self.conv2_b = self.conv2_b_op(self.conv2_a)
        self.pool2 = self.pool2_op(self.conv2_b)
        self.conv3 = self.conv3_op(self.pool2)
        self.conv4 = self.conv4_op(self.conv3)
        self.pool4 = self.pool4_op(self.conv4)
        
        x = self.mixed_5a.forward_materialized(self.pool4)
        x = self.mixed_5b.forward_materialized(x)
        x = self.mixed_5c.forward_materialized(x)
        
        x = self.mixed_6a.forward_materialized(x)
        x = self.mixed_6b.forward_materialized(x)
        x = self.mixed_6c.forward_materialized(x)
        x = self.mixed_6d.forward_materialized(x)
        x = self.mixed_6e.forward_materialized(x)
    
        x = self.mixed_7a.forward_materialized(x)    
        x = self.mixed_7b.forward_materialized(x)
        x = self.mixed_7c.forward_materialized(x)
    
        x = F.avg_pool2d(x, kernel_size=8)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
        
        x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
 
        if self.gpu:
            x = x.cuda()
            locations = locations.cuda()
        
        # conv1
        out = self.__get_tensor('conv1', batch_size, 32, p_height, p_width, 3, 2, 299, 149)
        p_height, p_width = inc_convolution(self.image.data, x, self.conv1_op[0].weight.data, self.conv1_op[0].bias.data,
                                            out.data, locations.data, 0, 0, 2, 2, p_height, p_width, beta)
        out = batch_normalization(out, self.conv1_op[1].running_mean.data, self.conv1_op[1].running_var.data, self.conv1_op[1].weight.data, self.conv1_op[1].bias.data, eps=1e-3)
        x = F.relu(out)        
        if debug: print(locations, p_height, x.shape)
            
            
        # conv2_a
        out = self.__get_tensor('conv2_a', batch_size, 32, p_height, p_width, 3, 1, 149, 147)
        p_height, p_width = inc_convolution(self.conv1.data, x, self.conv2_a_op[0].weight.data, self.conv2_a_op[0].bias.data,
                                            out.data, locations.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.conv2_a_op[1].running_mean.data, self.conv2_a_op[1].running_var.data, self.conv2_a_op[1].weight.data, self.conv2_a_op[1].bias.data, eps=1e-3)
        x = F.relu(out)        
        if debug: print(locations, p_height, x.shape)            
            
        # conv2_b
        out = self.__get_tensor('conv2_b', batch_size, 64, p_height, p_width, 3, 1, 147, 147)
        p_height, p_width = inc_convolution(self.conv2_a.data, x, self.conv2_b_op[0].weight.data, self.conv2_b_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.conv2_b_op[1].running_mean.data, self.conv2_b_op[1].running_var.data, self.conv2_b_op[1].weight.data, self.conv2_b_op[1].bias.data, eps=1e-3)
        x = F.relu(out)        
        if debug: print(locations, p_height, x.shape)            
            
        # pool2
        out = self.__get_tensor('pool2', batch_size, 64, p_height, p_width, 3, 2, 147, 73)
        p_height, p_width = inc_max_pool(self.conv2_b.data, x,
                                            out, locations, 0, 0, 2, 2, 3, 3, p_height, p_width, beta)
        x = out
        if debug: print(locations, p_height, x.shape)
                     
        # conv3
        out = self.__get_tensor('conv3', batch_size, 80, p_height, p_width, 1, 1, 73, 73)
        p_height, p_width = inc_convolution(self.pool2.data, x, self.conv3_op[0].weight.data, self.conv3_op[0].bias.data,
                                            out.data, locations.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.conv3_op[1].running_mean.data, self.conv3_op[1].running_var.data, self.conv3_op[1].weight.data, self.conv3_op[1].bias.data, eps=1e-3)
        x = F.relu(out)        
        if debug: print(locations, p_height, x.shape)  
            
        # conv4
        out = self.__get_tensor('conv4', batch_size, 192, p_height, p_width, 3, 1, 71, 71)
        p_height, p_width = inc_convolution(self.conv3.data, x, self.conv4_op[0].weight.data, self.conv4_op[0].bias.data,
                                            out.data, locations.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.conv4_op[1].running_mean.data, self.conv4_op[1].running_var.data, self.conv4_op[1].weight.data, self.conv4_op[1].bias.data, eps=1e-3)
        x = F.relu(out)        
        if debug: print(locations, p_height, x.shape)              
    
        # pool4
        out = self.__get_tensor('pool4', batch_size, 192, p_height, p_width, 3, 2, 71, 35)
        p_height, p_width = inc_max_pool(self.conv4.data, x,
                                            out, locations, 0, 0, 2, 2, 3, 3, p_height, p_width, beta)
        x = out
        if debug: print(locations, p_height, x.shape)
   
        x, p_height, p_width = self.mixed_5a.forward_gpu(x, locations, p_height, p_width, beta)        
        x, p_height, p_width = self.mixed_5b.forward_gpu(x, locations, p_height, p_width, beta)      
        x, p_height, p_width = self.mixed_5c.forward_gpu(x, locations, p_height, p_width, beta)
   
        x, p_height, p_width = self.mixed_6a.forward_gpu(x, locations, p_height, p_width, beta)        
        x, p_height, p_width = self.mixed_6b.forward_gpu(x, locations, p_height, p_width, beta)    
        x, p_height, p_width = self.mixed_6c.forward_gpu(x, locations, p_height, p_width, beta)
        x, p_height, p_width = self.mixed_6d.forward_gpu(x, locations, p_height, p_width, beta)
        x, p_height, p_width = self.mixed_6e.forward_gpu(x, locations, p_height, p_width, beta)        
        
        x, p_height, p_width = self.mixed_7a.forward_gpu(x, locations, p_height, p_width, beta)
        x, p_height, p_width = self.mixed_7b.forward_gpu(x, locations, p_height, p_width, beta)
        x, p_height, p_width = self.mixed_7c.forward_gpu(x, locations, p_height, p_width, beta)      

        #final full-projection
        out = self.__get_tensor('global-pool-full', batch_size, 2048, 8, 8, 1, 1, 8, 8, truncate=False)
        full_projection(self.mixed_7c.concat.data, x, out, locations, p_height, p_width)
        x = out
        
        x = F.avg_pool2d(x, kernel_size=8)      

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        
        return x
        


    def __initialize_weights(self, gpu):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if self.weights_data is None:
            weights_data = load_dict_from_hdf5(dir_path + "/inception3_weights_ptch.h5", gpu)
        else:
            weights_data = self.weights_data
        
        keys = [key for key in sorted(weights_data, key=lambda i: int(i.split('.')[0]))]
        values = [weights_data[key] for key in sorted(weights_data, key=lambda i: int(i.split('.')[0]))]

        count = 0
        for mod in [self.conv1_op, self.conv2_a_op, self.conv2_b_op, self.conv3_op, self.conv4_op]:
            mod[0].weight.data = values[count]
            mod[0].bias.data.fill_(0.0)
            count += 1
            mod[1].weight.data = values[count]
            count += 1
            mod[1].bias.data = values[count]
            count += 1
            mod[1].running_mean.data = values[count]
            count += 1
            mod[1].running_var.data = values[count]
            count += 1
            
        count = 0
        for mod in [self.conv1_inc_op, self.conv2_a_inc_op, self.conv2_b_inc_op, self.conv3_inc_op, self.conv4_inc_op]:
            mod[0].weight.data = values[count]
            mod[0].bias.data.fill_(0.0)
            count += 1
            mod[1].weight.data = values[count]
            count += 1
            mod[1].bias.data = values[count]
            count += 1
            mod[1].running_mean.data = values[count]
            count += 1
            mod[1].running_var.data = values[count]
            count += 1

        for mods in [self.mixed_5a.children(), self.mixed_5b.children(), self.mixed_5c.children()]:
            mods = list(mods)
            for mod_count in range(0, len(mods), 2):
                mods[mod_count][0].weight.data = values[count]
                mods[mod_count][0].bias.data.fill_(0.0)
                mods[mod_count+1][0].weight.data = values[count]
                mods[mod_count+1][0].bias.data.fill_(0.0)
                count += 1
                mods[mod_count][1].weight.data = values[count]
                mods[mod_count+1][1].weight.data = values[count]                
                count += 1
                mods[mod_count][1].bias.data = values[count]
                mods[mod_count+1][1].bias.data = values[count]                
                count += 1
                mods[mod_count][1].running_mean.data = values[count]
                mods[mod_count+1][1].running_mean.data = values[count]                
                count += 1
                mods[mod_count][1].running_var.data = values[count]
                mods[mod_count+1][1].running_var.data = values[count]                
                count += 1


        mods = list(self.mixed_6a.children())
        for mod_count in range(0, len(mods), 2):                
            mods[mod_count][0].weight.data = values[count]
            mods[mod_count+1][0].weight.data = values[count]
            mods[mod_count][0].bias.data.fill_(0.0)
            mods[mod_count+1][0].bias.data.fill_(0.0)
            count += 1
            mods[mod_count][1].weight.data = values[count]
            mods[mod_count+1][1].weight.data = values[count]
            count += 1
            mods[mod_count][1].bias.data = values[count]
            mods[mod_count+1][1].bias.data = values[count]
            count += 1
            mods[mod_count][1].running_mean.data = values[count]
            mods[mod_count+1][1].running_mean.data = values[count]
            count += 1
            mods[mod_count][1].running_var.data = values[count]
            mods[mod_count+1][1].running_var.data = values[count]
            count += 1

        for mods in [self.mixed_6b.children(), self.mixed_6c.children(), self.mixed_6d.children(),
                     self.mixed_6e.children()]:
            mods = list(mods)
            for mod_count in range(0, len(mods), 2):
                mods[mod_count][0].weight.data = values[count]
                mods[mod_count][0].bias.data.fill_(0.0)
                mods[mod_count+1][0].weight.data = values[count]
                mods[mod_count+1][0].bias.data.fill_(0.0)
                count += 1
                mods[mod_count][1].weight.data = values[count]
                mods[mod_count+1][1].weight.data = values[count]                
                count += 1
                mods[mod_count][1].bias.data = values[count]
                mods[mod_count+1][1].bias.data = values[count]                
                count += 1
                mods[mod_count][1].running_mean.data = values[count]
                mods[mod_count+1][1].running_mean.data = values[count]                
                count += 1
                mods[mod_count][1].running_var.data = values[count]
                mods[mod_count+1][1].running_var.data = values[count]                
                count += 1
                
        count += 12
        
        mods = list(self.mixed_7a.children())
        for mod_count in range(0, len(mods), 2):
            mods[mod_count][0].weight.data = values[count]
            mods[mod_count][0].bias.data.fill_(0.0)
            mods[mod_count+1][0].weight.data = values[count]
            mods[mod_count+1][0].bias.data.fill_(0.0)            
            count += 1
            mods[mod_count][1].weight.data = values[count]
            mods[mod_count+1][1].weight.data = values[count]            
            count += 1
            mods[mod_count][1].bias.data = values[count]
            mods[mod_count+1][1].bias.data = values[count]            
            count += 1
            mods[mod_count][1].running_mean.data = values[count]
            mods[mod_count+1][1].running_mean.data = values[count]            
            count += 1
            mods[mod_count][1].running_var.data = values[count]
            mods[mod_count+1][1].running_var.data = values[count]            
            count += 1

            
        for mods in [self.mixed_7b.children(), self.mixed_7c.children()]:
            mods = list(mods)
            for mod_count in range(0, len(mods), 2):
                mods[mod_count][0].weight.data = values[count]
                mods[mod_count][0].bias.data.fill_(0.0)
                mods[mod_count+1][0].weight.data = values[count]
                mods[mod_count+1][0].bias.data.fill_(0.0)
                count += 1
                mods[mod_count][1].weight.data = values[count]
                mods[mod_count+1][1].weight.data = values[count]                
                count += 1
                mods[mod_count][1].bias.data = values[count]
                mods[mod_count+1][1].bias.data = values[count]                
                count += 1
                mods[mod_count][1].running_mean.data = values[count]
                mods[mod_count+1][1].running_mean.data = values[count]                
                count += 1
                mods[mod_count][1].running_var.data = values[count]
                mods[mod_count+1][1].running_var.data = values[count]                
                count += 1
    
        self.fc.weight.data = values[count]
        count += 1
        self.fc.bias.data = values[count]
        
    
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
        for mod in self.children():
            mod.reset_tensor_cache()
        
    def __get_output_shape(self, p_height, p_width, k_size, stride, in_size, out_size, truncate):
        temp_p_height = min(int(math.ceil((p_height+k_size-1)*1.0/stride)), out_size)
        
        if truncate and (temp_p_height > round(out_size*self.beta)):
            temp_p_height = round(out_size*self.beta)
            
        return (temp_p_height, temp_p_height)        
        
            
class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, beta=1.0, gpu=True):
        super(InceptionA, self).__init__()
        self.tensor_cache = {}
        
        self.b1_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.b1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))

        self.b5_1_op = nn.Sequential(nn.Conv2d(in_channels, 48, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(48, eps=0.001), nn.ReLU(inplace=True))
        self.b5_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 48, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(48, eps=0.001), nn.ReLU(inplace=True))

        self.b5_2_op = nn.Sequential(nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=2),
                                     nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.b5_2_inc_op = nn.Sequential(nn.Conv2d(48, 64, kernel_size=5, stride=1),
                                     nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))

        self.b3_1_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.b3_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
                                     nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))

        self.b3_2_op = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2_inc_op = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3),
                                     nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))        

        self.b3_3_op = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))
        self.b3_3_inc_op = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3),
                                     nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))

        self.branch_pool_op = nn.Sequential(nn.Conv2d(in_channels, pool_features, kernel_size=1, stride=1),
                                            nn.BatchNorm2d(pool_features, eps=0.001),
                                            nn.ReLU(inplace=True))
        self.branch_pool_inc_op = nn.Sequential(nn.Conv2d(in_channels, pool_features, kernel_size=1, stride=1),
                                            nn.BatchNorm2d(pool_features, eps=0.001),
                                            nn.ReLU(inplace=True))
        
        self.beta = beta
        self.gpu = gpu
        self.in_channels = in_channels
        self.pool_features = pool_features

    def forward(self, x):
        b1 = self.b1_op(x)
        b5 = self.b5_1_op(x)
        b5 = self.b5_2_op(b5)
        b3 = self.b3_1_op(x)
        b3 = self.b3_2_op(b3)
        b3 = self.b3_3_op(b3)        
        b_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        b_pool = self.branch_pool_op(b_pool)
        x = torch.cat([b1, b5, b3, b_pool], 1)
        return x

    def forward_materialized(self, x):
        self.input = x
        self.b1 = self.b1_op(x)
        self.b5_1 = self.b5_1_op(x)
        self.b5_2 = self.b5_2_op(self.b5_1)
        self.b3_1 = self.b3_1_op(x)
        self.b3_2 = self.b3_2_op(self.b3_1)
        self.b3_3 = self.b3_3_op(self.b3_2)
        self.b_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        self.b_pool_2 = self.branch_pool_op(self.b_pool_1)

        self.concat = torch.cat([self.b1, self.b5_2, self.b3_3, self.b_pool_2], 1)
        return self.concat

    def forward_gpu(self, x, locations, p_height, p_width, beta=None):

        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            
        batch_size = x.shape[0]
        
        debug = False
            
        # 1x1
        locations1 = locations.clone()
        out = self.__get_tensor('b1', batch_size, 64, p_height, p_width, 1, 1, 35, 35)
        p_height1, p_width1 = inc_convolution(self.input.data, x, self.b1_op[0].weight.data, self.b1_op[0].bias.data,
                                            out.data, locations1.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b1_op[1].running_mean.data, self.b1_op[1].running_var.data, self.b1_op[1].weight.data, self.b1_op[1].bias.data, eps=1e-3)
        x1 = F.relu(out)        
        if debug: print(locations1, p_height1, x1.shape)

    
        # pool
        locationsp = locations.clone()
        out = self.__get_tensor('pool_1', batch_size, self.in_channels, p_height, p_width, 3, 1, 35, 35)
        p_heightp, p_widthp = inc_avg_pool(self.input.data, x,
                                            out, locationsp, 1, 1, 1, 1, 3, 3, p_height, p_width, beta)
        xp = out
        if debug: print(locationsp, p_heightp, xp.shape)
        
        out = self.__get_tensor('pool_2', batch_size, self.pool_features, p_heightp, p_widthp, 1, 1, 35, 35)
        p_heightp, p_widthp = inc_convolution(self.b_pool_1, xp, self.branch_pool_op[0].weight.data, self.branch_pool_op[0].bias.data, out.data, locationsp.data, 0, 0, 1, 1, p_heightp, p_widthp, beta)
        out = batch_normalization(out, self.branch_pool_op[1].running_mean.data, self.branch_pool_op[1].running_var.data, self.branch_pool_op[1].weight.data, self.branch_pool_op[1].bias.data, eps=1e-3)
        xp = F.relu(out)        
        if debug: print(locationsp, p_heightp, xp.shape)
            

        # 3x3
        locations3 = locations.clone()
        out = self.__get_tensor('b3_1', batch_size, 64, p_height, p_width, 1, 1, 35, 35)
        p_height3, p_width3 = inc_convolution(self.input.data, x, self.b3_1_op[0].weight.data, self.b3_1_op[0].bias.data,
                                            out.data, locations3.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b3_1_op[1].running_mean.data, self.b3_1_op[1].running_var.data, self.b3_1_op[1].weight.data, self.b3_1_op[1].bias.data, eps=1e-3)
        x3 = F.relu(out)        
        if debug: print(locations3, p_height3, x3.shape)

        out = self.__get_tensor('b3_2', batch_size, 96, p_height3, p_width3, 3, 1, 35, 35)
        p_height3, p_width3 = inc_convolution(self.b3_1, x3, self.b3_2_op[0].weight.data, self.b3_2_op[0].bias.data,
                                            out.data, locations3.data, 1, 1, 1, 1, p_height3, p_width3, beta)
        out = batch_normalization(out, self.b3_2_op[1].running_mean.data, self.b3_2_op[1].running_var.data, self.b3_2_op[1].weight.data, self.b3_2_op[1].bias.data, eps=1e-3)
        x3 = F.relu(out)        
        if debug: print(locations3, p_height3, x3.shape)
            
        out = self.__get_tensor('b3_3', batch_size, 96, p_height3, p_width3, 3, 1, 35, 35)
        p_height3, p_width3 = inc_convolution(self.b3_2, x3, self.b3_3_op[0].weight.data, self.b3_3_op[0].bias.data,
                                            out.data, locations3.data, 1, 1, 1, 1, p_height3, p_width3, beta)
        out = batch_normalization(out, self.b3_3_op[1].running_mean.data, self.b3_3_op[1].running_var.data, self.b3_3_op[1].weight.data, self.b3_3_op[1].bias.data, eps=1e-3)
        x3 = F.relu(out)        
        if debug: print(locations3, p_height3, x3.shape)
    
    
        # 5x5
        out = self.__get_tensor('b5_1', batch_size, 48, p_height, p_width, 1, 1, 35, 35)
        p_height5, p_width5 = inc_convolution(self.input.data, x, self.b5_1_op[0].weight.data, self.b5_1_op[0].bias.data,
                                            out.data, locations.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b5_1_op[1].running_mean.data, self.b5_1_op[1].running_var.data, self.b5_1_op[1].weight.data, self.b5_1_op[1].bias.data, eps=1e-3)
        x5 = F.relu(out)        
        if debug: print(locations, p_height5, x5.shape)

        out = self.__get_tensor('b5_2', batch_size, 64, p_height5, p_width5, 5, 1, 35, 35)
        p_height5, p_width5 = inc_convolution(self.b5_1, x5, self.b5_2_op[0].weight.data, self.b5_2_op[0].bias.data,
                                            out.data, locations.data, 2, 2, 1, 1, p_height5, p_width5, beta)
        out = batch_normalization(out, self.b5_2_op[1].running_mean.data, self.b5_2_op[1].running_var.data, self.b5_2_op[1].weight.data, self.b5_2_op[1].bias.data, eps=1e-3)
        x5 = F.relu(out)        
        if debug: print(locations, p_height5, x5.shape)
        
        
        out = self.__get_tensor('stack', batch_size, 224+self.pool_features,
                                p_height5, p_width5, 1, 1, 35, 35, truncate=False)
        
        inc_stack(out.data, 224+self.pool_features, 0, locations, x1, locations1, self.b1.data)
        inc_stack(out.data, 224+self.pool_features, 64, locations, x5, locations, self.b5_2.data)
        inc_stack(out.data, 224+self.pool_features, 64+64, locations, x3, locations3, self.b3_3.data)
        inc_stack(out.data, 224+self.pool_features, 64+64+96, locations, xp, locationsp, self.b_pool_2.data)        
        
        x = out
        return x, p_height5, p_width5
    
    
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
        temp_p_height = min(int(math.ceil((p_height+k_size-1)*1.0/stride)), out_size)
        
        if truncate and (temp_p_height > round(out_size*self.beta)):
            temp_p_height = round(out_size*self.beta)
            
        return (temp_p_height, temp_p_height)

    
class InceptionB(nn.Module):

    def __init__(self, in_channels, beta=1.0, gpu=True):
        super(InceptionB, self).__init__()
        self.b3_op = nn.Sequential(nn.Conv2d(in_channels, 384, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_inc_op = nn.Sequential(nn.Conv2d(in_channels, 384, kernel_size=3, stride=2),
                                   nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))

        self.b3_db_1_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_2_op = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_2_inc_op = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, stride=1),
                                        nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))        
        self.b3_db_3_op = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=2),
                                        nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_3_inc_op = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=2),
                                nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))

        self.in_channels = in_channels
        self.beta = beta
        self.gpu = gpu
        self.tensor_cache = {}

    def forward(self, x):
        b3 = self.b3_op(x)

        b3_db = self.b3_db_1_op(x)
        b3_db = self.b3_db_2_op(b3_db)
        b3_db = self.b3_db_3_op(b3_db)

        b_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)

        outputs = [b3, b3_db, b_pool]
        x = torch.cat(outputs, 1)
        return x
    
    
    def forward_materialized(self, x):
        self.input = x
        
        self.b3 = self.b3_op(x)

        self.b3_db_1 = self.b3_db_1_op(x)
        self.b3_db_2 = self.b3_db_2_op(self.b3_db_1)
        self.b3_db_3 = self.b3_db_3_op(self.b3_db_2)

        self.b_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)

        outputs = [self.b3, self.b3_db_3, self.b_pool]
        self.concat = torch.cat(outputs, 1)
        return self.concat

    def forward_gpu(self, x, locations, p_height, p_width, beta=None):
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            
        batch_size = x.shape[0]
        
        debug = False
        
        # 3x3
        locations3 = locations.clone()
        out = self.__get_tensor('b3', batch_size, 384, p_height, p_width, 3, 2, 35, 17)
        p_height3, p_width3 = inc_convolution(self.input.data, x, self.b3_op[0].weight.data, self.b3_op[0].bias.data,
                                            out.data, locations3.data, 0, 0, 2, 2, p_height, p_width, beta)
        out = batch_normalization(out, self.b3_op[1].running_mean.data, self.b3_op[1].running_var.data, self.b3_op[1].weight.data, self.b3_op[1].bias.data, eps=1e-3)
        x3 = F.relu(out)        
        if debug: print(locations3, p_height3, x3.shape)        
        
        # pool
        locationsp = locations.clone()
        locationsp = locations.clone()
        out = self.__get_tensor('pool', batch_size, self.in_channels, p_height, p_width, 3, 2, 35, 17)
        p_heightp, p_widthp = inc_max_pool(self.input.data, x,
                                            out, locationsp, 0, 0, 2, 2, 3, 3, p_height, p_width, beta)
        xp = out
        if debug: print(locationsp, p_heightp, xp.shape)
                    
        # 3x3_db
        out = self.__get_tensor('b3_db_1', batch_size, 64, p_height, p_width, 1, 1, 35, 35)
        p_height3_db, p_width3_db = inc_convolution(self.input.data, x, self.b3_db_1_op[0].weight.data, self.b3_db_1_op[0].bias.data, out.data, locations.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b3_db_1_op[1].running_mean.data, self.b3_db_1_op[1].running_var.data, self.b3_db_1_op[1].weight.data, self.b3_db_1_op[1].bias.data, eps=1e-3)
        x3_db = F.relu(out)        
        if debug: print(locations3_db, p_height3_db, x3_db.shape)
            
        out = self.__get_tensor('b3_db_2', batch_size, 96, p_height3_db, p_width3_db, 3, 1, 35, 35)
        p_height3_db, p_width3_db = inc_convolution(self.b3_db_1.data, x3_db, self.b3_db_2_op[0].weight.data, self.b3_db_2_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height3_db, p_width3_db, beta)
        out = batch_normalization(out, self.b3_db_2_op[1].running_mean.data, self.b3_db_2_op[1].running_var.data, self.b3_db_2_op[1].weight.data, self.b3_db_2_op[1].bias.data, eps=1e-3)
        x3_db = F.relu(out)        
        if debug: print(locations3_db, p_height3_db, x3_db.shape)    

        out = self.__get_tensor('b3_db_3', batch_size, 96, p_height3_db, p_width3_db, 3, 2, 35, 17)
        p_height3_db, p_width3_db = inc_convolution(self.b3_db_2.data, x3_db, self.b3_db_3_op[0].weight.data, self.b3_db_3_op[0].bias.data, out.data, locations.data, 0, 0, 2, 2, p_height3_db, p_width3_db, beta)
        out = batch_normalization(out, self.b3_db_3_op[1].running_mean.data, self.b3_db_3_op[1].running_var.data, self.b3_db_3_op[1].weight.data, self.b3_db_3_op[1].bias.data, eps=1e-3)
        x3_db = F.relu(out)        
        if debug: print(locations3_db, p_height3_db, x3_db.shape)            
            
        out = self.__get_tensor('stack', batch_size, 480+self.in_channels,
                                p_height3_db, p_width3_db, 1, 1, 17, 17, truncate=False)
        
        inc_stack(out.data,  480+self.in_channels, 0, locations, x3, locations3, self.b3.data)
        inc_stack(out.data,  480+self.in_channels, 384, locations, x3_db, locations, self.b3_db_3.data)
        inc_stack(out.data,  480+self.in_channels, 384+96, locations, xp, locationsp, self.b_pool.data)
        x = out
        
        return x, p_height3_db, p_width3_db
    
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
        temp_p_height = min(int(math.ceil((p_height+k_size-1)*1.0/stride)), out_size)
        
        if truncate and (temp_p_height > round(out_size*self.beta)):
            temp_p_height = round(out_size*self.beta)
            
        return (temp_p_height, temp_p_height)        


class InceptionC(nn.Module):

    def __init__(self, in_channels, c7, beta=1.0, gpu=True):
        super(InceptionC, self).__init__()
        
        self.b1_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        
        
        self.b7_1_op = nn.Sequential(nn.Conv2d(in_channels, c7, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, c7, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_2_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_2_inc_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(1, 7)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_3_op = nn.Sequential(nn.Conv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_3_inc_op = nn.Sequential(nn.Conv2d(c7, 192, kernel_size=(7, 1)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        
        self.b7_db_1_op = nn.Sequential(nn.Conv2d(in_channels, c7, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, c7, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_2_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(7, 1), padding=(3,0)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_2_inc_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(7, 1)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_3_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(1, 7), padding=(0,3)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_3_inc_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(1, 7)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_4_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(7, 1), padding=(3,0)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_4_inc_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(7, 1)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_5_op = nn.Sequential(nn.Conv2d(c7, 192, kernel_size=(1, 7), padding=(0,3)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_5_inc_op = nn.Sequential(nn.Conv2d(c7, 192, kernel_size=(1, 7)),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))

        self.branch_pool_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.branch_pool_inc_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        
        self.in_channels = in_channels
        self.gpu = gpu
        self.beta = beta
        self.c7 = c7
        self.tensor_cache = {}
        

    def forward(self, x):
        b1 = self.b1_op(x)

        b7 = self.b7_1_op(x)           
        b7 = self.b7_2_op(b7)        
        b7 = self.b7_3_op(b7)   
        
        b7_db = self.b7_db_1_op(x)        
        b7_db = self.b7_db_2_op(b7_db)        
        b7_db = self.b7_db_3_op(b7_db)
        b7_db = self.b7_db_4_op(b7_db)
        b7_db = self.b7_db_5_op(b7_db)     
    
        b_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        b_pool = self.branch_pool_op(b_pool)        
        
        outputs = [b1, b7, b7_db, b_pool]
        output = torch.cat(outputs, 1)
        return output

    
    def forward_materialized(self, x):
        self.input = x
        self.b1 = self.b1_op(x)

        self.b7_1 = self.b7_1_op(x)
        self.b7_2 = self.b7_2_op(self.b7_1)
        self.b7_3 = self.b7_3_op(self.b7_2)

        self.b7_db_1 = self.b7_db_1_op(x)
        self.b7_db_2 = self.b7_db_2_op(self.b7_db_1)
        self.b7_db_3 = self.b7_db_3_op(self.b7_db_2)
        self.b7_db_4 = self.b7_db_4_op(self.b7_db_3)
        self.b7_db_5 = self.b7_db_5_op(self.b7_db_4 )

        self.b_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        self.b_pool_2 = self.branch_pool_op(self.b_pool_1)

        outputs = [self.b1, self.b7_3, self.b7_db_5, self.b_pool_2]
        self.concat = torch.cat(outputs, 1)
        return self.concat


    def forward_gpu(self, x, locations, p_height, p_width, beta=None):
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            
        batch_size = x.shape[0]
        
        debug = False        
        
        # 1x1
        locations1 = locations.clone()
        out = self.__get_tensor('b1', batch_size, 192, p_height, p_width, 1, 1, 1, 1, 17, 17)
        p_height1, p_width1 = inc_convolution(self.input.data, x, self.b1_op[0].weight.data, self.b1_op[0].bias.data,
                                            out.data, locations1.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b1_op[1].running_mean.data, self.b1_op[1].running_var.data, self.b1_op[1].weight.data, self.b1_op[1].bias.data, eps=1e-3)
        x1 = F.relu(out)        
        if debug: print(locations1, p_height1, p_width1, x1.shape)
            
            
        # 7x7
        locations7 = locations.clone()
        out = self.__get_tensor('b7_1', batch_size, self.c7, p_height, p_width, 1, 1, 1, 1, 17, 17)
        p_height7, p_width7 = inc_convolution(self.input.data, x, self.b7_1_op[0].weight.data, self.b7_1_op[0].bias.data,
                                            out.data, locations7.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b7_1_op[1].running_mean.data, self.b7_1_op[1].running_var.data, self.b7_1_op[1].weight.data, self.b7_1_op[1].bias.data, eps=1e-3)
        x7 = F.relu(out)        
        if debug: print(locations7, p_height7, p_width7, x7.shape)
            
            
        out = self.__get_tensor('b7_2', batch_size, self.c7, p_height7, p_width7, 1, 7, 1, 1, 17, 17)
        p_height7, p_width7 = inc_convolution(self.b7_1.data, x7, self.b7_2_op[0].weight.data, self.b7_2_op[0].bias.data,
                                            out.data, locations7.data, 0, 3, 1, 1, p_height7, p_width7, beta)
        out = batch_normalization(out, self.b7_2_op[1].running_mean.data, self.b7_2_op[1].running_var.data, self.b7_2_op[1].weight.data, self.b7_2_op[1].bias.data, eps=1e-3)
        x7 = F.relu(out)        
        if debug: print(locations7, p_height7, p_width7, x7.shape)    
                      
                
        out = self.__get_tensor('b7_3', batch_size, 192, p_height7, p_width7, 7, 1, 1, 1, 17, 17)
        p_height7, p_width7 = inc_convolution(self.b7_2.data, x7, self.b7_3_op[0].weight.data, self.b7_3_op[0].bias.data,
                                            out.data, locations7.data, 3, 0, 1, 1, p_height7, p_width7, beta)
        out = batch_normalization(out, self.b7_3_op[1].running_mean.data, self.b7_3_op[1].running_var.data, self.b7_3_op[1].weight.data, self.b7_3_op[1].bias.data, eps=1e-3)
        x7 = F.relu(out)        
        if debug: print(locations7, p_height7, p_width7, x7.shape)            
        
        # pool
        locationsp = locations.clone()
        out = self.__get_tensor('pool_1', batch_size, self.in_channels, p_height, p_width, 3, 3, 1, 1, 17, 17)
        p_heightp, p_widthp = inc_avg_pool(self.input.data, x,
                                            out, locationsp, 1, 1, 1, 1, 3, 3, p_height, p_width, beta)
        xp = out
        if debug: print(locationsp, p_heightp, p_widthp, xp.shape)

        out = self.__get_tensor('pool_2', batch_size, 192, p_heightp, p_widthp, 1, 1, 1, 1, 17, 17)
        p_heightp, p_widthp = inc_convolution(self.b_pool_1.data, xp, self.branch_pool_op[0].weight.data, self.branch_pool_op[0].bias.data, out.data, locationsp.data, 0, 0, 1, 1, p_heightp, p_widthp, beta)
        out = batch_normalization(out, self.branch_pool_op[1].running_mean.data, self.branch_pool_op[1].running_var.data, self.branch_pool_op[1].weight.data, self.branch_pool_op[1].bias.data, eps=1e-3)
        xp = F.relu(out)    
            
        # 7x7 db
        out = self.__get_tensor('b7_db_1', batch_size, self.c7, p_height, p_width, 1, 1, 1, 1, 17, 17)
        p_height7_db, p_width7_db = inc_convolution(self.input.data, x, self.b7_db_1_op[0].weight.data, self.b7_db_1_op[0].bias.data, out.data, locations.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b7_db_1_op[1].running_mean.data, self.b7_db_1_op[1].running_var.data, self.b7_db_1_op[1].weight.data, self.b7_db_1_op[1].bias.data, eps=1e-3)
        x7_db = F.relu(out)        
        if debug: print(locations, p_height7_db, p_width7_db, x7_db.shape)
            
            
        out = self.__get_tensor('b7_db_2', batch_size, self.c7, p_height7_db, p_width7_db, 7, 1, 1, 1, 17, 17)
        p_height7_db, p_width7_db = inc_convolution(self.b7_db_1.data, x7_db, self.b7_db_2_op[0].weight.data, self.b7_db_2_op[0].bias.data, out.data, locations.data, 3, 0, 1, 1, p_height7_db, p_width7_db, beta)
        out = batch_normalization(out, self.b7_db_2_op[1].running_mean.data, self.b7_db_2_op[1].running_var.data, self.b7_db_2_op[1].weight.data, self.b7_db_2_op[1].bias.data, eps=1e-3)
        x7_db = F.relu(out)        
        if debug: print(locations, p_height7_db, p_width7_db, x7_db.shape)
            
            
        out = self.__get_tensor('b7_db_3', batch_size, self.c7, p_height7_db, p_width7_db, 1, 7, 1, 1, 17, 17)
        p_height7_db, p_width7_db = inc_convolution(self.b7_db_2.data, x7_db, self.b7_db_3_op[0].weight.data, self.b7_db_3_op[0].bias.data, out.data, locations.data, 0, 3, 1, 1, p_height7_db, p_width7_db, beta)
        out = batch_normalization(out, self.b7_db_3_op[1].running_mean.data, self.b7_db_3_op[1].running_var.data, self.b7_db_3_op[1].weight.data, self.b7_db_3_op[1].bias.data, eps=1e-3)
        x7_db = F.relu(out)        
        if debug: print(locations, p_height7_db, p_width7_db, x7_db.shape)            
                
        out = self.__get_tensor('b7_db_4', batch_size, self.c7, p_height7_db, p_width7_db, 7, 1, 1, 1, 17, 17)
        p_height7_db, p_width7_db = inc_convolution(self.b7_db_3.data, x7_db, self.b7_db_4_op[0].weight.data, self.b7_db_4_op[0].bias.data, out.data, locations.data, 3, 0, 1, 1, p_height7_db, p_width7_db, beta)
        out = batch_normalization(out, self.b7_db_4_op[1].running_mean.data, self.b7_db_4_op[1].running_var.data, self.b7_db_4_op[1].weight.data, self.b7_db_4_op[1].bias.data, eps=1e-3)
        x7_db = F.relu(out)        
        if debug: print(locations, p_height7_db, p_width7_db, x7_db.shape)

        out = self.__get_tensor('b7_db_5', batch_size, 192, p_height7_db, p_width7_db, 1, 7, 1, 1, 17, 17)
        p_height7_db, p_width7_db = inc_convolution(self.b7_db_4.data, x7_db, self.b7_db_5_op[0].weight.data, self.b7_db_5_op[0].bias.data, out.data, locations.data, 0, 3, 1, 1, p_height7_db, p_width7_db, beta)
        out = batch_normalization(out, self.b7_db_5_op[1].running_mean.data, self.b7_db_5_op[1].running_var.data, self.b7_db_5_op[1].weight.data, self.b7_db_5_op[1].bias.data, eps=1e-3)
        x7_db = F.relu(out)        
        if debug: print(locations, p_height7_db, p_width7_db, x7_db.shape)
         
        
        out = self.__get_tensor('stack', batch_size, 192*4,
                                p_height7_db, p_width7_db, 1, 1, 1, 1, 17, 17, truncate=False)
        
        inc_stack(out.data,  192*4, 0, locations, x1, locations1, self.b1.data)
        inc_stack(out.data,  192*4, 192, locations, x7, locations7, self.b7_3.data)
        inc_stack(out.data,  192*4, 192*2, locations, x7_db, locations, self.b7_db_5.data)
        inc_stack(out.data,  192*4, 192*3, locations, xp, locationsp, self.b_pool_2.data)        
        x = out
        
        return x, p_height7_db, p_width7_db 
    
    
    def __get_tensor(self, name, batch_size, channels, p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate=True):
        if name in self.tensor_cache:
            return self.tensor_cache[name]
        else:
            tensor = torch.FloatTensor(batch_size, channels, *self.__get_output_shape(p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate))
            if self.gpu:
                tensor = tensor.cuda()
            self.tensor_cache[name] = tensor
            return tensor
        
    def reset_tensor_cache(self):
        self.tensor_cache = {}
        
    def __get_output_shape(self, p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate):
        temp_p_height = min(int(math.ceil((p_height+k_size_y-1)*1.0/stride_y)), out_size)
        temp_p_width = min(int(math.ceil((p_width+k_size_x-1)*1.0/stride_x)), out_size)        
        
        if truncate and ((temp_p_height > round(out_size*self.beta)) or (temp_p_width > round(out_size*self.beta))):
            temp_p_height = min(round(out_size*self.beta), temp_p_height)
            temp_p_width = min(round(out_size*self.beta), temp_p_width)
            
        return (temp_p_height, temp_p_width)
    

class InceptionD(nn.Module):

    def __init__(self, in_channels, beta=1.0, gpu=True):
        super(InceptionD, self).__init__()
        self.b3_1_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b3_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2_op = nn.Sequential(nn.Conv2d(192, 320, kernel_size=3, stride=2),
                                     nn.BatchNorm2d(320, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2_inc_op = nn.Sequential(nn.Conv2d(192, 320, kernel_size=3, stride=2),
                                     nn.BatchNorm2d(320, eps=0.001), nn.ReLU(inplace=True))

        
        self.b7_1_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_2_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0,3)),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_2_inc_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=(1, 7)),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_3_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=(7, 1), padding=(3,0)),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_3_inc_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=(7, 1)),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_4_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=2),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_4_inc_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=2),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        
        self.beta = beta
        self.gpu = gpu
        self.tensor_cache = {}
        self.in_channels = in_channels

        
    def forward(self, x):
        b3 = self.b3_1_op(x)
        b3 = self.b3_2_op(b3)

        b7 = self.b7_1_op(x)
        b7 = self.b7_2_op(b7)        
        b7 = self.b7_3_op(b7)
        b7 = self.b7_4_op(b7)    

        b_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)
    
        outputs = [b3, b7, b_pool]
        return torch.cat(outputs, 1)

    
    def forward_materialized(self, x):
        self.input =  x
        self.b3_1 = self.b3_1_op(x)
        self.b3_2 = self.b3_2_op(self.b3_1)

        self.b7_1 = self.b7_1_op(x)
        self.b7_2 = self.b7_2_op(self.b7_1)
        self.b7_3 = self.b7_3_op(self.b7_2)
        self.b7_4 = self.b7_4_op(self.b7_3)

        self.b_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)
        outputs = [self.b3_2, self.b7_4, self.b_pool]
        self.concat = torch.cat(outputs, 1)
        return self.concat


    def forward_gpu(self, x, locations, p_height, p_width, beta=None):
        
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            
        batch_size = x.shape[0]
        
        debug = False
        
        
        # 3x3
        locations3 = locations.clone()
        out = self.__get_tensor('b3_1', batch_size, 192, p_height, p_width, 1, 1, 1, 1, 17, 17)
        p_height3, p_width3 = inc_convolution(self.input.data, x, self.b3_1_op[0].weight.data, self.b3_1_op[0].bias.data,
                                            out.data, locations3.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b3_1_op[1].running_mean.data, self.b3_1_op[1].running_var.data, self.b3_1_op[1].weight.data, self.b3_1_op[1].bias.data, eps=1e-3)
        x3 = F.relu(out)        
        if debug: print(locations3, p_height3, x3.shape)
            
        out = self.__get_tensor('b3_2', batch_size, 320, p_height3, p_width3, 3, 3, 2, 2, 17, 8)
        p_height3, p_width3 = inc_convolution(self.b3_1.data, x3, self.b3_2_op[0].weight.data, self.b3_2_op[0].bias.data,
                                            out.data, locations3.data, 0, 0, 2, 2, p_height3, p_width3, beta)
        out = batch_normalization(out, self.b3_2_op[1].running_mean.data, self.b3_2_op[1].running_var.data, self.b3_2_op[1].weight.data, self.b3_2_op[1].bias.data, eps=1e-3)
        x3 = F.relu(out)        
        if debug: print(locations3, p_height3, x3.shape)
            
        
        # pool
        locationsp = locations.clone()
        out = self.__get_tensor('pool', batch_size, self.in_channels, p_height, p_width, 3, 3, 2, 2, 17, 8)
        p_heightp, p_widthp = inc_max_pool(self.input.data, x,
                                            out, locationsp, 0, 0, 2, 2, 3, 3, p_height, p_width, beta)
        xp = out
        if debug: print(locationsp, p_heightp, xp.shape)
        
        # 7x7
        out = self.__get_tensor('b7_1', batch_size, 192, p_height, p_width, 1, 1, 1, 1, 17, 17)
        p_height7, p_width7 = inc_convolution(self.input.data, x, self.b7_1_op[0].weight.data, self.b7_1_op[0].bias.data,
                                            out.data, locations.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b7_1_op[1].running_mean.data, self.b7_1_op[1].running_var.data, self.b7_1_op[1].weight.data, self.b7_1_op[1].bias.data, eps=1e-3)
        x7 = F.relu(out)        
        if debug: print(locations, p_height7, x7.shape)
    
        out = self.__get_tensor('b7_2', batch_size, 192, p_height7, p_width7, 1, 7, 1, 1, 17, 17)
        p_height7, p_width7 = inc_convolution(self.b7_1.data, x7, self.b7_2_op[0].weight.data, self.b7_2_op[0].bias.data,
                                            out.data, locations.data, 0, 3, 1, 1, p_height7, p_width7, beta)
        out = batch_normalization(out, self.b7_2_op[1].running_mean.data, self.b7_2_op[1].running_var.data, self.b7_2_op[1].weight.data, self.b7_2_op[1].bias.data, eps=1e-3)
        x7 = F.relu(out)        
        if debug: print(locations, p_height7, x7.shape)
            

        out = self.__get_tensor('b7_3', batch_size, 192, p_height7, p_width7, 7, 1, 1, 1, 17, 17)
        p_height7, p_width7 = inc_convolution(self.b7_2.data, x7, self.b7_3_op[0].weight.data, self.b7_3_op[0].bias.data,
                                            out.data, locations.data, 3, 0, 1, 1, p_height7, p_width7, beta)
        out = batch_normalization(out, self.b7_3_op[1].running_mean.data, self.b7_3_op[1].running_var.data, self.b7_3_op[1].weight.data, self.b7_3_op[1].bias.data, eps=1e-3)
        x7 = F.relu(out)        
        if debug: print(locations, p_height7, x7.shape)
            
        out = self.__get_tensor('b7_4', batch_size, 192, p_height7, p_width7, 3, 3, 2, 2, 17, 8)
        p_height7, p_width7 = inc_convolution(self.b7_3.data, x7, self.b7_4_op[0].weight.data, self.b7_4_op[0].bias.data,
                                            out.data, locations.data, 0, 0, 2, 2, p_height7, p_width7, beta)
        out = batch_normalization(out, self.b7_4_op[1].running_mean.data, self.b7_4_op[1].running_var.data, self.b7_4_op[1].weight.data, self.b7_4_op[1].bias.data, eps=1e-3)
        x7 = F.relu(out)        
        if debug: print(locations, p_height7, x7.shape) 
                                 
        out = self.__get_tensor('stack', batch_size, 320+192+self.in_channels,
                                p_height7, p_width7, 1, 1, 1, 1, 8, 8, truncate=False)
        
        inc_stack(out.data,  320+192+self.in_channels, 0, locations, x3, locations3, self.b3_2.data)
        inc_stack(out.data,  320+192+self.in_channels, 320, locations, x7, locations, self.b7_4.data)
        inc_stack(out.data,  320+192+self.in_channels, 320+192, locations, xp, locationsp, self.b_pool.data)      
        x = out
        
        return x, p_height7, p_width7 
    
    def __get_tensor(self, name, batch_size, channels, p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate=True):
        if name in self.tensor_cache:
            return self.tensor_cache[name]
        else:
            tensor = torch.FloatTensor(batch_size, channels, *self.__get_output_shape(p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate))
            if self.gpu:
                tensor = tensor.cuda()
            self.tensor_cache[name] = tensor
            return tensor
        
    def reset_tensor_cache(self):
        self.tensor_cache = {}
        
    def __get_output_shape(self, p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate):    
        temp_p_height = min(int(math.ceil((p_height+k_size_y-1)*1.0/stride_y)), out_size)
        temp_p_width = min(int(math.ceil((p_width+k_size_x-1)*1.0/stride_x)), out_size)        
        
        if truncate and ((temp_p_height > round(out_size*self.beta)) or (temp_p_width > round(out_size*self.beta))):
            temp_p_height = min(round(out_size*self.beta), temp_p_height)
            temp_p_width = min(round(out_size*self.beta), temp_p_width)
            
        return (temp_p_height, temp_p_width)    


class InceptionE(nn.Module):

    def __init__(self, in_channels, beta=1.0, gpu=True):
        super(InceptionE, self).__init__()
        self.b1_op = nn.Sequential(nn.Conv2d(in_channels, 320, kernel_size=1),
                                     nn.BatchNorm2d(320, eps=0.001), nn.ReLU(inplace=True))
        self.b1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 320, kernel_size=1),
                                     nn.BatchNorm2d(320, eps=0.001), nn.ReLU(inplace=True))        
        
        self.b3_1_op = nn.Sequential(nn.Conv2d(in_channels, 384, kernel_size=1),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 384, kernel_size=1),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2a_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1)),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2a_inc_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(1, 3)),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2b_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0)),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2b_inc_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(3, 1)),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))

        
        self.b3_db_1_op =  nn.Sequential(nn.Conv2d(in_channels, 448, kernel_size=1),
                                     nn.BatchNorm2d(448, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_1_inc_op =  nn.Sequential(nn.Conv2d(in_channels, 448, kernel_size=1),
                                     nn.BatchNorm2d(448, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_2_op = nn.Sequential(nn.Conv2d(448, 384, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_2_inc_op = nn.Sequential(nn.Conv2d(448, 384, kernel_size=3),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_3a_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1)),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_3a_inc_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(1, 3)),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_3b_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0)),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_3b_inc_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(3, 1)),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))

        self.branch_pool_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.branch_pool_inc_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        
        self.beta = beta
        self.gpu = gpu
        self.in_channels = in_channels
        self.tensor_cache = {}

        
    def forward(self, x):
        b1 = self.b1_op(x)

        b3 = self.b3_1_op(x)
        b3 = [
            self.b3_2a_op(b3),
            self.b3_2b_op(b3),
        ]
        b3 = torch.cat(b3, 1)

        b3_db = self.b3_db_1_op(x)
        b3_db = self.b3_db_2_op(b3_db)
        b3_db = [
            self.b3_db_3a_op(b3_db),
            self.b3_db_3b_op(b3_db),
        ]
        b3_db = torch.cat(b3_db, 1)

        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        branch_pool = self.branch_pool_op(branch_pool)
    
        outputs = [b1, b3, b3_db, branch_pool]
        return torch.cat(outputs, 1)
    
    def forward_materialized(self, x):
        self.input = x
        self.b1 = self.b1_op(x)

        self.b3_1 = self.b3_1_op(x)
        self.b3_2a = self.b3_2a_op(self.b3_1)
        self.b3_2b = self.b3_2b_op(self.b3_1)
        b3 = [
            self.b3_2a,
            self.b3_2b,
        ]
        self.b3 = torch.cat(b3, 1)

        self.b3_db_1 = self.b3_db_1_op(x)
        self.b3_db_2 = self.b3_db_2_op(self.b3_db_1)
        self.b3_db_3a = self.b3_db_3a_op(self.b3_db_2)
        self.b3_db_3b = self.b3_db_3b_op(self.b3_db_2)

        b3_db = [
            self.b3_db_3a,
            self.b3_db_3b,
        ]
        self.b3_db = torch.cat(b3_db, 1)

        self.b_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)
        self.b_pool_2 = self.branch_pool_op(self.b_pool_1)

        outputs = [self.b1, self.b3, self.b3_db, self.b_pool_2]
        self.concat = torch.cat(outputs, 1)
        return self.concat


    def forward_gpu(self, x, locations, p_height, p_width, beta=None):
        
        if beta is None:
            beta = self.beta
        else:
            self.beta = beta
            
        batch_size = x.shape[0]
        
        debug = False
        
        # 1x1
        locations1 = locations.clone()
        out = self.__get_tensor('b1', batch_size, 320, p_height, p_width, 1, 1, 1, 1, 8, 8)
        p_height1, p_width1 = inc_convolution(self.input.data, x, self.b1_op[0].weight.data, self.b1_op[0].bias.data,
                                            out.data, locations1.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b1_op[1].running_mean.data, self.b1_op[1].running_var.data, self.b1_op[1].weight.data, self.b1_op[1].bias.data, eps=1e-3)
        x1 = F.relu(out)        
        if debug: print(locations1, p_height1, x1.shape)
            
            
        # 3x3
        locations3 = locations.clone()
        out = self.__get_tensor('b3_1', batch_size, 384, p_height, p_width, 1, 1, 1, 1, 8, 8)
        p_height3, p_width3 = inc_convolution(self.input.data, x, self.b3_1_op[0].weight.data, self.b3_1_op[0].bias.data,
                                            out.data, locations3.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b3_1_op[1].running_mean.data, self.b3_1_op[1].running_var.data, self.b3_1_op[1].weight.data, self.b3_1_op[1].bias.data, eps=1e-3)
        x3 = F.relu(out)        
        if debug: print(locations3, p_height3, x3.shape)

        
        locations3_2a = locations3.clone()
        out = self.__get_tensor('b3_2a', batch_size, 384, p_height3, p_width3, 1, 3, 1, 1, 8, 8)
        p_height3_2a, p_width3_2a = inc_convolution(self.b3_1.data, x3, self.b3_2a_op[0].weight.data, self.b3_2a_op[0].bias.data, out.data, locations3_2a.data, 0, 1, 1, 1, p_height3, p_width3, beta)
        out = batch_normalization(out, self.b3_2a_op[1].running_mean.data, self.b3_2a_op[1].running_var.data, self.b3_2a_op[1].weight.data, self.b3_2a_op[1].bias.data, eps=1e-3)
        x3_2a = F.relu(out)        
        if debug: print(locations3_2a, p_height3_2a, x3_2a.shape)
            
        locations3_2b = locations3.clone()
        out = self.__get_tensor('b3_2b', batch_size, 384, p_height3, p_width3, 3, 1, 1, 1, 8, 8)
        p_height3_2b, p_width3_2b = inc_convolution(self.b3_1.data, x3, self.b3_2b_op[0].weight.data, self.b3_2b_op[0].bias.data, out.data, locations3_2b.data, 1, 0, 1, 1, p_height3, p_width3, beta)
        out = batch_normalization(out, self.b3_2b_op[1].running_mean.data, self.b3_2b_op[1].running_var.data, self.b3_2b_op[1].weight.data, self.b3_2b_op[1].bias.data, eps=1e-3)
        x3_2b = F.relu(out)        
        if debug: print(locations3_2b, p_height3_2b, x3_2b.shape)
            
        locations3_2a_temp = locations3_2a.cpu().data.numpy().flatten().tolist()
        locations3_2b_temp = locations3_2b.cpu().data.numpy().flatten().tolist()
        
        for i in range(batch_size):
            locations3[i][0] = min(locations3_2a_temp[2*i], locations3_2b_temp[2*i])
            locations3[i][1] = min(locations3_2a_temp[2*i+1], locations3_2b_temp[2*i+1])            

        
        p_height3, p_width3 = max(p_height3_2a, p_height3_2b), max(p_width3_2a, p_width3_2b)
        
        out = self.__get_tensor('b3_stack', batch_size, 384*2,
                        p_height3, p_width3, 1, 1, 1, 1, 8, 8, truncate=False)
        
        inc_stack(out.data,  384*2, 0, locations3, x3_2a, locations3_2a, self.b3_2a.data)
        inc_stack(out.data,  384*2, 384, locations3, x3_2b, locations3_2b, self.b3_2b.data)
        
        x3 = out

        # pool
        locationsp = locations.clone()
        locationsp = locations.clone()
        out = self.__get_tensor('pool_1', batch_size, self.in_channels, p_height, p_width, 3, 3, 1, 1, 8, 8)
        p_heightp, p_widthp = inc_avg_pool(self.input.data, x,
                                            out, locationsp, 1, 1, 1, 1, 3, 3, p_height, p_width, beta)
        xp = out
        if debug: print(locationsp, p_heightp, xp.shape)

        out = self.__get_tensor('pool_2', batch_size, 192, p_heightp, p_widthp, 1, 1, 1, 1, 8, 8)
        p_heightp, p_widthp = inc_convolution(self.b_pool_1.data, xp, self.branch_pool_op[0].weight.data, self.branch_pool_op[0].bias.data, out.data, locationsp.data, 0, 0, 1, 1, p_heightp, p_widthp, beta)
        out = batch_normalization(out, self.branch_pool_op[1].running_mean.data, self.branch_pool_op[1].running_var.data, self.branch_pool_op[1].weight.data, self.branch_pool_op[1].bias.data, eps=1e-3)
        xp = F.relu(out)    

        # 3x3 db
        out = self.__get_tensor('b3_db_1', batch_size, 448, p_height, p_width, 1, 1, 1, 1, 8, 8)
        p_height3_db, p_width3_db = inc_convolution(self.input.data, x, self.b3_db_1_op[0].weight.data, self.b3_db_1_op[0].bias.data, out.data, locations.data, 0, 0, 1, 1, p_height, p_width, beta)
        out = batch_normalization(out, self.b3_db_1_op[1].running_mean.data, self.b3_db_1_op[1].running_var.data, self.b3_db_1_op[1].weight.data, self.b3_db_1_op[1].bias.data, eps=1e-3)
        x3_db = F.relu(out)        
        if debug: print(locations3_db, p_height3_db, x3_db.shape)
            
        out = self.__get_tensor('b3_db_2', batch_size, 384, p_height3_db, p_width3_db, 3, 3, 1, 1, 8, 8)
        p_height3_db, p_width3_db = inc_convolution(self.b3_db_1.data, x3_db, self.b3_db_2_op[0].weight.data, self.b3_db_2_op[0].bias.data, out.data, locations.data, 1, 1, 1, 1, p_height3_db, p_width3_db, beta)
        out = batch_normalization(out, self.b3_db_2_op[1].running_mean.data, self.b3_db_2_op[1].running_var.data, self.b3_db_2_op[1].weight.data, self.b3_db_2_op[1].bias.data, eps=1e-3)
        x3_db = F.relu(out)        
        if debug: print(locations3_db, p_height3_db, x3_db.shape)   
            
        locations3_db_3a = locations.clone()
        out = self.__get_tensor('b3_db_3a', batch_size, 384, p_height3_db, p_width3_db, 1, 3, 1, 1, 8, 8)
        p_height3_db_3a, p_width3_db_3a = inc_convolution(self.b3_db_2.data, x3_db, self.b3_db_3a_op[0].weight.data, self.b3_db_3a_op[0].bias.data, out.data, locations3_db_3a.data, 0, 1, 1, 1, p_height3_db, p_width3_db, beta)
        out = batch_normalization(out, self.b3_db_3a_op[1].running_mean.data, self.b3_db_3a_op[1].running_var.data, self.b3_db_3a_op[1].weight.data, self.b3_db_3a_op[1].bias.data, eps=1e-3)
        x3_db_3a = F.relu(out)        
        if debug: print(locations3_db_3a, p_height3_db_3a, x3_db_3a.shape)
        
        locations3_db_3b = locations.clone()
        out = self.__get_tensor('b3_db_3b', batch_size, 384, p_height3_db, p_width3_db, 3, 1, 1, 1, 8, 8)
        p_height3_db_3b, p_width3_db_3b = inc_convolution(self.b3_db_2.data, x3_db, self.b3_db_3b_op[0].weight.data, self.b3_db_3b_op[0].bias.data, out.data, locations3_db_3b.data, 1, 0, 1, 1, p_height3_db, p_width3_db, beta)
        out = batch_normalization(out, self.b3_db_3b_op[1].running_mean.data, self.b3_db_3b_op[1].running_var.data, self.b3_db_3b_op[1].weight.data, self.b3_db_3b_op[1].bias.data, eps=1e-3)
        x3_db_3b = F.relu(out)        
        if debug: print(locations3_db_3b, p_height3_db_3b, x3_db_3b.shape)
            
        locations3_db_3a_temp = locations3_db_3a.cpu().data.numpy().flatten().tolist()
        locations3_db_3b_temp = locations3_db_3b.cpu().data.numpy().flatten().tolist()
        
        for i in range(batch_size):
            locations[i][0] = min(locations3_db_3a_temp[2*i], locations3_db_3b_temp[2*i])
            locations[i][1] = min(locations3_db_3a_temp[2*i+1], locations3_db_3b_temp[2*i+1])
        
        p_height3_db, p_width3_db = max(p_height3_db_3a, p_height3_db_3b), max(p_width3_db_3a, p_width3_db_3b)
        
        out = self.__get_tensor('b3_db_stack', batch_size, 384*2,
                        p_height3_db, p_width3_db, 1, 1, 1, 1, 8, 8, truncate=False)
        
        inc_stack(out.data,  384*2, 0, locations, x3_db_3a, locations3_db_3a, self.b3_db_3a.data)
        inc_stack(out.data,  384*2, 384, locations, x3_db_3b, locations3_db_3b, self.b3_db_3b.data)
        x3_db = out
        
        out = self.__get_tensor('stack', batch_size, 320+384*4+192,
                        p_height3_db, p_width3_db, 1, 1, 1, 1, 8, 8, truncate=False)
        inc_stack(out.data,  320+384*4+192, 0, locations, x1, locations1, self.b1.data)
        inc_stack(out.data,  320+384*4+192, 320, locations, x3, locations3, self.b3.data)
        inc_stack(out.data,  320+384*4+192, 320+384*2, locations, x3_db, locations, self.b3_db.data)
        inc_stack(out.data,  320+384*4+192, 320+384*4, locations, xp, locations3_db_3b, self.b_pool_2.data)
        x = out
        
        return x, p_height3_db, p_width3_db
    
    
    def __get_tensor(self, name, batch_size, channels, p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate=True):
        if name in self.tensor_cache:
            return self.tensor_cache[name]
        else:
            tensor = torch.FloatTensor(batch_size, channels, *self.__get_output_shape(p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate))
            if self.gpu:
                tensor = tensor.cuda()
            self.tensor_cache[name] = tensor
            return tensor
        
    def reset_tensor_cache(self):
        self.tensor_cache = {}
        
    def __get_output_shape(self, p_height, p_width, k_size_y, k_size_x, stride_y, stride_x, in_size, out_size, truncate):
        
        temp_p_height = min(int(math.ceil((p_height+k_size_y-1)*1.0/stride_y)), out_size)
        temp_p_width = min(int(math.ceil((p_width+k_size_x-1)*1.0/stride_x)), out_size)        
        
        if truncate and ((temp_p_height > round(out_size*self.beta)) or (temp_p_width > round(out_size*self.beta))):
            temp_p_height = min(round(out_size*self.beta), temp_p_height)
            temp_p_width = min(round(out_size*self.beta), temp_p_width)
            
        return (temp_p_height, temp_p_width)
        