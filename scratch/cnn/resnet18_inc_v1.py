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
import torch.nn.functional as F

from commons import inc_convolution, inc_max_pool
from imagenet_classes import class_names
from resnet18 import ResNet18

class IncrementalResNet18V1(nn.Module):

    def __init__(self, in_tensor, gpu=True, num_classes=1000):
        super(IncrementalResNet18V1, self).__init__()

        # performing initial full inference
        full_model = ResNet18()
        full_model.eval()
        self.gpu = gpu
        
        if self.gpu:
            in_tensor = in_tensor.cuda()
            
        full_model.forward_materialized(in_tensor)
        self.full_model = full_model

        
        self.conv1_op = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1_op = nn.MaxPool2d(kernel_size=3, stride=2)
        
        #layer2
        self.conv2_1_a_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_1_b_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(64))
        
        self.conv2_2_a_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_2_b_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(64))
        
        
        #layer3
        self.conv3_1_a_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3_1_b_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(128))
        
        self.residual_3_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(128))
        
        self.conv3_2_a_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3_2_b_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(128))
        
              
        #layer4
        self.conv4_1_a_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv4_1_b_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(256))

        self.residual_4_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(256))

        
        self.conv4_2_a_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv4_2_b_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(256))
          
        #layer5
        self.conv5_1_a_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv5_1_b_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(512))

        self.residual_5_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(512))

        
        self.conv5_2_a_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv5_2_b_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(512))
        
       
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
        
        
        self.__initialize_weights(gpu)

        self.cache = {}

        
    def forward(self, image, patch, in_locations, patch_size, beta=1.0):
        if self.gpu:
            patch = patch.cuda()
            image = image.cuda()
        
        b = len(in_locations)
        
        patches = patch.repeat(b, 1, 1, 1)

        layers = [self.conv1_op, self.pool1_op]
        premat_data = [image, self.full_model.conv1]
        C = [3, 64]
        sizes = [112, 56]
        S = [2, 2]
        P = [3, 1]
        K = [7, 3]
                
        prev_size = 224
        for layer, data, c, size, s, p, k in zip(layers, premat_data, C, sizes, S, P, K):
            out_p_size = int(min(math.ceil((patch_size + k - 1.0)/s), size))
    
            patch_growing = True
            if out_p_size > round(size*beta):
                out_p_size = int(round(size*beta))
                    
                patch_growing = False
            
            in_p_size = k + (out_p_size-1)*s
            out_locations = self.__get_output_locations(in_locations, out_p_size, s, p, k, prev_size, size, patch_growing)
            
            if layer in self.cache:
                x = self.cache[layer].fill_(0.0)
            else:
                x = torch.FloatTensor(b, c, in_p_size, in_p_size).fill_(0.0)
                self.cache[layer] = x
                
            if self.gpu:
                x = x.cuda()

            for i in range(b):
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
            [self.conv2_1_a_op, self.conv2_1_b_op, None, self.conv2_2_a_op, self.conv2_2_b_op, "merge_2_2"],
            [self.conv3_1_a_op, self.conv3_1_b_op, self.residual_3_op, self.conv3_2_a_op, self.conv3_2_b_op, "merge_3_2"],
            [self.conv4_1_a_op, self.conv4_1_b_op, self.residual_4_op, self.conv4_2_a_op, self.conv4_2_b_op, "merge_4_2"],
            [self.conv5_1_a_op, self.conv5_1_b_op, self.residual_5_op, self.conv5_2_a_op, self.conv5_2_b_op, "merge_5_2"]
        ]
        
        premat_data = [
            [self.full_model.pool1, self.full_model.conv2_1_a, self.full_model.conv2_1_b, self.full_model.merge_2_1, self.full_model.conv2_2_a, self.full_model.conv2_2_b],
            [self.full_model.merge_2_2, self.full_model.conv3_1_a, self.full_model.conv3_1_b, self.full_model.merge_3_1, self.full_model.conv3_2_a, self.full_model.conv3_2_b],
            [self.full_model.merge_3_2, self.full_model.conv4_1_a, self.full_model.conv4_1_b, self.full_model.merge_4_1, self.full_model.conv4_2_a, self.full_model.conv4_2_b],
            [self.full_model.merge_4_2, self.full_model.conv5_1_a, self.full_model.conv5_1_b, self.full_model.merge_5_1, self.full_model.conv5_2_a, self.full_model.conv5_2_b]
        ]
        
        C = [64, 128, 256, 512]
        sizes = [56, 28, 14, 7]
        
        first_layer = True
        
        for sub_layers, data, c, size in zip(layers, premat_data, C, sizes):
           
            r_in_locations = in_locations
            r_patch_size = patch_size
            r_patches = patches
            residual_prev_size = prev_size
            
            if first_layer:
                first_layer = False
                c1 = c
                s1 = 1
            else:
                c1 = c/2
                s1 = 2
                
            x, out_p_size, in_p_size, out_locations = self.__get_patch_sizes(sub_layers[0], patch_size, prev_size, 3, s1, 1, size, beta, in_locations, b, c1)
            for i in range(b):
                x0, x1, y0, y1 = self.__get_patch_coordinates(i, out_locations, s1, 1, in_p_size, prev_size)
                temp = data[0][0,:,:,:].clone()        
                temp[:,in_locations[i][0]:in_locations[i][0]+patch_size,in_locations[i][1]:in_locations[i][1]+patch_size] = patches[i,:,:,:]
                x[i,:,x0:x1,y0:y1] = temp[:,max(s1*out_locations[i][0]-1,0):max(0, s1*out_locations[i][0]-1)+x1-x0,
                     max(0, s1*out_locations[i][1]-1):max(0, s1*out_locations[i][1]-1)+y1-y0]
                    
        
            patches = sub_layers[0](x).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
            
            x, out_p_size, in_p_size, out_locations = self.__get_patch_sizes(sub_layers[1], patch_size, prev_size, 3, 1, 1, size, beta, in_locations, b, c)
            
            for i in range(b):
                x0, x1, y0, y1 = self.__get_patch_coordinates(i, out_locations, 1, 1, in_p_size, prev_size)
                temp = data[1][0,:,:,:].clone()  
                temp[:,in_locations[i][0]:in_locations[i][0]+patch_size,in_locations[i][1]:in_locations[i][1]+patch_size] = patches[i,:,:,:]
                x[i,:,x0:x1,y0:y1] = temp[:,max(out_locations[i][0]-1,0):max(0, out_locations[i][0]-1)+x1-x0,
                     max(0, out_locations[i][1]-1):max(0, out_locations[i][1]-1)+y1-y0]
                    
        
            patches = sub_layers[1](x).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
            if sub_layers[2] is None:
                if sub_layers[2]  in self.cache:
                    x = self.cache[sub_layers[2]].fill_(0.0)
                else:
                    x = torch.FloatTensor(b, c, patch_size, patch_size).fill_(0.0)
                    if self.gpu: x = x.cuda()
                    self.cache[sub_layers[2]] = x  
                    
                for i in range(b):
                    temp = data[0][0,:,:,:].clone()
                    temp[:,r_in_locations[i][0]:r_in_locations[i][0]+r_patch_size,r_in_locations[i][1]:r_in_locations[i][1]+r_patch_size] = r_patches[i,:,:,:]
                    x[i,:,:,:] = temp[:,out_locations[i][0]:out_locations[i][0]+patch_size,out_locations[i][1]:out_locations[i][1]+patch_size]
            else:
                if sub_layers[2] in self.cache:
                    x = self.cache[sub_layers[2]].fill_(0.0)
                else:
                    x = torch.FloatTensor(b, c/2, patch_size*2, patch_size*2).fill_(0.0)
                    if self.gpu: x = x.cuda()
                    self.cache[sub_layers[2]] = x  
                    
                for i in range(b):
                    temp = data[0][0,:,:,:].clone()
                    temp[:,r_in_locations[i][0]:r_in_locations[i][0]+r_patch_size,r_in_locations[i][1]:r_in_locations[i][1]+r_patch_size] = r_patches[i,:,:,:]
                    x[i,:,:,:] = temp[:,2*out_locations[i][0]:2*out_locations[i][0]+patch_size*2,2*out_locations[i][1]:2*out_locations[i][1]+2*patch_size]
                x = sub_layers[2](x)
                        
            patches = F.relu(patches + x)
            
            r_in_locations = in_locations
            r_patch_size = patch_size
            r_patches = patches
            residual_prev_size = prev_size
            
            x, out_p_size, in_p_size, out_locations = self.__get_patch_sizes(sub_layers[3], patch_size, prev_size, 3, 1, 1, size, beta, in_locations, b, c)
            
            for i in range(b):
                x0, x1, y0, y1 = self.__get_patch_coordinates(i, out_locations, 1, 1, in_p_size, prev_size)
                temp = data[3][0,:,:,:].clone()        
                temp[:,in_locations[i][0]:in_locations[i][0]+patch_size,in_locations[i][1]:in_locations[i][1]+patch_size] = patches[i,:,:,:]
                x[i,:,x0:x1,y0:y1] = temp[:,max(out_locations[i][0]-1,0):max(0, out_locations[i][0]-1)+x1-x0,
                     max(0, out_locations[i][1]-1):max(0, out_locations[i][1]-1)+y1-y0]
                    
        
            patches = sub_layers[3](x).data
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
            
            x, out_p_size, in_p_size, out_locations = self.__get_patch_sizes(sub_layers[4], patch_size, prev_size, 3, 1, 1, size, beta, in_locations, b, c)
            
            for i in range(b):
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
                x = torch.FloatTensor(b, c, patch_size, patch_size).fill_(0.0)
                if self.gpu: x = x.cuda()
                self.cache[sub_layers[5]] = x  

            for i in range(b):
                temp = data[3][0,:,:,:].clone()
                temp[:,r_in_locations[i][0]:r_in_locations[i][0]+r_patch_size,r_in_locations[i][1]:r_in_locations[i][1]+r_patch_size] = r_patches[i,:,:,:]
                x[i,:,:,:] = temp[:,out_locations[i][0]:out_locations[i][0]+patch_size,out_locations[i][1]:out_locations[i][1]+patch_size]
            
            patches = F.relu(patches + x)
            in_locations = out_locations
            patch_size = out_p_size
            prev_size = size
            
            
        merge_5_2 = self.full_model.merge_5_2.data.repeat(b, 1, 1, 1)
        for i, (x, y) in enumerate(out_locations):
            merge_5_2[i,:,x:x+out_p_size,y:y+out_p_size] = patches[i,:,:,:]
                    
        x = self.avgpool(merge_5_2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
                
        return x
    
    def __get_patch_coordinates(self, i, out_locations, s, p, in_p_size, prev_size):
        x0 = 0 if s*out_locations[i][0]-p >= 0 else -1*(s*out_locations[i][0]-p)
        x1 = min(prev_size - s*out_locations[i][0]+p, in_p_size)
        y0 = 0 if s*out_locations[i][1]-p >= 0 else -1*(s*out_locations[i][1]-p)
        y1 = min(prev_size - s*out_locations[i][1]+p, in_p_size)
        
        return x0, x1, y0, y1
        
        
    def __get_patch_sizes(self, layer, patch_size, prev_size, k, s, p, size, beta, in_locations, b, c):
        out_p_size = int(min(math.ceil((patch_size + k - 1.0)/s), size))
    
        patch_growing = True
        if out_p_size > round(size*beta):
            out_p_size = int(round(size*beta))

            patch_growing = False

        in_p_size = k + (out_p_size-1)*s
        out_locations = self.__get_output_locations(in_locations, out_p_size, s, p, k, prev_size, size, patch_growing)

        if layer in self.cache:
            x = self.cache[layer].fill_(0.0)
        else:
            x = torch.FloatTensor(b, c, in_p_size, in_p_size).fill_(0.0)
            self.cache[layer] = x
        
        if self.gpu:
            x = x.cuda()
        
        return x, out_p_size, in_p_size, out_locations
        
    def __get_output_locations(self, in_locations, out_p_size, stride, padding, ksize, in_size, out_size, patch_growing=True):
        out_locations = []
        
        for x,y in in_locations:
            if patch_growing:
                x_out = int(max(math.ceil((padding + x - ksize + 1.0)/stride), 0))
                y_out = int(max(math.ceil((padding + y - ksize + 1.0)/stride), 0))
            else:
                x_out = int(round(x*out_size/in_size))
                y_out = int(round(y*out_size/in_size))
            
            if x_out + out_p_size > out_size:
                x_out = out_size - out_p_size
            if y_out + out_p_size > out_size:
                y_out = out_size - out_p_size
                
            out_locations.append((x_out, y_out))
            
        return out_locations
    
            
    def __initialize_weights(self, gpu):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_data = load_dict_from_hdf5(dir_path + "/resnet18_weights_ptch.h5", gpu)
        
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
            
        for mod in self.children():
            if isinstance(mod, nn.Sequential):
                mod[0].weight.data = values[count]
                count += 1
                mod[1].running_mean.data = values[count]
                count += 1
                mod[1].running_var.data = values[count]
                count += 1
                mod[1].weight.data = values[count]
                count += 1
                mod[1].bias.data = values[count]
                count += 1
            elif isinstance(mod, nn.Linear):
                mod.weight.data = values[count]
                count += 1
                mod.bias.data = values[count]
                count += 1
                
                
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
    image = image.unsqueeze(0).cuda()

    full_model = ResNet18().cuda()
    inc_model = IncrementalResNet18V1(image)

    full_model.eval()
    inc_model.eval()
    x = inc_model(image, image_patch, patch_locations, patch_size)
    image[:,:,x_loc[0]:x_loc[0]+patch_size,y_loc[0]:y_loc[0]+patch_size] = image_patch
    y = full_model(image)
    #print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])

    temp = y - x
    print(np.max(np.abs(temp.cpu().data.numpy())))

        