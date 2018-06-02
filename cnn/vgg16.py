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

    def __init__(self, beta=1.0, gpu=True):
        super(VGG16, self).__init__()
        self.initialized = False
        self.tensor_cache = {}
        
        self.conv1_1_op = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv1_2_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool1_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv2_2_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool2_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv3_2_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv3_3_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool3_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv4_2_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv4_3_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool4_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv5_2_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.conv5_3_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.pool5_op = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1)
        )

        self.__initialize_weights(gpu)
        self.gpu = gpu
        self.beta = beta

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

    def forward_gpu(self, x, locations, p_height, p_width):
        if not self.initialized:
            raise Exception("Not initialized...")
        
        beta = self.beta
        batch_size = x.shape[0]
        
        if self.gpu:
            x = x.cuda()
            locations = locations.cuda()

        # conv1_1
        out = self.__get_tensor('conv1_1', batch_size, 64, p_height, p_width, 3, 1, 224)
        p_height, p_width = inc_convolution(self.image.data, x, self.conv1_1_op[0].weight.data, self.conv1_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
    
        # conv1_2
        out = self.__get_tensor('conv1_2', batch_size, 64, p_height, p_width, 3, 1, 224)
        p_height, p_width = inc_convolution(self.conv1_1.data, x, self.conv1_2_op[0].weight.data, self.conv1_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # pool1
        out = self.__get_tensor('pool1', batch_size, 64, p_height, p_width, 2, 2, 224)
        p_height, p_width = inc_max_pool(self.conv1_2.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out                                            
        print(locations, p_height)
        
        # conv2_1
        out = self.__get_tensor('conv2_1', batch_size, 128, p_height, p_width, 3, 1, 112)
        p_height, p_width = inc_convolution(self.pool1.data, x, self.conv2_1_op[0].weight.data, self.conv2_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # conv2_2
        out = self.__get_tensor('conv2_2', batch_size, 128, p_height, p_width, 3, 1, 112)
        p_height, p_width = inc_convolution(self.conv2_1.data, x, self.conv2_2_op[0].weight.data, self.conv2_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # pool2
        out = self.__get_tensor('pool2', batch_size, 128, p_height, p_width, 2, 2, 112)
        p_height, p_width = inc_max_pool(self.conv2_2.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out
        print(locations, p_height)
        
        # conv3_1
        out = self.__get_tensor('conv3_1', batch_size, 256, p_height, p_width, 3, 1, 56)
        p_height, p_width = inc_convolution(self.pool2.data, x, self.conv3_1_op[0].weight.data, self.conv3_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # conv3_2
        out = self.__get_tensor('conv3_2', batch_size, 256, p_height, p_width, 3, 1, 56)
        p_height, p_width = inc_convolution(self.conv3_1.data, x, self.conv3_2_op[0].weight.data, self.conv3_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)    
        print(locations, p_height)
        
        # conv3_3
        out = self.__get_tensor('conv3_3', batch_size, 256, p_height, p_width, 3, 1, 56)
        p_height, p_width = inc_convolution(self.conv3_2.data, x, self.conv3_3_op[0].weight.data, self.conv3_3_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)                             
        print(locations, p_height)
        
        # pool3
        out = self.__get_tensor('pool3', batch_size, 256, p_height, p_width, 2, 2, 56)
        p_height, p_width = inc_max_pool(self.conv3_3.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out        
        print(locations, p_height)
        
        # conv4_1
        out = self.__get_tensor('conv4_1', batch_size, 512, p_height, p_width, 3, 1, 28)
        p_height, p_width = inc_convolution(self.pool3.data, x, self.conv4_1_op[0].weight.data, self.conv4_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # conv4_2
        out = self.__get_tensor('conv4_2', batch_size, 512, p_height, p_width, 3, 1, 28)
        p_height, p_width = inc_convolution(self.conv4_1.data, x, self.conv4_2_op[0].weight.data, self.conv4_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # conv4_3
        out = self.__get_tensor('conv4_3', batch_size, 512, p_height, p_width, 3, 1, 28)
        p_height, p_width = inc_convolution(self.conv4_2.data, x, self.conv4_3_op[0].weight.data, self.conv4_3_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # pool4
        out = self.__get_tensor('pool4', batch_size, 512, p_height, p_width, 2, 2, 28)
        p_height, p_width = inc_max_pool(self.conv4_3.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out                
        print(locations, p_height)
        
        # conv5_1
        out = self.__get_tensor('conv5_1', batch_size, 512, p_height, p_width, 3, 1, 14)
        p_height, p_width = inc_convolution(self.pool4.data, x, self.conv5_1_op[0].weight.data, self.conv5_1_op[0].bias.data,
                                            out.data, locations.data, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)        
        print(locations, p_height)
        
        # conv5_2
        out = self.__get_tensor('conv5_2', batch_size, 512, p_height, p_width, 3, 1, 14)
        p_height, p_width = inc_convolution(self.conv5_1.data, x, self.conv5_2_op[0].weight.data, self.conv5_2_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # conv5_3
        out = self.__get_tensor('conv5_3', batch_size, 512, p_height, p_width, 3, 1, 14)
        p_height, p_width = inc_convolution(self.conv5_2.data, x, self.conv5_3_op[0].weight.data, self.conv5_3_op[0].bias.data,
                                            out, locations, 1, 1, 1, 1, p_height, p_width, beta)
        x = F.relu(out)
        print(locations, p_height)
        
        # pool5
        out = self.__get_tensor('pool5', batch_size, 512, p_height, p_width, 2, 2, 14)
        p_height, p_width = inc_max_pool(self.conv5_3.data, x,
                                            out, locations, 0, 0, 2, 2, 2, 2, p_height, p_width, beta)
        x = out                      
        print(locations, p_height)
        
        #final full-projection
        out = self.__get_tensor('pool5-full', batch_size, 512, 7, 7, 1, 1, 7)
        full_projection(self.pool5.data, x, out, locations, p_height, p_width)
        x = out
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x    
    
    def __initialize_weights(self, gpu):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        weights_data = load_dict_from_hdf5(dir_path + "/vgg16_weights_ptch.h5", gpu)

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

        
    def __get_tensor(self, name, batch_size, channels, p_height, p_width, k_size, stride, out_size):
        if name in self.tensor_cache:
            return self.tensor_cache[name]
        else:
            tensor = torch.FloatTensor(batch_size, channels, *self.__get_output_shape(p_height, p_width, k_size, stride, out_size)).cuda()
            self.tensor_cache[name] = tensor
            return tensor


    def __get_output_shape(self, p_height, p_width, k_size, stride, out_size):
        p_height = min(int(math.ceil((p_height+k_size-1)*1.0/stride)), out_size)
        
        if p_height > round(out_size*self.beta):
            p_height = int(math.ceil(p_height*1.0/stride))
        
        return (p_height,p_height)
    

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
