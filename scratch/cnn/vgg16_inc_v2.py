#!/usr/bin/env python
# coding=utf-8

import random

import numpy as np
import torch
import math
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms

from commons import inc_convolution, inc_convolution2, inc_max_pool, inc_max_pool2, final_full_projection
from cnn.vgg16 import VGG16


class IncrementalVGG16V2(nn.Module):
    
    def __init__(self, in_tensor, beta=1.0, cuda=True):
        super(IncrementalVGG16V2, self).__init__()
    
        self.tensor_cache = {}
            
        # performing initial full inference
        full_model = VGG16()
        #full_model.eval()
        self.cuda = cuda
        if self.cuda:
            in_tensor = in_tensor.cuda()

        self.initial_result = full_model.forward_materialized(in_tensor).cpu().data.numpy()
        self.full_model = full_model
        self.beta = beta

        torch.cuda.empty_cache()

    def forward(self, x, locations, p_height, p_width):
        return self.forward_gpu(x, locations, p_height, p_width)

    def forward_gpu(self, x, locations, p_height, p_width):
        m = self.full_model
        beta = self.beta

        if self.cuda:
            x = x.cuda()
            locations = locations.cuda()

        # conv1_1
        p_height, p_width = inc_convolution(x, m.conv1_1_op[0].weight.data, m.conv1_1_op[0].bias.data,
                                            m.conv1_1.data, locations, 1, 1, p_height, p_width, beta)
        
        # conv1_2
        p_height, p_width = inc_convolution(m.conv1_1.data, m.conv1_2_op[0].weight.data, m.conv1_2_op[0].bias.data,
                                            m.conv1_2.data, locations, 1, 1, p_height, p_width, beta)
        
        # pool1
        p_height, p_width = inc_max_pool(m.conv1_2.data, m.pool1.data, locations, 0, 2, 2, p_height, p_width, beta)

        # conv2_1
        p_height, p_width = inc_convolution(m.pool1.data, m.conv2_1_op[0].weight.data, m.conv2_1_op[0].bias.data,
                                            m.conv2_1.data, locations, 1, 1, p_height, p_width, beta)

        # conv2_2
        p_height, p_width = inc_convolution(m.conv2_1.data, m.conv2_2_op[0].weight.data, m.conv2_2_op[0].bias.data,
                                            m.conv2_2.data, locations, 1, 1, p_height, p_width, beta)

        # pool2
        p_height, p_width = inc_max_pool(m.conv2_2.data, m.pool2.data, locations,
                                         0, 2, 2, p_height, p_width, beta)

        # conv3_1
        p_height, p_width = inc_convolution(m.pool2.data, m.conv3_1_op[0].weight.data,
                                            m.conv3_1_op[0].bias.data, m.conv3_1.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # conv3_2
        p_height, p_width = inc_convolution(m.conv3_1.data, m.conv3_2_op[0].weight.data,
                                            m.conv3_2_op[0].bias.data, m.conv3_2.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # conv3_3
        p_height, p_width = inc_convolution(m.conv3_2.data, m.conv3_3_op[0].weight.data,
                                            m.conv3_3_op[0].bias.data, m.conv3_3.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # pool3
        p_height, p_width = inc_max_pool(m.conv3_3.data, m.pool3.data, locations,
                                         0, 2, 2, p_height, p_width, beta)

        # conv4_1
        p_height, p_width = inc_convolution(m.pool3.data, m.conv4_1_op[0].weight.data,
                                            m.conv4_1_op[0].bias.data, m.conv4_1.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # conv4_2
        p_height, p_width = inc_convolution(m.conv4_1.data, m.conv4_2_op[0].weight.data,
                                            m.conv4_2_op[0].bias.data, m.conv4_2.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # conv4_3
        p_height, p_width = inc_convolution(m.conv4_2.data, m.conv4_3_op[0].weight.data,
                                            m.conv4_3_op[0].bias.data, m.conv4_3.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # pool4
        p_height, p_width = inc_max_pool(m.conv4_3.data, m.pool4.data, locations,
                                         0, 2, 2, p_height, p_width, beta)

        # conv5_1
        p_height, p_width = inc_convolution(m.pool4.data, m.conv5_1_op[0].weight.data,
                                            m.conv5_1_op[0].bias.data, m.conv5_1.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # conv5_2
        p_height, p_width = inc_convolution(m.conv5_1.data, m.conv5_2_op[0].weight.data,
                                            m.conv5_2_op[0].bias.data, m.conv5_2.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # conv5_3
        p_height, p_width = inc_convolution(m.conv5_2.data, m.conv5_3_op[0].weight.data,
                                            m.conv5_3_op[0].bias.data, m.conv5_3.data,
                                            locations, 1, 1, p_height, p_width, beta)

        # pool5
        inc_max_pool(m.conv5_3.data, m.pool5.data, locations,
                     0, 2, 2, p_height, p_width, beta)

        x = m.pool5.data
        x = x.view(x.size(0), -1)
        x = m.classifier(x)
        return x


    def forward_gpu2(self, x, locations, p_height, p_width):
        m = self.full_model
        beta = self.beta
        batch_size = x.shape[0]
        
        if self.cuda:
            x = x.cuda()
            locations = locations.cuda()

        # conv1_1
        out = self.__get_tensor('conv1_1', batch_size, 64, p_height, p_width, 3, 1, 224)
        p_height, p_width = inc_convolution2(m.image.data, x, m.conv1_1_op[0].weight.data, m.conv1_1_op[0].bias.data,
                                            out.data, locations.data, 224, 1, 1, p_height, p_width, beta)
        x = out
    
        # conv1_2
        out = self.__get_tensor('conv1_2', batch_size, 64, p_height, p_width, 3, 1, 224)
        p_height, p_width = inc_convolution2(m.conv1_1.data, x, m.conv1_2_op[0].weight.data, m.conv1_2_op[0].bias.data,
                                            out, locations, 224, 1, 1, p_height, p_width, beta)
        x = out
    
        # pool1
        out = self.__get_tensor('pool1', batch_size, 64, p_height, p_width, 2, 2, 224)
        p_height, p_width = inc_max_pool2(m.conv1_2.data, x,
                                            out, locations, 112, 0, 2, 2, p_height, p_width, beta)
        x = out                                            
    
        # conv2_1
        out = self.__get_tensor('conv2_1', batch_size, 128, p_height, p_width, 3, 1, 112)
        p_height, p_width = inc_convolution2(m.pool1.data, x, m.conv2_1_op[0].weight.data, m.conv2_1_op[0].bias.data,
                                            out.data, locations.data, 112, 1, 1, p_height, p_width, beta)
        x = out
        
        # conv2_2
        out = self.__get_tensor('conv2_2', batch_size, 128, p_height, p_width, 3, 1, 112)
        p_height, p_width = inc_convolution2(m.conv2_1.data, x, m.conv2_2_op[0].weight.data, m.conv2_2_op[0].bias.data,
                                            out, locations, 112, 1, 1, p_height, p_width, beta)
        x = out
    
        # pool2
        out = self.__get_tensor('pool2', batch_size, 128, p_height, p_width, 2, 2, 112)
        p_height, p_width = inc_max_pool2(m.conv2_2.data, x,
                                            out, locations, 56, 0, 2, 2, p_height, p_width, beta)
        x = out
    
        # conv3_1
        out = self.__get_tensor('conv3_1', batch_size, 256, p_height, p_width, 3, 1, 56)
        p_height, p_width = inc_convolution2(m.pool2.data, x, m.conv3_1_op[0].weight.data, m.conv3_1_op[0].bias.data,
                                            out.data, locations.data, 56, 1, 1, p_height, p_width, beta)
        x = out
        
        # conv3_2
        out = self.__get_tensor('conv3_2', batch_size, 256, p_height, p_width, 3, 1, 56)
        p_height, p_width = inc_convolution2(m.conv3_1.data, x, m.conv3_2_op[0].weight.data, m.conv3_2_op[0].bias.data,
                                            out, locations, 56, 1, 1, p_height, p_width, beta)
        x = out    
        
        # conv3_3
        out = self.__get_tensor('conv3_3', batch_size, 256, p_height, p_width, 3, 1, 56)
        p_height, p_width = inc_convolution2(m.conv3_2.data, x, m.conv3_3_op[0].weight.data, m.conv3_3_op[0].bias.data,
                                            out, locations, 56, 1, 1, p_height, p_width, beta)
        x = out                             
    
        # pool3
        out = self.__get_tensor('pool3', batch_size, 256, p_height, p_width, 2, 2, 56)
        p_height, p_width = inc_max_pool2(m.conv3_3.data, x,
                                            out, locations, 28, 0, 2, 2, p_height, p_width, beta)
        x = out        
        
        # conv4_1
        out = self.__get_tensor('conv4_1', batch_size, 512, p_height, p_width, 3, 1, 28)
        p_height, p_width = inc_convolution2(m.pool3.data, x, m.conv4_1_op[0].weight.data, m.conv4_1_op[0].bias.data,
                                            out.data, locations.data, 28, 1, 1, p_height, p_width, beta)
        x = out
        
        # conv4_2
        out = self.__get_tensor('conv4_2', batch_size, 512, p_height, p_width, 3, 1, 28)
        p_height, p_width = inc_convolution2(m.conv4_1.data, x, m.conv4_2_op[0].weight.data, m.conv4_2_op[0].bias.data,
                                            out, locations, 28, 1, 1, p_height, p_width, beta)
        x = out
        
        # conv4_3
        out = self.__get_tensor('conv4_3', batch_size, 512, p_height, p_width, 3, 1, 28)
        p_height, p_width = inc_convolution2(m.conv4_2.data, x, m.conv4_3_op[0].weight.data, m.conv4_3_op[0].bias.data,
                                            out, locations, 28, 1, 1, p_height, p_width, beta)
        x = out
        
        # pool4
        out = self.__get_tensor('pool4', batch_size, 512, p_height, p_width, 2, 2, 28)
        p_height, p_width = inc_max_pool2(m.conv4_3.data, x,
                                            out, locations, 14, 0, 2, 2, p_height, p_width, beta)
        x = out                
        
        # conv5_1
        out = self.__get_tensor('conv5_1', batch_size, 512, p_height, p_width, 3, 1, 14)
        p_height, p_width = inc_convolution2(m.pool4.data, x, m.conv5_1_op[0].weight.data, m.conv5_1_op[0].bias.data,
                                            out.data, locations.data, 14, 1, 1, p_height, p_width, beta)
        x = out        
        
        # conv5_2
        out = self.__get_tensor('conv5_2', batch_size, 512, p_height, p_width, 3, 1, 14)
        p_height, p_width = inc_convolution2(m.conv5_1.data, x, m.conv5_2_op[0].weight.data, m.conv5_2_op[0].bias.data,
                                            out, locations, 14, 1, 1, p_height, p_width, beta)
        x = out
        
        # conv5_3
        out = self.__get_tensor('conv5_3', batch_size, 512, p_height, p_width, 3, 1, 14)
        p_height, p_width = inc_convolution2(m.conv5_2.data, x, m.conv5_3_op[0].weight.data, m.conv5_3_op[0].bias.data,
                                            out, locations, 14, 1, 1, p_height, p_width, beta)
        x = out
        
        # pool5
        out = self.__get_tensor('pool5', batch_size, 512, p_height, p_width, 2, 2, 14)
        p_height, p_width = inc_max_pool2(m.conv5_3.data, x,
                                            out, locations, 7, 0, 2, 2, p_height, p_width, beta)
        x = out                      
        
        #final full-projection
        out = self.__get_tensor('pool5-full', batch_size, 512, 7, 7, 1, 1, 7)
        final_full_projection(m.pool5.data, x, out, locations, p_height, p_width)
        x = out
        
        x = x.view(x.size(0), -1)
        x = m.classifier(x)
        return x
    
      
    def __get_tensor(self, name, batch_size, channels, p_height, p_width, k_size, stride, out_size):
        if name in self.tensor_cache:
            return self.tensor_cache[name].fill_(0.0)
        else:
            tensor = torch.FloatTensor(batch_size, channels, *self.__get_output_shape(p_height, p_width, k_size, stride, out_size)).cuda()
            self.tensor_cache[name] = tensor
            return tensor.fill_(0.0)
        
                
    def __get_output_shape(self, p_height, p_width, k_size, stride, out_size):
        p_height = min(int(math.ceil((p_height+k_size-1)*1.0/stride)), out_size)
        p_width = min(int(math.ceil((p_width+k_size-1)*1.0/stride)), out_size)
        return (p_height,p_width)
    
    
    
if __name__ == "__main__":
    batch_size = 1
    patch_size = 16
    input_size = 224

    image_patches = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0).repeat(batch_size, 1, 1)

    x_loc = random.sample(range(0, input_size - patch_size), batch_size)
    y_loc = random.sample(range(0, input_size - patch_size), batch_size)
    patch_locations = zip(x_loc, y_loc)
    patch_locations = [(0, 0)]

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')
    images = loader(images)

    images = images.unsqueeze(0)
    #images = images.repeat(batch_size, 1, 1, 1)

    #for i,(x,y) in enumerate(patch_locations):
    #    images[i, :, x:x+patch_size, y:y+patch_size] = image_patch

    y = VGG16().forward(images.cuda())

    patch_locations = torch.from_numpy(np.array(patch_locations, dtype=np.int32))

    inc_model = IncrementalVGG16V2(images, beta=1.0)

    inc_model.eval()
    x = inc_model.forward_gpu2(image_patches, patch_locations, patch_size, patch_size)
    # print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])

    temp = y[:,:,0:18,0:18] - x
    print(np.max(np.abs(temp.cpu().data.numpy())))