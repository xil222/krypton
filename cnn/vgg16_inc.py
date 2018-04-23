#!/usr/bin/env python
# coding=utf-8

import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from commons import inc_convolution, inc_max_pool
from imagenet_classes import class_names
from vgg16 import VGG16


class IncrementalVGG16(nn.Module):

    def __init__(self, full_model, initial_in_tenosor, patch_growth_threshold=1.0):
        super(IncrementalVGG16, self).__init__()

        # performing initial full inference
        full_model.eval()
        self.initial_result = full_model(initial_in_tenosor).cpu().data.numpy()
        self.full_model = full_model
        self.patch_growth_threshold = patch_growth_threshold

    def forward(self, x, patch_location_tensor, p_height, p_width):
        full_model = self.full_model
        beta = self.patch_growth_threshold
        
        #conv1_1
        p_height, p_width = inc_convolution(x.data, full_model.conv1_1_op[0].weight.data, full_model.conv1_1_op[0].bias.data, full_model.conv1_1.data, patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        #conv1_2
        p_height, p_width = inc_convolution(full_model.conv1_1.data, full_model.conv1_2_op[0].weight.data, full_model.conv1_2_op[0].bias.data, full_model.conv1_2.data, patch_location_tensor.data, 1, 1, p_height, p_width, beta)

    
        #pool1
        p_height, p_width = inc_max_pool(full_model.conv1_2.data, full_model.pool1.data, patch_location_tensor.data, 0, 2, 2, p_height, p_width, beta)

        #conv2_1
        p_height, p_width = inc_convolution(full_model.pool1.data, full_model.conv2_1_op[0].weight.data, full_model.conv2_1_op[0].bias.data, full_model.conv2_1.data, patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        #conv2_2
        p_height, p_width = inc_convolution(full_model.conv2_1.data, full_model.conv2_2_op[0].weight.data, full_model.conv2_2_op[0].bias.data, full_model.conv2_2.data, patch_location_tensor.data, 1, 1, p_height, p_width, beta)

    
        #pool2
        p_height, p_width = inc_max_pool(full_model.conv2_2.data, full_model.pool2.data, patch_location_tensor.data,
                                         0, 2, 2, p_height, p_width, beta)

        #conv3_1
        p_height, p_width = inc_convolution(full_model.pool2.data, full_model.conv3_1_op[0].weight.data,
                                            full_model.conv3_1_op[0].bias.data, full_model.conv3_1.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        #conv3_2
        p_height, p_width = inc_convolution(full_model.conv3_1.data, full_model.conv3_2_op[0].weight.data,
                                            full_model.conv3_2_op[0].bias.data, full_model.conv3_2.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        #conv3_3
        p_height, p_width = inc_convolution(full_model.conv3_2.data, full_model.conv3_3_op[0].weight.data,
                                            full_model.conv3_3_op[0].bias.data, full_model.conv3_3.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)
    
        #pool3
        p_height, p_width = inc_max_pool(full_model.conv3_3.data, full_model.pool3.data, patch_location_tensor.data,
                                         0, 2, 2, p_height, p_width, beta)


        #conv4_1
        p_height, p_width = inc_convolution(full_model.pool3.data, full_model.conv4_1_op[0].weight.data,
                                            full_model.conv4_1_op[0].bias.data, full_model.conv4_1.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        #conv4_2
        p_height, p_width = inc_convolution(full_model.conv4_1.data, full_model.conv4_2_op[0].weight.data,
                                            full_model.conv4_2_op[0].bias.data, full_model.conv4_2.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        # beta = 1.0
        # print(p_height, p_width)

        #conv4_3
        p_height, p_width = inc_convolution(full_model.conv4_2.data, full_model.conv4_3_op[0].weight.data,
                                            full_model.conv4_3_op[0].bias.data, full_model.conv4_3.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)


        #pool4
        p_height, p_width = inc_max_pool(full_model.conv4_3.data, full_model.pool4.data, patch_location_tensor.data,
                                         0, 2, 2, p_height, p_width, beta)


        #conv5_1
        p_height, p_width = inc_convolution(full_model.pool4.data, full_model.conv5_1_op[0].weight.data,
                                            full_model.conv5_1_op[0].bias.data, full_model.conv5_1.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        #conv5_2
        p_height, p_width = inc_convolution(full_model.conv5_1.data, full_model.conv5_2_op[0].weight.data,
                                            full_model.conv5_2_op[0].bias.data, full_model.conv5_2.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        #conv5_3
        p_height, p_width = inc_convolution(full_model.conv5_2.data, full_model.conv5_3_op[0].weight.data,
                                            full_model.conv5_3_op[0].bias.data, full_model.conv5_3.data,
                                            patch_location_tensor.data, 1, 1, p_height, p_width, beta)

        #pool5
        p_height, p_width = inc_max_pool(full_model.conv5_3.data, full_model.pool5.data, patch_location_tensor.data,
                                         0, 2, 2, p_height, p_width, beta)

        x = Variable(full_model.pool5.data)
        x = x.view(x.size(0), -1)
        x = full_model.classifier(x)
        return x
    
        
        

if __name__ == "__main__":
    batch_size = 128
    patch_size = 16
    input_size = 224

    x_loc = random.sample(range(0, input_size - patch_size), batch_size)
    y_loc = random.sample(range(0, input_size - patch_size), batch_size)
    patch_locations = zip(x_loc, y_loc)

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')
    images = loader(images)

    images = Variable(images.unsqueeze(0), requires_grad=False, volatile=True).cuda()
    images = images.repeat(batch_size, 1, 1, 1)
    patch_locations = Variable(torch.from_numpy(np.array(patch_locations, dtype=np.int32))).cuda()

    full_model = VGG16()
    full_model.eval()
    inc_model = IncrementalVGG16(full_model, images, patch_growth_threshold=0.25)
    del full_model
    torch.cuda.empty_cache()

    inc_model.eval()
    x = inc_model(images, patch_locations, patch_size, patch_size)
    print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])
