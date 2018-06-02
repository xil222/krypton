#!/usr/bin/env python
# coding=utf-8

import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
import torch.nn.functional as F

from commons import inc_convolution_bn, inc_max_pool, inc_add
from imagenet_classes import class_names
from resnet18 import ResNet18


class IncrementalResNet18V2(nn.Module):

    def __init__(self, in_tensor, beta=1.0, cuda=True):
        super(IncrementalResNet18V2, self).__init__()

        # performing initial full inference
        full_model = ResNet18()
        full_model.eval()
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
        
        # conv1
        p_height, p_width = inc_convolution_bn(x, m.conv1_op[0].weight.data,
                                               m.conv1_op[1].running_mean.data,
                                               m.conv1_op[1].running_var.data,
                                               m.conv1_op[1].weight.data,
                                               m.conv1_op[1].bias.data,
                                            m.conv1.data, locations, 3, 2, p_height, p_width, beta, relu=True)
        
        # pool1
        p_height, p_width = inc_max_pool(m.conv1.data, m.pool1.data, locations, 1, 2, 3, p_height, p_width, beta)
        
        # conv2
        p_height, p_width = inc_convolution_bn(m.pool1.data, m.conv2_1_a_op[0].weight.data,
                                               m.conv2_1_a_op[1].running_mean.data,
                                               m.conv2_1_a_op[1].running_var.data,
                                               m.conv2_1_a_op[1].weight.data,
                                               m.conv2_1_a_op[1].bias.data,
                                            m.conv2_1_a.data, locations, 1, 1, p_height, p_width, beta, relu=True)
        p_height, p_width = inc_convolution_bn(m.conv2_1_a.data, m.conv2_1_b_op[0].weight.data,
                                               m.conv2_1_b_op[1].running_mean.data,
                                               m.conv2_1_b_op[1].running_var.data,
                                               m.conv2_1_b_op[1].weight.data,
                                               m.conv2_1_b_op[1].bias.data,
                                            m.conv2_1_b.data, locations, 1, 1, p_height, p_width, beta, relu=False)
        
        inc_add(m.pool1.data, m.conv2_1_b.data, m.merge_2_1.data, locations, p_height, p_width, relu=True)
        
        p_height, p_width = inc_convolution_bn(m.merge_2_1.data, m.conv2_2_a_op[0].weight.data,
                                               m.conv2_2_a_op[1].running_mean.data,
                                               m.conv2_2_a_op[1].running_var.data,
                                               m.conv2_2_a_op[1].weight.data,
                                               m.conv2_2_a_op[1].bias.data,
                                            m.conv2_2_a.data, locations, 1, 1, p_height, p_width, beta, relu=True)
        p_height, p_width = inc_convolution_bn(m.conv2_2_a.data, m.conv2_2_b_op[0].weight.data,
                                               m.conv2_2_b_op[1].running_mean.data,
                                               m.conv2_2_b_op[1].running_var.data,
                                               m.conv2_2_b_op[1].weight.data,
                                               m.conv2_2_b_op[1].bias.data,
                                            m.conv2_2_b.data, locations, 1, 1, p_height, p_width, beta, relu=False)

        inc_add(m.merge_2_1.data, m.conv2_2_b.data, m.merge_2_2.data, locations, p_height, p_width, relu=True)
        
        r_p_height, r_p_width = p_height, p_width
        r_locations = locations.clone()
        # conv3
        p_height, p_width = inc_convolution_bn(m.merge_2_2.data, m.conv3_1_a_op[0].weight.data,
                                               m.conv3_1_a_op[1].running_mean.data,
                                               m.conv3_1_a_op[1].running_var.data,
                                               m.conv3_1_a_op[1].weight.data,
                                               m.conv3_1_a_op[1].bias.data,
                                               m.conv3_1_a.data, locations, 1, 2, p_height, p_width, beta, relu=True)
        p_height, p_width = inc_convolution_bn(m.conv3_1_a.data, m.conv3_1_b_op[0].weight.data,
                                               m.conv3_1_b_op[1].running_mean.data,
                                               m.conv3_1_b_op[1].running_var.data,
                                               m.conv3_1_b_op[1].weight.data,
                                               m.conv3_1_b_op[1].bias.data,
                                               m.conv3_1_b.data, locations, 1, 1, p_height, p_width, beta, relu=False)
        
        
        _, _ = inc_convolution_bn(m.merge_2_2.data, m.residual_3_op[0].weight.data,
                                               m.residual_3_op[1].running_mean.data,
                                               m.residual_3_op[1].running_var.data,
                                               m.residual_3_op[1].weight.data,
                                               m.residual_3_op[1].bias.data,
                                               m.residual_3.data, r_locations, 0, 2, r_p_height, r_p_width, beta, relu=False)
        inc_add(m.residual_3.data, m.conv3_1_b.data, m.merge_3_1.data, locations, p_height, p_width, relu=True)
        
        p_height, p_width = inc_convolution_bn(m.merge_3_1.data, m.conv3_2_a_op[0].weight.data,
                                               m.conv3_2_a_op[1].running_mean.data,
                                               m.conv3_2_a_op[1].running_var.data,
                                               m.conv3_2_a_op[1].weight.data,
                                               m.conv3_2_a_op[1].bias.data,
                                            m.conv3_2_a.data, locations, 1, 1, p_height, p_width, beta, relu=True)
        p_height, p_width = inc_convolution_bn(m.conv3_2_a.data, m.conv3_2_b_op[0].weight.data,
                                               m.conv3_2_b_op[1].running_mean.data,
                                               m.conv3_2_b_op[1].running_var.data,
                                               m.conv3_2_b_op[1].weight.data,
                                               m.conv3_2_b_op[1].bias.data,
                                            m.conv3_2_b.data, locations, 1, 1, p_height, p_width, beta, relu=False)

        inc_add(m.merge_3_1.data, m.conv3_2_b.data, m.merge_3_2, locations, p_height, p_width, relu=True)
        
        
        r_p_height, r_p_width = p_height, p_width
        r_locations = locations.clone()
        # conv4
        p_height, p_width = inc_convolution_bn(m.merge_3_2.data, m.conv4_1_a_op[0].weight.data,
                                               m.conv4_1_a_op[1].running_mean.data,
                                               m.conv4_1_a_op[1].running_var.data,
                                               m.conv4_1_a_op[1].weight.data,
                                               m.conv4_1_a_op[1].bias.data,
                                               m.conv4_1_a.data, locations, 1, 2, p_height, p_width, beta, relu=True)
        p_height, p_width = inc_convolution_bn(m.conv4_1_a.data, m.conv4_1_b_op[0].weight.data,
                                               m.conv4_1_b_op[1].running_mean.data,
                                               m.conv4_1_b_op[1].running_var.data,
                                               m.conv4_1_b_op[1].weight.data,
                                               m.conv4_1_b_op[1].bias.data,
                                               m.conv4_1_b.data, locations, 1, 1, p_height, p_width, beta, relu=False)
        
        
        _, _ = inc_convolution_bn(m.merge_3_2.data, m.residual_4_op[0].weight.data,
                                               m.residual_4_op[1].running_mean.data,
                                               m.residual_4_op[1].running_var.data,
                                               m.residual_4_op[1].weight.data,
                                               m.residual_4_op[1].bias.data,
                                               m.residual_4.data, r_locations, 0, 2, r_p_height, r_p_width, beta, relu=False)
        inc_add(m.residual_4.data, m.conv4_1_b.data, m.merge_4_1, locations, p_height, p_width, relu=True)
        
        p_height, p_width = inc_convolution_bn(m.merge_4_1.data, m.conv4_2_a_op[0].weight.data,
                                               m.conv4_2_a_op[1].running_mean.data,
                                               m.conv4_2_a_op[1].running_var.data,
                                               m.conv4_2_a_op[1].weight.data,
                                               m.conv4_2_a_op[1].bias.data,
                                            m.conv4_2_a.data, locations, 1, 1, p_height, p_width, beta, relu=True)
        p_height, p_width = inc_convolution_bn(m.conv4_2_a.data, m.conv4_2_b_op[0].weight.data,
                                               m.conv4_2_b_op[1].running_mean.data,
                                               m.conv4_2_b_op[1].running_var.data,
                                               m.conv4_2_b_op[1].weight.data,
                                               m.conv4_2_b_op[1].bias.data,
                                            m.conv4_2_b.data, locations, 1, 1, p_height, p_width, beta, relu=False)

        inc_add(m.merge_4_1.data, m.conv4_2_b.data, m.merge_4_2, locations, p_height, p_width, relu=True)
        
        
        r_p_height, r_p_width = p_height, p_width
        r_locations = locations.clone()
        # conv5
        p_height, p_width = inc_convolution_bn(m.merge_4_2.data, m.conv5_1_a_op[0].weight.data,
                                               m.conv5_1_a_op[1].running_mean.data,
                                               m.conv5_1_a_op[1].running_var.data,
                                               m.conv5_1_a_op[1].weight.data,
                                               m.conv5_1_a_op[1].bias.data,
                                               m.conv5_1_a.data, locations, 1, 2, p_height, p_width, beta, relu=True)
        p_height, p_width = inc_convolution_bn(m.conv5_1_a.data, m.conv5_1_b_op[0].weight.data,
                                               m.conv5_1_b_op[1].running_mean.data,
                                               m.conv5_1_b_op[1].running_var.data,
                                               m.conv5_1_b_op[1].weight.data,
                                               m.conv5_1_b_op[1].bias.data,
                                               m.conv5_1_b.data, locations, 1, 1, p_height, p_width, beta, relu=False)
        
        
        _, _ = inc_convolution_bn(m.merge_4_2.data, m.residual_5_op[0].weight.data,
                                               m.residual_5_op[1].running_mean.data,
                                               m.residual_5_op[1].running_var.data,
                                               m.residual_5_op[1].weight.data,
                                               m.residual_5_op[1].bias.data,
                                               m.residual_5.data, r_locations, 0, 2, r_p_height, r_p_width, beta, relu=False)
        inc_add(m.residual_5.data, m.conv5_1_b.data, m.merge_5_1, locations, p_height, p_width, relu=True)
        
        p_height, p_width = inc_convolution_bn(m.merge_5_1.data, m.conv5_2_a_op[0].weight.data,
                                               m.conv5_2_a_op[1].running_mean.data,
                                               m.conv5_2_a_op[1].running_var.data,
                                               m.conv5_2_a_op[1].weight.data,
                                               m.conv5_2_a_op[1].bias.data,
                                            m.conv5_2_a.data, locations, 1, 1, p_height, p_width, beta, relu=True)
        p_height, p_width = inc_convolution_bn(m.conv5_2_a.data, m.conv5_2_b_op[0].weight.data,
                                               m.conv5_2_b_op[1].running_mean.data,
                                               m.conv5_2_b_op[1].running_var.data,
                                               m.conv5_2_b_op[1].weight.data,
                                               m.conv5_2_b_op[1].bias.data,
                                            m.conv5_2_b.data, locations, 1, 1, p_height, p_width, beta, relu=False)

        inc_add(m.merge_5_1.data, m.conv5_2_b.data, m.merge_5_2, locations, p_height, p_width, relu=True)
                
        x = m.avgpool(m.merge_5_2)
        x = x.view(x.size(0), -1)
        x = m.fc(x)

        return x


if __name__ == "__main__":
    batch_size = 1
    patch_size = 16
    input_size = 224

    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0)

    x_loc = random.sample(range(0, input_size - patch_size), batch_size)
    y_loc = random.sample(range(0, input_size - patch_size), batch_size)
    patch_locations = zip(x_loc, y_loc)
    patch_locations = [(0, 0)]

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')
    images = loader(images)

    images = images.unsqueeze(0)
    images = images.repeat(batch_size, 1, 1, 1)

    for i,(x,y) in enumerate(patch_locations):
        images[i, :, x:x+patch_size, y:y+patch_size] = image_patch

    y = ResNet18().forward(Variable(images).cuda())

    patch_locations = torch.from_numpy(np.array(patch_locations, dtype=np.int32))

    inc_model = IncrementalResNet18V2(images, beta=1.0)

    inc_model.eval()
    x = inc_model(images, patch_locations, patch_size, patch_size)
    # print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])

    temp = (y - x).cpu().data.numpy()
    import matplotlib.pyplot as plt
    plt.imshow(temp[0,0,:,:])
    plt.show()
    print(np.max(np.abs(temp)))