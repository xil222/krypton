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

    def __init__(self, in_tensor, beta=1.0, cuda=True):
        super(IncrementalVGG16, self).__init__()

        # performing initial full inference
        full_model = VGG16()
        full_model.eval()
        self.cuda = cuda
        if self.cuda:
            in_tensor = in_tensor.cuda()

        self.initial_result = full_model.forward_materialized(Variable(in_tensor, volatile=True)).cpu().data.numpy()
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

        # beta = 1.0
        # print(p_height, p_width)

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

        x = Variable(m.pool5.data)
        x = x.view(x.size(0), -1)
        x = m.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 1
    patch_size = 16
    input_size = 224

    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0)

    x_loc = random.sample(range(0, input_size - patch_size), batch_size)
    y_loc = random.sample(range(0, input_size - patch_size), batch_size)
    patch_locations = zip(x_loc, y_loc)
    patch_locations = [(175, 175)]

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')
    images = loader(images)

    images = images.unsqueeze(0)
    images = images.repeat(batch_size, 1, 1, 1)

    for i,(x,y) in enumerate(patch_locations):
        images[i, :, x:x+patch_size, y:y+patch_size] = image_patch

    y = VGG16().forward(Variable(images).cuda())

    patch_locations = torch.from_numpy(np.array(patch_locations, dtype=np.int32))

    inc_model = IncrementalVGG16(images, beta=1.0)

    inc_model.eval()
    x = inc_model(images, patch_locations, patch_size, patch_size)
    # print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])

    temp = y - x
    print(np.max(np.abs(temp.cpu().data.numpy())))