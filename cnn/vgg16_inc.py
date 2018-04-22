#!/usr/bin/env python
# coding=utf-8

import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from commons import IncConvModule, IncMaxPoolModule
from imagenet_classes import class_names
from vgg16 import VGG16


class IncrementalVGG16(nn.Module):

    def __init__(self, full_model, initial_in_tenosor, patch_growth_threshold=1.0):
        super(IncrementalVGG16, self).__init__()

        # performing initial full inference
        full_model.eval()
        full_model.forward(initial_in_tenosor)

        self.conv1_1_op = IncConvModule(initial_in_tenosor,
                                        full_model.conv1_1_op[0].weight.data, full_model.conv1_1_op[0].bias.data,
                                        full_model.conv1_1, 1, 1, 3, patch_growth_threshold)
        self.conv1_2_op = IncConvModule(full_model.conv1_1,
                                        full_model.conv1_2_op[0].weight.data, full_model.conv1_2_op[0].bias.data,
                                        full_model.conv1_2, 1, 1, 3, patch_growth_threshold)
        self.pool1_op = IncMaxPoolModule(full_model.conv1_2, full_model.pool1, 0, 2, 2,
                                         patch_growth_threshold)

        self.conv2_1_op = IncConvModule(full_model.pool1,
                                        full_model.conv2_1_op[0].weight.data, full_model.conv2_1_op[0].bias.data,
                                        full_model.conv2_1, 1, 1, 3, patch_growth_threshold)
        self.conv2_2_op = IncConvModule(full_model.conv2_1,
                                        full_model.conv2_2_op[0].weight.data, full_model.conv2_2_op[0].bias.data,
                                        full_model.conv2_2, 1, 1, 3, patch_growth_threshold)
        self.pool2_op = IncMaxPoolModule(full_model.conv2_2, full_model.pool2, 0, 2, 2,
                                         patch_growth_threshold)

        self.conv3_1_op = IncConvModule(full_model.pool2,
                                        full_model.conv3_1_op[0].weight.data, full_model.conv3_1_op[0].bias.data,
                                        full_model.conv3_1, 1, 1, 3, patch_growth_threshold)
        self.conv3_2_op = IncConvModule(full_model.conv3_1,
                                        full_model.conv3_2_op[0].weight.data, full_model.conv3_2_op[0].bias.data,
                                        full_model.conv2_2, 1, 1, 3, patch_growth_threshold)
        self.conv3_3_op = IncConvModule(full_model.conv3_2,
                                        full_model.conv3_3_op[0].weight.data, full_model.conv3_3_op[0].bias.data,
                                        full_model.conv3_3, 1, 1, 3, patch_growth_threshold)
        self.pool3_op = IncMaxPoolModule(full_model.conv3_3, full_model.pool3, 0, 2, 2,
                                         patch_growth_threshold)

        self.conv4_1_op = IncConvModule(full_model.pool3,
                                        full_model.conv4_1_op[0].weight.data, full_model.conv4_1_op[0].bias.data,
                                        full_model.conv4_1, 1, 1, 3, patch_growth_threshold)
        self.conv4_2_op = IncConvModule(full_model.conv4_1,
                                        full_model.conv4_2_op[0].weight.data, full_model.conv4_2_op[0].bias.data,
                                        full_model.conv4_2, 1, 1, 3, patch_growth_threshold)
        self.conv4_3_op = IncConvModule(full_model.conv4_2,
                                        full_model.conv4_3_op[0].weight.data, full_model.conv4_3_op[0].bias.data,
                                        full_model.conv4_3, 1, 1, 3, patch_growth_threshold)
        self.pool4_op = IncMaxPoolModule(full_model.conv4_3, full_model.pool4, 0, 2, 2,
                                         patch_growth_threshold)

        self.conv5_1_op = IncConvModule(full_model.pool4,
                                        full_model.conv5_1_op[0].weight.data, full_model.conv5_1_op[0].bias.data,
                                        full_model.conv5_1, 1, 1, 3,
                                        patch_growth_threshold)
        self.conv5_2_op = IncConvModule(full_model.conv5_1,
                                        full_model.conv5_2_op[0].weight.data, full_model.conv5_2_op[0].bias.data,
                                        full_model.conv5_2, 1, 1, 3,
                                        patch_growth_threshold)
        self.conv5_3_op = IncConvModule(full_model.conv5_2,
                                        full_model.conv5_3_op[0].weight.data, full_model.conv5_3_op[0].bias.data,
                                        full_model.conv5_3, 1, 1, 3,
                                        patch_growth_threshold)
        self.pool5_op = IncMaxPoolModule(full_model.conv5_3, full_model.pool5, 0, 2, 2,
                                         patch_growth_threshold)

        self.classifier = full_model.classifier

    def forward(self, x, patch_location_tensor, p_height=0, p_width=0):
        # set the new input
        self.conv1_1_op.in_tensor = x

        (_, patch_location_tensor), (p_height, p_width) = self.conv1_1_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.conv1_2_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.pool1_op(patch_location_tensor, p_height, p_width)

        (_, patch_location_tensor), (p_height, p_width) = self.conv2_1_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.conv2_2_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.pool2_op(patch_location_tensor, p_height, p_width)

        (_, patch_location_tensor), (p_height, p_width) = self.conv3_1_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.conv3_2_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.conv3_3_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.pool2_op(patch_location_tensor, p_height, p_width)

        (_, patch_location_tensor), (p_height, p_width) = self.conv4_1_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.conv4_2_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.conv4_3_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.pool4_op(patch_location_tensor, p_height, p_width)

        (_, patch_location_tensor), (p_height, p_width) = self.conv5_1_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.conv5_2_op(patch_location_tensor, p_height, p_width)
        (_, patch_location_tensor), (p_height, p_width) = self.conv5_3_op(patch_location_tensor, p_height, p_width)
        (x, patch_location_tensor), _ = self.pool5_op(patch_location_tensor, p_height, p_width)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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
