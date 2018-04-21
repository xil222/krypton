#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from commons import load_dict_from_hdf5
from imagenet_classes import class_names


class VGG16(nn.Module):

    def __init__(self, init_weights=True):
        super(VGG16, self).__init__()

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

        if init_weights:
            self._initialize_weights()

        #for param in self.parameters():
        #   param.requires_grad = False

    def forward(self, x):
        self.conv1_1 = self.conv1_1_op(x)
        self.conv1_2 = self.conv1_2_op(self.conv1_1)
        self.pool1 = self.pool1_op(self.conv1_2)

        self.conv2_1 = self.conv2_1_op(self.pool1)
        self.conv2_2 = self.conv2_2_op(self.conv2_1)
        self.pool2 = self.pool2_op(self.conv2_2)

        self.conv3_1 = self.conv3_1_op(self.pool2)
        self.conv3_2 = self.conv3_2_op(self.conv3_1)
        self.conv3_3 = self.conv3_3_op(self.conv3_2)
        self.pool3 = self.pool2_op(self.conv3_3)

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

    def _initialize_weights(self):
        weights_data = load_dict_from_hdf5("./vgg16_weights_ptch.h5")

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
    print(class_names[np.argmax(x.data.cpu().numpy()[0,:])])
