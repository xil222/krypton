import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from PIL import Image
from torchvision.transforms import transforms
import os
import sys

sys.path.append('../')

from cnn.imagenet_classes import class_names
from cnn.commons import load_dict_from_hdf5


class Inception3(nn.Module):

    def __init__(self, gpu=True, num_classes=1000):
        super(Inception3, self).__init__()

        self.gpu = gpu

        # layer1
        self.conv1_op = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False),
                                      nn.BatchNorm2d(32, eps=0.001), nn.ReLU(inplace=True))
        # layer2
        self.conv2_a_op = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, bias=False),
                                        nn.BatchNorm2d(32, eps=0.001), nn.ReLU(inplace=True))
        self.conv2_b_op = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.pool2_op = nn.MaxPool2d(kernel_size=3, stride=2)

        # layer3
        self.conv3_op = nn.Sequential(nn.Conv2d(64, 80, kernel_size=1, stride=1, bias=False),
                                      nn.BatchNorm2d(80, eps=0.001), nn.ReLU(inplace=True))

        # layer4
        self.conv4_op = nn.Sequential(nn.Conv2d(80, 192, kernel_size=3, stride=1, bias=False),
                                      nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.pool4_op = nn.MaxPool2d(kernel_size=3, stride=2)

        # layer5
        self.mixed_5a = InceptionA(192, 32)
        self.mixed_5b = InceptionA(256, 64)
        self.mixed_5c = InceptionA(288, 64)

        # layer6
        self.mixed_6a = InceptionB(288)
        self.mixed_6b = InceptionC(768, c7=128)
        self.mixed_6c = InceptionC(768, c7=160)
        self.mixed_6d = InceptionC(768, c7=160)
        self.mixed_6e = InceptionC(768, c7=192)
        
        # layer 7
        self.mixed_7a = InceptionD(768)
        self.mixed_7b = InceptionE(1280)
        self.mixed_7c = InceptionE(2048)

        self.fc = nn.Linear(2048, num_classes)
        
        self.__initialize_weights(gpu)

    def forward(self, x):
        return self.forward_materialized(x)

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
        
        return x

    def forward_materialized(self, x):
        x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

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
        
        return x

    
    def __initialize_weights(self, gpu):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        values = load_dict_from_hdf5(dir_path + "/inception3_weights_ptch.h5", gpu)
        keys = [key for key in sorted(values, key=lambda i: int(i.split('.')[0]))]
        values = [values[key] for key in sorted(values, key=lambda i: int(i.split('.')[0]))]

        count = 0
        for mod in [self.conv1_op, self.conv2_a_op, self.conv2_b_op]:
            mod[0].weight.data = values[count]
            count += 1
            mod[1].weight.data = values[count]
            count += 1
            mod[1].bias.data = values[count]
            count += 1
            mod[1].running_mean.data = values[count]
            count += 1
            mod[1].running_var.data = values[count]
            count += 1

        for mod in [self.conv3_op, self.conv4_op]:
            mod[0].weight.data = values[count]
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
            for mod in list(mods):
                mod[0].weight.data = values[count]
                count += 1
                mod[1].weight.data = values[count]
                count += 1
                mod[1].bias.data = values[count]
                count += 1
                mod[1].running_mean.data = values[count]
                count += 1
                mod[1].running_var.data = values[count]
                count += 1

        for mod in list(self.mixed_6a.children()):
            mod[0].weight.data = values[count]
            count += 1
            mod[1].weight.data = values[count]
            count += 1
            mod[1].bias.data = values[count]
            count += 1
            mod[1].running_mean.data = values[count]
            count += 1
            mod[1].running_var.data = values[count]
            count += 1

        for mods in [self.mixed_6b.children(), self.mixed_6c.children(), self.mixed_6d.children(),
                     self.mixed_6e.children()]:
            for mod in list(mods):
                mod[0].weight.data = values[count]
                count += 1
                mod[1].weight.data = values[count]
                count += 1
                mod[1].bias.data = values[count]
                count += 1
                mod[1].running_mean.data = values[count]
                count += 1
                mod[1].running_var.data = values[count]
                count += 1
                
        count += 12
        
        for mod in list(self.mixed_7a.children()):
            mod[0].weight.data = values[count]
            count += 1
            mod[1].weight.data = values[count]
            count += 1
            mod[1].bias.data = values[count]
            count += 1
            mod[1].running_mean.data = values[count]
            count += 1
            mod[1].running_var.data = values[count]
            count += 1

            
        for mods in [self.mixed_7b.children(), self.mixed_7c.children()]:
            for mod in list(mods):
                mod[0].weight.data = values[count]
                count += 1
                mod[1].weight.data = values[count]
                count += 1
                mod[1].bias.data = values[count]
                count += 1
                mod[1].running_mean.data = values[count]
                count += 1
                mod[1].running_var.data = values[count]
                count += 1
    
        self.fc.weight.data = values[count]
        count += 1
        self.fc.bias.data = values[count]
            
            
class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.b1_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))

        self.b5_1_op = nn.Sequential(nn.Conv2d(in_channels, 48, kernel_size=1, stride=1, bias=False),
                                     nn.BatchNorm2d(48, eps=0.001), nn.ReLU(inplace=True))
        self.b5_2_op = nn.Sequential(nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=2, bias=False),
                                     nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))

        self.b3_1_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, bias=False),
                                     nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2_op = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))
        self.b3_3_op = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))

        self.branch_pool_op = nn.Sequential(nn.Conv2d(in_channels, pool_features, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(pool_features, eps=0.001),
                                            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.b1_op(x)
        b5 = self.b5_1_op(x)
        b5 = self.b5_2_op(b5)
        b3 = self.b3_1_op(x)
        b3 = self.b3_2_op(b3)
        b3 = self.b3_3_op(b3)
        b_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        b_pool = self.branch_pool_op(b_pool)

        x = torch.cat([b1, b5, b3, b_pool], 1)
        return x

    def forward_materialized(self, x):
        self.b1 = self.b1_op(x)
        self.b5_1 = self.b5_1_op(x)
        self.b5_2 = self.b5_2_op(self.b5_1)
        self.b3_1 = self.b3_1_op(x)
        self.b3_2 = self.b3_2_op(self.b3_1)
        self.b3_3 = self.b3_3_op(self.b3_2)
        self.b_pool_1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.b_pool_2 = self.branch_pool_op(self.b_pool_1)

        x = torch.cat([self.b1, self.b5_2, self.b3_3, self.b_pool_2], 1)
        return x

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.b3_op = nn.Sequential(nn.Conv2d(in_channels, 384, kernel_size=3, stride=2, bias=False),
                                   nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))

        self.b3_db_1_op = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(64, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_2_op = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False),
                                        nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_3_op = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, stride=2, bias=False),
                                        nn.BatchNorm2d(96, eps=0.001), nn.ReLU(inplace=True))

    def forward(self, x):
        b3 = self.b3_op(x)

        b3_db = self.b3_db_1_op(x)
        b3_db = self.b3_db_2_op(b3_db)
        b3_db = self.b3_db_3_op(b3_db)

        b_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [b3, b3_db, b_pool]
        return torch.cat(outputs, 1)
    
    
    def forward_materialized(self, x):
        self.b3 = self.b3_op(x)

        self.b3_db_1 = self.b3_db_1_op(x)
        self.b3_db_2 = self.b3_db_2_op(self.b3_db_1)
        self.b3_db_3 = self.b3_db_3_op(self.b3_db_2)

        self.b_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [self.b3, self.b3_db_3, self.b_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, c7):
        super(InceptionC, self).__init__()
        
        self.b1_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        
        self.b7_1_op = nn.Sequential(nn.Conv2d(in_channels, c7, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_2_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3), bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_3_op = nn.Sequential(nn.Conv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0), bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))

        self.b7_db_1_op = nn.Sequential(nn.Conv2d(in_channels, c7, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_2_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(7, 1), padding=(3,0), bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_3_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(1, 7), padding=(0,3), bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        
        self.b7_db_4_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(7, 1), padding=(3,0), bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_db_5_op = nn.Sequential(nn.Conv2d(c7, 192, kernel_size=(1, 7), padding=(0,3), bias=False),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))

        self.branch_pool = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))

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

        b_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        b_pool = self.branch_pool(b_pool)

        outputs = [b1, b7, b7_db, b_pool]
        return torch.cat(outputs, 1)

    
    def forward_materialized(self, x):
        self.b1 = self.b1_op(x)

        self.b7_1 = self.b7_1_op(x)
        self.b7_2 = self.b7_2_op(self.b7_1)
        self.b7_3 = self.b7_3_op(self.b7_2)

        self.b7_db_1 = self.b7_db_1_op(x)
        self.b7_db_2 = self.b7_db_2_op(self.b7_db_1)
        self.b7_db_3 = self.b7_db_3_op(self.b7_db_2)
        self.b7_db_4 = self.b7_db_4_op(self.b7_db_3)
        self.b7_db_5 = self.b7_db_5_op(self.b7_db_4 )

        self.b_pool_1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.b_pool_2 = self.branch_pool(self.b_pool_1)

        outputs = [self.b1, self.b7_3, self.b7_db_5, self.b_pool_2]
        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.b3_1_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2_op = nn.Sequential(nn.Conv2d(192, 320, kernel_size=3, stride=2, bias=False),
                                     nn.BatchNorm2d(320, eps=0.001), nn.ReLU(inplace=True))

        self.b7_1_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))

        self.b7_2_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=(1, 7), padding=(0,3), bias=False),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_3_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=(7, 1), padding=(3,0), bias=False),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))
        self.b7_4_op = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, stride=2, bias=False),
                                     nn.BatchNorm2d(192, eps=0.001), nn.ReLU(inplace=True))

    def forward(self, x):
        b3 = self.b3_1_op(x)
        b3 = self.b3_2_op(b3)

        b7 = self.b7_1_op(x)
        b7 = self.b7_2_op(b7)
        b7 = self.b7_3_op(b7)
        b7 = self.b7_4_op(b7)

        b_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [b3, b7, b_pool]
        return torch.cat(outputs, 1)

    
    def forward_materialized(self, x):
        self.b3_1 = self.b3_1_op(x)
        self.b3_2 = self.b3_2_op(self.b3_1)

        self.b7_1 = self.b7_1_op(x)
        self.b7_2 = self.b7_2_op(self.b7_1)
        self.b7_3 = self.b7_3_op(self.b7_2)
        self.b7_4 = self.b7_4_op(self.b7_3)

        self.b_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [self.b3_2, self.b7_4, self.b_pool]
        return torch.cat(outputs, 1)
    
    
class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.b1_op = nn.Sequential(nn.Conv2d(in_channels, 320, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(320, eps=0.001), nn.ReLU(inplace=True))
        
        self.b3_1_op = nn.Sequential(nn.Conv2d(in_channels, 384, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2a_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), bias=False),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_2b_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), bias=False),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        
        self.b3_db_1_op =  nn.Sequential(nn.Conv2d(in_channels, 448, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(448, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_2_op = nn.Sequential(nn.Conv2d(448, 384, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_3a_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(1, 3), padding=(0, 1), bias=False),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))
        self.b3_db_3b_op = nn.Sequential(nn.Conv2d(384, 384, kernel_size=(3, 1), padding=(1, 0), bias=False),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))

        self.branch_pool = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(384, eps=0.001), nn.ReLU(inplace=True))

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

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [b1, b3, b3_db, branch_pool]
        return torch.cat(outputs, 1)
    
    def forward_materialized(self, x):
        self.b1 = self.b1_op(x)

        self.b3 = self.b3_1_op(x)
        b3 = [
            self.b3_2a_op(self.b3),
            self.b3_2b_op(self.b3),
        ]
        b3 = torch.cat(b3, 1)

        self.b3_db_1 = self.b3_db_1_op(x)
        self.b3_db_2 = self.b3_db_2_op(self.b3_db_1)
        b3_db = [
            self.b3_db_3a_op(self.b3_db_2),
            self.b3_db_3b_op(self.b3_db_2),
        ]
        b3_db = torch.cat(b3_db, 1)

        self.b_pool_1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.b_pool_2 = self.branch_pool(self.b_pool_1)

        outputs = [self.b1, b3, b3_db, self.b_pool_2]
        return torch.cat(outputs, 1)
    
    
if __name__ == "__main__":
    batch_size = 1

    loader = transforms.Compose([transforms.Resize([299, 299]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')

    images = loader(images)
    images = images.unsqueeze(0).cuda()

    images = images.repeat(batch_size, 1, 1, 1)

    model = Inception3().cuda()
    model.eval()

    x = model(images)
    print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])
