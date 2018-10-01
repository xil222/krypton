import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms

sys.path.append('../')

from cnn.imagenet_classes import class_names
from cnn.commons import load_dict_from_hdf5, inc_convolution_bn, inc_max_pool, inc_avg_pool, inc_convolution_bn2, update_output_locations


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


    def forward_inc_v2(self, x, locations, p_height, p_width, beta=1.0):
        if self.gpu:
            x = x.cuda()
            locations = locations.cuda()

        x = x.clone()
        x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        self.conv1 = self.conv1_op(x)
        # conv1
        p_height, p_width = inc_convolution_bn(x, self.conv1_op[0].weight.data,
                                               self.conv1_op[1].running_mean.data,
                                               self.conv1_op[1].running_var.data,
                                               self.conv1_op[1].weight.data,
                                               self.conv1_op[1].bias.data,
                                               self.conv1.data, locations, 0, 2, p_height, p_width, beta, relu=True, eps=1e-3)

        # conv2
        p_height, p_width = inc_convolution_bn(self.conv1.data, self.conv2_a_op[0].weight.data,
                                               self.conv2_a_op[1].running_mean.data,
                                               self.conv2_a_op[1].running_var.data,
                                               self.conv2_a_op[1].weight.data,
                                               self.conv2_a_op[1].bias.data,
                                               self.conv2_a.data, locations, 0, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)
        p_height, p_width = inc_convolution_bn(self.conv2_a.data, self.conv2_b_op[0].weight.data,
                                               self.conv2_b_op[1].running_mean.data,
                                               self.conv2_b_op[1].running_var.data,
                                               self.conv2_b_op[1].weight.data,
                                               self.conv2_b_op[1].bias.data,
                                               self.conv2_b.data, locations, 1, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)

        p_height, p_width = inc_max_pool(self.conv2_b.data, self.pool2.data, locations, 0, 2, 3, p_height, p_width, beta)

        # conv3
        p_height, p_width = inc_convolution_bn(self.pool2.data, self.conv3_op[0].weight.data,
                                               self.conv3_op[1].running_mean.data,
                                               self.conv3_op[1].running_var.data,
                                               self.conv3_op[1].weight.data,
                                               self.conv3_op[1].bias.data,
                                               self.conv3.data, locations, 0, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)

        # conv4
        p_height, p_width = inc_convolution_bn(self.conv3.data, self.conv4_op[0].weight.data,
                                               self.conv4_op[1].running_mean.data,
                                               self.conv4_op[1].running_var.data,
                                               self.conv4_op[1].weight.data,
                                               self.conv4_op[1].bias.data,
                                               self.conv4.data, locations, 0, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)
        p_height, p_width = inc_max_pool(self.conv4.data, self.pool4.data, locations, 0, 2, 3, p_height, p_width,
                                         beta)

        x, p_height, p_width = self.mixed_5a.forward_inc_v2(self.pool4.data, locations, p_height, p_width, beta)        
        x, p_height, p_width = self.mixed_5b.forward_inc_v2(x, locations, p_height, p_width, beta)      
        x, p_height, p_width = self.mixed_5c.forward_inc_v2(x, locations, p_height, p_width, beta)
        
        x, p_height, p_width = self.mixed_6a.forward_inc_v2(x, locations, p_height, p_width, beta)
        x, p_height, p_width = self.mixed_6b.forward_inc_v2(x, locations, p_height, p_width, beta)
        x, p_height, p_width = self.mixed_6c.forward_inc_v2(x, locations, p_height, p_width, beta)
        x, p_height, p_width = self.mixed_6d.forward_inc_v2(x, locations, p_height, p_width, beta)
        x, p_height, p_width = self.mixed_6e.forward_inc_v2(x, locations, p_height, p_width, beta)             
                
        x, p_height, p_width = self.mixed_7a.forward_inc_v2(x, locations, p_height, p_width, beta)   
        x, p_height, p_width = self.mixed_7b.forward_inc_v2(x, locations, p_height, p_width, beta)        
        x, p_height, p_width = self.mixed_7c.forward_inc_v2(x, locations, p_height, p_width, beta)      
        
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
        b_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)#F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
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
        self.b_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)#F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.b_pool_2 = self.branch_pool_op(self.b_pool_1)

        x = torch.cat([self.b1, self.b5_2, self.b3_3, self.b_pool_2], 1)
        return x

    def forward_inc_v2(self, x, locations, p_height, p_width, beta=1.0):

        # 1x1
        locations1 = locations.clone()
        p_height1, p_width1 = inc_convolution_bn(x, self.b1_op[0].weight.data,
                                               self.b1_op[1].running_mean.data,
                                               self.b1_op[1].running_var.data,
                                               self.b1_op[1].weight.data,
                                               self.b1_op[1].bias.data,
                                               self.b1.data, locations1, 0, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)
        # 3x3
        locations3 = locations.clone()
        p_height3, p_width3 = inc_convolution_bn(x, self.b3_1_op[0].weight.data,
                                                 self.b3_1_op[1].running_mean.data,
                                                 self.b3_1_op[1].running_var.data,
                                                 self.b3_1_op[1].weight.data,
                                                 self.b3_1_op[1].bias.data,
                                                 self.b3_1.data, locations3, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        p_height3, p_width3 = inc_convolution_bn(self.b3_1.data, self.b3_2_op[0].weight.data,
                                                 self.b3_2_op[1].running_mean.data,
                                                 self.b3_2_op[1].running_var.data,
                                                 self.b3_2_op[1].weight.data,
                                                 self.b3_2_op[1].bias.data,
                                                 self.b3_2.data, locations3, 1, 1, p_height3, p_width3, beta, relu=True,
                                                 eps=1e-3)
        p_height3, p_width3 = inc_convolution_bn(self.b3_2.data, self.b3_3_op[0].weight.data,
                                                 self.b3_3_op[1].running_mean.data,
                                                 self.b3_3_op[1].running_var.data,
                                                 self.b3_3_op[1].weight.data,
                                                 self.b3_3_op[1].bias.data,
                                                 self.b3_3.data, locations3, 1, 1, p_height3, p_width3, beta, relu=True,
                                                 eps=1e-3)

        # pool
        locationsp = locations.clone()
        p_heightp, p_widthp = inc_avg_pool(x, self.b_pool_1, locationsp, 1, 1, 3, p_height, p_width, beta)
        
        p_heightp, p_widthp = inc_convolution_bn(self.b_pool_1, self.branch_pool_op[0].weight.data,
                                               self.branch_pool_op[1].running_mean.data,
                                               self.branch_pool_op[1].running_var.data,
                                               self.branch_pool_op[1].weight.data,
                                               self.branch_pool_op[1].bias.data,
                                               self.b_pool_2.data, locationsp, 0, 1, p_heightp, p_widthp, beta, relu=True,
                                               eps=1e-3)

        # 5x5
        p_height, p_width = inc_convolution_bn(x, self.b5_1_op[0].weight.data,
                                                 self.b5_1_op[1].running_mean.data,
                                                 self.b5_1_op[1].running_var.data,
                                                 self.b5_1_op[1].weight.data,
                                                 self.b5_1_op[1].bias.data,
                                                 self.b5_1.data, locations, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)

        p_height, p_width = inc_convolution_bn(self.b5_1.data, self.b5_2_op[0].weight.data,
                                               self.b5_2_op[1].running_mean.data,
                                               self.b5_2_op[1].running_var.data,
                                               self.b5_2_op[1].weight.data,
                                               self.b5_2_op[1].bias.data,
                                               self.b5_2.data, locations, 2, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)

        x = torch.cat([self.b1, self.b5_2, self.b3_3, self.b_pool_2], 1)
        return x, p_height, p_width

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

        b_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)#F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [b3, b3_db, b_pool]
        x = torch.cat(outputs, 1)
        return x
    
    
    def forward_materialized(self, x):
        self.b3 = self.b3_op(x)

        self.b3_db_1 = self.b3_db_1_op(x)
        self.b3_db_2 = self.b3_db_2_op(self.b3_db_1)
        self.b3_db_3 = self.b3_db_3_op(self.b3_db_2)

        self.b_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)#F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [self.b3, self.b3_db_3, self.b_pool]
        return torch.cat(outputs, 1)

    def forward_inc_v2(self, x, locations, p_height, p_width, beta=1.0):

        # 3x3
        locations3 = locations.clone()
        p_height3, p_width3 = inc_convolution_bn(x, self.b3_op[0].weight.data,
                                               self.b3_op[1].running_mean.data,
                                               self.b3_op[1].running_var.data,
                                               self.b3_op[1].weight.data,
                                               self.b3_op[1].bias.data,
                                               self.b3.data, locations3, 0, 2, p_height, p_width, beta, relu=True,
                                               eps=1e-3)

        # pool
        locationsp = locations.clone()
        p_heightp, p_widthp = inc_max_pool(x, self.b_pool, locationsp, 0, 2, 3, p_height, p_width, beta)
        
        # 3x3_db
        p_height, p_width = inc_convolution_bn(x, self.b3_db_1_op[0].weight.data,
                                                 self.b3_db_1_op[1].running_mean.data,
                                                 self.b3_db_1_op[1].running_var.data,
                                                 self.b3_db_1_op[1].weight.data,
                                                 self.b3_db_1_op[1].bias.data,
                                                 self.b3_db_1.data, locations, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        p_height, p_width = inc_convolution_bn(self.b3_db_1.data, self.b3_db_2_op[0].weight.data,
                                                 self.b3_db_2_op[1].running_mean.data,
                                                 self.b3_db_2_op[1].running_var.data,
                                                 self.b3_db_2_op[1].weight.data,
                                                 self.b3_db_2_op[1].bias.data,
                                                 self.b3_db_2.data, locations, 1, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        p_height, p_width = inc_convolution_bn(self.b3_db_2.data, self.b3_db_3_op[0].weight.data,
                                                 self.b3_db_3_op[1].running_mean.data,
                                                 self.b3_db_3_op[1].running_var.data,
                                                 self.b3_db_3_op[1].weight.data,
                                                 self.b3_db_3_op[1].bias.data,
                                                 self.b3_db_3.data, locations, 0, 2, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)

        outputs = [self.b3, self.b3_db_3, self.b_pool]
        x = torch.cat(outputs, 1)
        return x, p_height, p_width


class InceptionC(nn.Module):

    def __init__(self, in_channels, c7):
        super(InceptionC, self).__init__()
        
        self.b1_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b1_inc_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))        
        
        self.b7_1_op = nn.Sequential(nn.Conv2d(in_channels, c7, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_1_inc_op = nn.Sequential(nn.Conv2d(in_channels, c7, kernel_size=1),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_2_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_2_inc_op = nn.Sequential(nn.Conv2d(c7, c7, kernel_size=(1, 7),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_3_op = nn.Sequential(nn.Conv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0),
                                     nn.BatchNorm2d(c7, eps=0.001), nn.ReLU(inplace=True))
        self.b7_3_inc_op = nn.Sequential(nn.Conv2d(c7, 192, kernel_size=(7, 1),
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

        b_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)#F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        b_pool = self.branch_pool_op(b_pool)

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

        self.b_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)#F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.b_pool_2 = self.branch_pool_op(self.b_pool_1)

        outputs = [self.b1, self.b7_3, self.b7_db_5, self.b_pool_2]
        return torch.cat(outputs, 1)


    def forward_gpu(self, x, locations, p_height, p_width, beta=1.0):
        # 1x1
        locations1 = locations.clone()
        p_height1, p_width1 = inc_convolution_bn(x, self.b1_op[0].weight.data,
                                                 self.b1_op[1].running_mean.data,
                                                 self.b1_op[1].running_var.data,
                                                 self.b1_op[1].weight.data,
                                                 self.b1_op[1].bias.data,
                                                 self.b1.data, locations1, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        # 7x7
        locations7 = locations.clone()
        p_height7, p_width7 = inc_convolution_bn(x, self.b7_1_op[0].weight.data,
                                                 self.b7_1_op[1].running_mean.data,
                                                 self.b7_1_op[1].running_var.data,
                                                 self.b7_1_op[1].weight.data,
                                                 self.b7_1_op[1].bias.data,
                                                 self.b7_1.data, locations7, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        
        p_height7, p_width7 = inc_convolution_bn2(self.b7_1.data, self.b7_2_op[0].weight.data,
                                                 self.b7_2_op[1].running_mean.data,
                                                 self.b7_2_op[1].running_var.data,
                                                 self.b7_2_op[1].weight.data,
                                                 self.b7_2_op[1].bias.data,
                                                 self.b7_2.data, locations7, 0, 3, 1, p_height7, p_width7, beta, relu=True,
                                                 eps=1e-3)
        p_height7, p_width7 = inc_convolution_bn2(self.b7_2.data, self.b7_3_op[0].weight.data,
                                                 self.b7_3_op[1].running_mean.data,
                                                 self.b7_3_op[1].running_var.data,
                                                 self.b7_3_op[1].weight.data,
                                                 self.b7_3_op[1].bias.data,
                                                 self.b7_3.data, locations7, 3, 0, 1, p_height7, p_width7, beta, relu=True,
                                                 eps=1e-3)
        # pool
        locationsp = locations.clone()
        p_heightp, p_widthp = inc_avg_pool(x, self.b_pool_1, locationsp, 1, 1, 3, p_height, p_width, beta)
        p_heightp, p_widthp = inc_convolution_bn(self.b_pool_1, self.branch_pool_op[0].weight.data,
                                               self.branch_pool_op[1].running_mean.data,
                                               self.branch_pool_op[1].running_var.data,
                                               self.branch_pool_op[1].weight.data,
                                               self.branch_pool_op[1].bias.data,
                                               self.b_pool_2.data, locationsp, 0, 1, p_heightp, p_widthp, beta, relu=True,
                                               eps=1e-3)
        
        # 7x7 db
        p_height, p_width = inc_convolution_bn(x, self.b7_db_1_op[0].weight.data,
                                                 self.b7_db_1_op[1].running_mean.data,
                                                 self.b7_db_1_op[1].running_var.data,
                                                 self.b7_db_1_op[1].weight.data,
                                                 self.b7_db_1_op[1].bias.data,
                                                 self.b7_db_1.data, locations, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        p_height, p_width = inc_convolution_bn2(self.b7_db_1.data, self.b7_db_2_op[0].weight.data,
                                               self.b7_db_2_op[1].running_mean.data,
                                               self.b7_db_2_op[1].running_var.data,
                                               self.b7_db_2_op[1].weight.data,
                                               self.b7_db_2_op[1].bias.data,
                                               self.b7_db_2.data, locations, 3, 0, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)
        p_height, p_width = inc_convolution_bn2(self.b7_db_2.data, self.b7_db_3_op[0].weight.data,
                                               self.b7_db_3_op[1].running_mean.data,
                                               self.b7_db_3_op[1].running_var.data,
                                               self.b7_db_3_op[1].weight.data,
                                               self.b7_db_3_op[1].bias.data,
                                               self.b7_db_3.data, locations, 0, 3, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)
        p_height, p_width = inc_convolution_bn2(self.b7_db_3.data, self.b7_db_4_op[0].weight.data,
                                               self.b7_db_4_op[1].running_mean.data,
                                               self.b7_db_4_op[1].running_var.data,
                                               self.b7_db_4_op[1].weight.data,
                                               self.b7_db_4_op[1].bias.data,
                                               self.b7_db_4.data, locations, 3, 0, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)
        p_height, p_width = inc_convolution_bn2(self.b7_db_4.data, self.b7_db_5_op[0].weight.data,
                                               self.b7_db_5_op[1].running_mean.data,
                                               self.b7_db_5_op[1].running_var.data,
                                               self.b7_db_5_op[1].weight.data,
                                               self.b7_db_5_op[1].bias.data,
                                               self.b7_db_5.data, locations, 0, 3, 1, p_height, p_width, beta, relu=True,
                                               eps=1e-3)

        outputs = [self.b1, self.b7_3, self.b7_db_5, self.b_pool_2]
        return torch.cat(outputs, 1), p_height, p_width    

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

        b_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)#F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [b3, b7, b_pool]
        return torch.cat(outputs, 1)

    
    def forward_materialized(self, x):
        self.b3_1 = self.b3_1_op(x)
        self.b3_2 = self.b3_2_op(self.b3_1)

        self.b7_1 = self.b7_1_op(x)
        self.b7_2 = self.b7_2_op(self.b7_1)
        self.b7_3 = self.b7_3_op(self.b7_2)
        self.b7_4 = self.b7_4_op(self.b7_3)

        self.b_pool = nn.MaxPool2d(kernel_size=3, stride=2)(x)#F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [self.b3_2, self.b7_4, self.b_pool]
        return torch.cat(outputs, 1)


    def forward_inc_v2(self, x, locations, p_height, p_width, beta=1.0):
        # 3x3
        locations3 = locations.clone()
        p_height3, p_width3 = inc_convolution_bn(x, self.b3_1_op[0].weight.data,
                                                 self.b3_1_op[1].running_mean.data,
                                                 self.b3_1_op[1].running_var.data,
                                                 self.b3_1_op[1].weight.data,
                                                 self.b3_1_op[1].bias.data,
                                                 self.b3_1.data, locations3, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        p_height3, p_width3 = inc_convolution_bn(self.b3_1.data, self.b3_2_op[0].weight.data,
                                                 self.b3_2_op[1].running_mean.data,
                                                 self.b3_2_op[1].running_var.data,
                                                 self.b3_2_op[1].weight.data,
                                                 self.b3_2_op[1].bias.data,
                                                 self.b3_2.data, locations3, 0, 2, p_height3, p_width3, beta, relu=True,
                                                 eps=1e-3)

        locationsp = locations.clone()
        p_heightp, p_widthp = inc_max_pool(x, self.b_pool, locationsp, 0, 2, 3, p_height, p_width, beta)
       
        
        # 7x7
        p_height, p_width = inc_convolution_bn(x, self.b7_1_op[0].weight.data,
                                                 self.b7_1_op[1].running_mean.data,
                                                 self.b7_1_op[1].running_var.data,
                                                 self.b7_1_op[1].weight.data,
                                                 self.b7_1_op[1].bias.data,
                                                 self.b7_1.data, locations, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        p_height, p_width = inc_convolution_bn2(self.b7_1.data, self.b7_2_op[0].weight.data,
                                                  self.b7_2_op[1].running_mean.data,
                                                  self.b7_2_op[1].running_var.data,
                                                  self.b7_2_op[1].weight.data,
                                                  self.b7_2_op[1].bias.data,
                                                  self.b7_2.data, locations, 0, 3, 1, p_height, p_width, beta,
                                                  relu=True,
                                                  eps=1e-3)
        p_height, p_width = inc_convolution_bn2(self.b7_2.data, self.b7_3_op[0].weight.data,
                                                  self.b7_3_op[1].running_mean.data,
                                                  self.b7_3_op[1].running_var.data,
                                                  self.b7_3_op[1].weight.data,
                                                  self.b7_3_op[1].bias.data,
                                                  self.b7_3.data, locations, 3, 0, 1, p_height, p_width, beta,
                                                  relu=True,
                                                  eps=1e-3)
        p_height, p_width = inc_convolution_bn(self.b7_3.data, self.b7_4_op[0].weight.data,
                                                  self.b7_4_op[1].running_mean.data,
                                                  self.b7_4_op[1].running_var.data,
                                                  self.b7_4_op[1].weight.data,
                                                  self.b7_4_op[1].bias.data,
                                                  self.b7_4.data, locations, 0, 2, p_height, p_width, beta,
                                                  relu=True,
                                                  eps=1e-3)
 
        
        outputs = [self.b3_2, self.b7_4, self.b_pool]
        return torch.cat(outputs, 1), p_height, p_width


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

        self.branch_pool_op = nn.Sequential(nn.Conv2d(in_channels, 192, kernel_size=1, bias=False),
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

        branch_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)#F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool_op(branch_pool)
    
        outputs = [b1, b3, b3_db, branch_pool]
        return torch.cat(outputs, 1)
    
    def forward_materialized(self, x):
        self.b1 = self.b1_op(x)

        self.b3_1 = self.b3_1_op(x)
        self.b3_2a = self.b3_2a_op(self.b3_1)
        self.b3_2b = self.b3_2b_op(self.b3_1)
        b3 = [
            self.b3_2a,
            self.b3_2b,
        ]
        b3 = torch.cat(b3, 1)

        self.b3_db_1 = self.b3_db_1_op(x)
        self.b3_db_2 = self.b3_db_2_op(self.b3_db_1)
        self.b3_db_3a = self.b3_db_3a_op(self.b3_db_2)
        self.b3_db_3b = self.b3_db_3b_op(self.b3_db_2)

        b3_db = [
            self.b3_db_3a,
            self.b3_db_3b,
        ]
        b3_db = torch.cat(b3_db, 1)

        self.b_pool_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(x)#F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        self.b_pool_2 = self.branch_pool_op(self.b_pool_1)

        outputs = [self.b1, b3, b3_db, self.b_pool_2]
        return torch.cat(outputs, 1)


    def forward_inc_v2(self, x, locations, p_height, p_width, beta=1.0):
        # 1x1
        locations1 = locations.clone()
        p_height1, p_width1 = inc_convolution_bn(x, self.b1_op[0].weight.data,
                                                 self.b1_op[1].running_mean.data,
                                                 self.b1_op[1].running_var.data,
                                                 self.b1_op[1].weight.data,
                                                 self.b1_op[1].bias.data,
                                                 self.b1.data, locations1, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        # 3x3
        locations3 = locations.clone()
        p_height3, p_width3 = inc_convolution_bn(x, self.b3_1_op[0].weight.data,
                                                 self.b3_1_op[1].running_mean.data,
                                                 self.b3_1_op[1].running_var.data,
                                                 self.b3_1_op[1].weight.data,
                                                 self.b3_1_op[1].bias.data,
                                                 self.b3_1.data, locations3, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        locations3_1 = locations3.clone()
        p_height3_2a, p_width3_2a = inc_convolution_bn2(self.b3_1.data, self.b3_2a_op[0].weight.data,
                                                 self.b3_2a_op[1].running_mean.data,
                                                 self.b3_2a_op[1].running_var.data,
                                                 self.b3_2a_op[1].weight.data,
                                                 self.b3_2a_op[1].bias.data,
                                                 self.b3_2a.data, locations3_1, 0, 1, 1, p_height3, p_width3, beta, relu=True,
                                                 eps=1e-3)
        locations3_1 = locations3.clone()
        p_height3_2b, p_width3_2b = inc_convolution_bn2(self.b3_1.data, self.b3_2b_op[0].weight.data,
                                                 self.b3_2b_op[1].running_mean.data,
                                                 self.b3_2b_op[1].running_var.data,
                                                 self.b3_2b_op[1].weight.data,
                                                 self.b3_2b_op[1].bias.data,
                                                 self.b3_2b.data, locations3_1, 1, 0, 1, p_height3, p_width3, beta, relu=True,
                                                 eps=1e-3)


        b3 = torch.cat([self.b3_2a, self.b3_2b], 1)

        # pool
        locationsp = locations.clone()
        p_heightp, p_widthp = inc_avg_pool(x, self.b_pool_1, locationsp, 1, 1, 3, p_height, p_width, beta)
        p_heightp, p_widthp = inc_convolution_bn(self.b_pool_1, self.branch_pool_op[0].weight.data,
                                               self.branch_pool_op[1].running_mean.data,
                                               self.branch_pool_op[1].running_var.data,
                                               self.branch_pool_op[1].weight.data,
                                               self.branch_pool_op[1].bias.data,
                                               self.b_pool_2.data, locationsp, 0, 1, p_heightp, p_widthp, beta, relu=True,
                                               eps=1e-3)

        # 3x3 db
        p_height, p_width = inc_convolution_bn(x, self.b3_db_1_op[0].weight.data,
                                                 self.b3_db_1_op[1].running_mean.data,
                                                 self.b3_db_1_op[1].running_var.data,
                                                 self.b3_db_1_op[1].weight.data,
                                                 self.b3_db_1_op[1].bias.data,
                                                 self.b3_db_1.data, locations, 0, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)
        p_height, p_width = inc_convolution_bn(self.b3_db_1.data, self.b3_db_2_op[0].weight.data,
                                                 self.b3_db_2_op[1].running_mean.data,
                                                 self.b3_db_2_op[1].running_var.data,
                                                 self.b3_db_2_op[1].weight.data,
                                                 self.b3_db_2_op[1].bias.data,
                                                 self.b3_db_2.data, locations, 1, 1, p_height, p_width, beta, relu=True,
                                                 eps=1e-3)

        locations3_db = locations.clone()
        p_height3_db_3a, p_width3_db_3a = inc_convolution_bn2(self.b3_db_2.data, self.b3_db_3a_op[0].weight.data,
                                                        self.b3_db_3a_op[1].running_mean.data,
                                                        self.b3_db_3a_op[1].running_var.data,
                                                        self.b3_db_3a_op[1].weight.data,
                                                        self.b3_db_3a_op[1].bias.data,
                                                        self.b3_db_3a.data, locations3_db, 0, 1, 1, p_height, p_width,
                                                        beta, relu=True,
                                                        eps=1e-3)
        locations3_db = locations.clone()
        p_height3_db_3b, p_width3_db_3b = inc_convolution_bn2(self.b3_db_2.data, self.b3_db_3b_op[0].weight.data,
                                                        self.b3_db_3b_op[1].running_mean.data,
                                                        self.b3_db_3b_op[1].running_var.data,
                                                        self.b3_db_3b_op[1].weight.data,
                                                        self.b3_db_3b_op[1].bias.data,
                                                        self.b3_db_3b.data, locations3_db, 1, 0, 1, p_height, p_width,
                                                        beta, relu=True,
                                                        eps=1e-3)
        
        b3_db = torch.cat([self.b3_db_3a.data, self.b3_db_3b.data], 1)
        outputs = [self.b1, b3, b3_db, self.b_pool_2]
        x = torch.cat(outputs, 1)
        
        p_height, p_width = update_output_locations(locations, 1, 1, 1, 3, 3, p_height, p_width, self.b3_db_2.shape[2], self.b3_db_3b.shape[2], beta)
        return x, p_height, p_width

    

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
