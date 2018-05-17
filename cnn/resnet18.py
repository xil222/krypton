import numpy as np
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils import model_zoo
from torchvision.transforms import transforms
import torch.nn.functional as F
import os 

import sys
sys.path.append('../')

from cnn.imagenet_classes import class_names
from cnn.commons import load_dict_from_hdf5


class ResNet18(nn.Module):

    def __init__(self, gpu=True, num_classes=1000):
        super(ResNet18, self).__init__()
        
        self.gpu = gpu
        
        #layer1
        self.conv1_op = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1_op = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #layer2
        self.conv2_1_a_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_1_b_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64))
        
        self.conv2_2_a_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_2_b_op = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64))
        
        
        #layer3
        self.conv3_1_a_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3_1_b_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128))
        
        self.residual_3_op = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(128))
        
        self.conv3_2_a_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3_2_b_op = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128))
        
              
        #layer4
        self.conv4_1_a_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv4_1_b_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256))

        self.residual_4_op = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(256))

        
        self.conv4_2_a_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv4_2_b_op = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256))
          
        #layer5
        self.conv5_1_a_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv5_1_b_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512))

        self.residual_5_op = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(512))

        
        self.conv5_2_a_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.conv5_2_b_op = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512))
        
       
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights(gpu)
        
        
    def forward(self, x):
        return self.forward_fused(x)

    def forward_fused(self, x):
        x = self.conv1_op(x)
        x = self.pool1_op(x)

        residual = x
        x = self.conv2_1_a_op(x)
        x = self.conv2_1_b_op(x)
        x = F.relu(x + residual)
        residual = x
        x = self.conv2_2_a_op(x)
        x = self.conv2_2_b_op(x)
        x = F.relu(x + residual)

        residual = self.residual_3_op(x)
        x = self.conv3_1_a_op(x)   
        x = self.conv3_1_b_op(x)
        x = F.relu(x + residual)
        residual = x
        x = self.conv3_2_a_op(x)
        x = self.conv3_2_b_op(x)
        x = F.relu(x + residual)
        
        residual = self.residual_4_op(x)
        x = self.conv4_1_a_op(x)
        x = self.conv4_1_b_op(x)
        x = F.relu(x + residual)
        residual = x
        x = self.conv4_2_a_op(x)
        x = self.conv4_2_b_op(x)
        x = F.relu(x + residual)
    
        residual = self.residual_5_op(x)
        x = self.conv5_1_a_op(x)
        x = self.conv5_1_b_op(x)
        x = F.relu(x + residual)
        residual = x
        x = self.conv5_2_a_op(x)
        x = self.conv5_2_b_op(x)
        x = F.relu(x + residual)        
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    
        return x
    
    def forward_materialized(self, x):
        if self.cuda:
            x = x.cuda()

        self.conv1 = self.conv1_op(x)
        self.pool1 = self.pool1_op(self.conv1)
        
        self.conv2_1_a = self.conv2_1_a_op(self.pool1)
        self.conv2_1_b = self.conv2_1_b_op(self.conv2_1_a)
        self.merge_2_1 = F.relu(self.conv2_1_b + self.pool1)
        self.conv2_2_a = self.conv2_2_a_op(self.merge_2_1)
        self.conv2_2_b = self.conv2_2_b_op(self.conv2_2_a)
        self.merge_2_2 = F.relu(self.conv2_2_b + self.merge_2_1)
        
        self.residual_3 = self.residual_3_op(self.merge_2_2)
        self.conv3_1_a = self.conv3_1_a_op(self.merge_2_2)
        self.conv3_1_b = self.conv3_1_b_op(self.conv3_1_a)
        self.merge_3_1 = F.relu(self.conv3_1_b + self.residual_3)
        self.conv3_2_a = self.conv3_2_a_op(self.merge_3_1)
        self.conv3_2_b = self.conv3_2_b_op(self.conv3_2_a)
        self.merge_3_2 = F.relu(self.conv3_2_b + self.merge_3_1)

        self.residual_4 = self.residual_4_op(self.merge_3_2)
        self.conv4_1_a = self.conv4_1_a_op(self.merge_3_2)
        self.conv4_1_b = self.conv4_1_b_op(self.conv4_1_a)
        self.merge_4_1 = F.relu(self.conv4_1_b + self.residual_4)
        self.conv4_2_a = self.conv4_2_a_op(self.merge_4_1)
        self.conv4_2_b = self.conv4_2_b_op(self.conv4_2_a)
        self.merge_4_2 = F.relu(self.conv4_2_b + self.merge_4_1)

        self.residual_5 = self.residual_5_op(self.merge_4_2)
        self.conv5_1_a = self.conv5_1_a_op(self.merge_4_2)
        self.conv5_1_b = self.conv5_1_b_op(self.conv5_1_a)
        self.merge_5_1 = F.relu(self.conv5_1_b + self.residual_5)
        self.conv5_2_a = self.conv5_2_a_op(self.merge_5_1)
        self.conv5_2_b = self.conv5_2_b_op(self.conv5_2_a)
        self.merge_5_2 = F.relu(self.conv5_2_b + self.merge_5_1)
        
        
        x = self.avgpool(self.merge_5_2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def _initialize_weights(self, gpu):
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

if __name__ == "__main__":
    batch_size = 64

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')

    images = loader(images)
    images = Variable(images.unsqueeze(0), requires_grad=False, volatile=True).cuda()

    images = images.repeat(batch_size, 1, 1, 1)

    model = ResNet18().cuda()
    model.eval()
   
    x = model(images)
    print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])