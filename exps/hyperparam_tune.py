from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import time
import os
import copy
import h5py
import sys
import torch
from PIL import Image
import gc

sys.path.append('../code')

from python.finetune_commons import show_images, ft_train_model, visualize_model
from python.commons import load_dict_from_hdf5, save_dict_to_hdf5, inc_inference_e2e, full_inference_e2e, adaptive_drilldown, generate_heatmap
from python.vgg16 import VGG16
from python.resnet18 import ResNet18
from python.inception3 import Inception3

from python.models import inception


torch.set_num_threads(8)

torch.manual_seed(245)
np.random.seed(345)

dataset = 'chest_xray'
data_dir = '../data/'+dataset

n_labels = 3

num_epochs = 25


for name, model, image_size, in zip(['VGG16', 'ResNet18', 'Inception3'],
                                    [models.vgg16, models.resnet18, inception.inception_v3],
                                    [224, 224, 299]):
    
    transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    data_transforms = {'train': transform, 'validation': transform, 'test': transform}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'validation', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                            shuffle=True, num_workers=8) for x in ['train', 'validation', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    best_acc = 0
    
    #for lr in [1e-2, 1e-4, 1e-6]:
    #for reg in [1e-2, 1e-4, 1e-6]:
    
    model_ft = model(pretrained=True)

    if name == 'VGG16':
        #lr = 1e-4
        #reg = 1e-4
        
        for param in model_ft.features.parameters():
            param.requires_grad = False

        num_ftrs = model_ft.classifier[-1].in_features
        temp = list(model_ft.classifier.children())[:-1]
        temp.append(nn.Linear(num_ftrs, n_labels))
        model_ft.classifier = nn.Sequential(*temp)

        for param in model_ft.classifier[0:-1].parameters():
            param.requires_grad = False


        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.classifier[-1].parameters(), lr=lr, weight_decay=reg)
    elif name == 'ResNet18':
        #lr = 1e-4
        #reg = 1e-6
        
        for param in model_ft.parameters():
            param.requires_grad = False

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_labels)

        for param in model_ft.fc.parameters():
            param.requires_grad = True


        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=lr, weight_decay=reg)
    elif name == 'Inception3':
        #lr = 1e-4
        #reg = 1e-2
        
        model_ft.aux_logit = False
        model_ft = model_ft.eval()
        for param in model_ft.parameters():
            param.requires_grad = False

        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_labels)

        for param in model_ft.fc.parameters():
            param.requires_grad = True

        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=lr, weight_decay=reg)



    model_ft, current_acc = ft_train_model(model_ft, criterion, optimizer_ft, dataloaders, device,
                  dataset_sizes, class_names, num_epochs=num_epochs)

    if best_acc < current_acc:
        final_layer = {}

        if name == 'VGG16':
            final_layer['fc8_W:0'] = model_ft.classifier[-1].weight.data.cpu().numpy()
            final_layer['fc8_b:0'] = model_ft.classifier[-1].bias.data.cpu().numpy()
            save_dict_to_hdf5(final_layer, './'+dataset+'_vgg16_ptch.h5')

        elif name == 'ResNet18':
            final_layer['fc:w'] = model_ft.fc.weight.data.cpu().numpy()
            final_layer['fc:b'] = model_ft.fc.bias.data.cpu().numpy()
            save_dict_to_hdf5(final_layer, './'+dataset+'_resnet18_ptch.h5')

        elif name == 'Inception3':
            final_layer['482.fc.weight'] = model_ft.fc.weight.data.cpu().numpy()
            final_layer['483.fc.bias'] = model_ft.fc.bias.data.cpu().numpy()
            save_dict_to_hdf5(final_layer, './'+dataset+'_inception3_ptch.h5')

        best_acc = current_acc

        print("\n"+name+" best hyper-parameters LR: " + str(lr) + " Reg: " + str(reg) + "\n\n")