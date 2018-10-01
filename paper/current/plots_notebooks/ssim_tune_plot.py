from __future__ import print_function, division

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from skimage.measure import compare_ssim as ssim
import time
import os
import copy
import h5py
import sys
import math
import random
import torch
import scipy
import numpy.polynomial.polynomial as poly
from PIL import Image
import gc

sys.path.append('../../../code')

from python.finetune_commons import show_images, ft_train_model, visualize_model
from python.commons import load_dict_from_hdf5, save_dict_to_hdf5, inc_inference_e2e, full_inference_e2e, adaptive_drilldown, generate_heatmap
from python.vgg16 import VGG16
from python.resnet18 import ResNet18
from python.inception3 import Inception3

random.seed(256)

for dataset in ['oct', 'chest_xray', 'imagenet']:
    gpu = True
    batch_size = 256
    patch_size = 16
    stride = 4
    file_amount = 60

    log_file = open("ssim_tune_"+dataset+".log", 'w')
    log_file.write("==============================================\n")
    log_file.flush()

    def inc_inference(model, image_file_path, beta, patch_size=4, stride=1,
                      adaptive=False, weights_data=None, image_size = 224):
        if gpu:
            torch.cuda.synchronize()

        if not adaptive:
            with torch.no_grad():
                x, prob, logit_index = inc_inference_e2e(model, image_file_path, patch_size, stride,
                                      batch_size=batch_size, beta=beta, gpu=gpu, version='v1',
                                      weights_data=weights_data, c=0.0,
                                     image_size=image_size, x_size=image_size, y_size=image_size)

        if gpu:
            torch.cuda.synchronize()

        return x, prob, logit_index


    image_files = []
    # OCT
    if dataset == 'oct':
        temp = os.listdir('../../../data/oct/test/DRUSEN')
        for name in temp:
            if name.endswith('jpeg'):
                image_files.append('../../../data/oct/test/DRUSEN/'+name)

        temp = os.listdir('../../../data/oct/test/DME')
        for name in temp:
            if name.endswith('jpeg'):
                image_files.append('../../../data/oct/test/DME/'+name)

        temp = os.listdir('../../../data/oct/test/CNV')
        for name in temp:
            if name.endswith('jpeg'):
                image_files.append('../../../data/oct/test/CNV/'+name)


        temp = os.listdir('../../../data/oct/test/NORMAL')        
        for name in temp:
            if name.endswith('jpeg'):
                image_files.append('../../../data/oct/test/NORMAL/'+name)

    elif dataset == 'chest_xray':
        temp = os.listdir('../../../data/chest_xray/test/BACTERIAL')
        for name in temp:
            if name.endswith('jpeg'):
                image_files.append('../../../data/chest_xray/test/BACTERIAL/'+name)

        temp = os.listdir('../../../data/chest_xray/test/VIRAL')
        for name in temp:
            if name.endswith('jpeg'):
                image_files.append('../../../data/chest_xray/test/VIRAL/'+name)

        temp = os.listdir('../../../data/chest_xray/test/NORMAL')
        for name in temp:
            if name.endswith('jpeg'):
                image_files.append('../../../data/chest_xray/test/NORMAL/'+name)


    elif dataset == 'imagenet':
    # ImageNet
        temp = os.listdir('../../../data/imagenet-sample')
        for name in temp:
            if name.endswith('jpg'):
                image_files.append('../../../data/imagenet-sample/'+name)

    image_files = random.sample(image_files, file_amount)
    random.shuffle(image_files)

    train_files = image_files[:file_amount//2]
    test_files = image_files[file_amount//2:]

    taus = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    plt.figure(figsize=(9,2.5))

    i = 1

    ssim_curve_fits = {}
    time_curve_fits = {}

    for model,model_name,weight_file in zip([VGG16, ResNet18, Inception3], ['VGG16', 'ResNet18', 'Inception3'],
        ['../../../code/python/vgg16_weights_ptch.h5', '../../../code/python/resnet18_weights_ptch.h5',
            '../../../code/python/inception3_weights_ptch.h5']):

        weights_data = load_dict_from_hdf5(weight_file, gpu=gpu)

        if dataset in ['oct', 'chest_xray']:
            temp_weights_data = load_dict_from_hdf5('../../../exps/'+dataset+'_'+model_name.lower()+'_ptch.h5', gpu=gpu)

        if model_name == 'VGG16':
            #continue
            image_size = 224
            if dataset in ['oct', 'chest_xray']:
                weights_data['fc8_W:0'] = temp_weights_data['fc8_W:0']
                weights_data['fc8_b:0'] = temp_weights_data['fc8_b:0']
        elif model_name == 'ResNet18':
            image_size = 224
            if dataset in ['oct', 'chest_xray']:
                weights_data['fc:w'] = temp_weights_data['fc:w']
                weights_data['fc:b'] = temp_weights_data['fc:b']
        elif model_name == 'Inception3':
            #continue
            image_size = 299
            if dataset in ['oct', 'chest_xray']:
                weights_data['482.fc.weight'] = temp_weights_data['482.fc.weight']
                weights_data['483.fc.bias'] = temp_weights_data['483.fc.bias']

        y_ssim = []
        y_time = []
        x_vals = []

        for file_path in train_files:

            x, prob, logit_index = inc_inference(model, file_path, 1.0, patch_size=patch_size, stride=stride,
                                 weights_data=weights_data, image_size=image_size)

            orig_hm = generate_heatmap(file_path, x, show=False, label="", prob=prob, width=image_size)

            for beta in taus:
                prev_time = time.time()
                x, prob, logit_index = inc_inference(model, file_path, beta, patch_size=patch_size, stride=stride,
                                 weights_data=weights_data, image_size=image_size)
                inc_inference_time = time.time()-prev_time
                hm = generate_heatmap(file_path, x, show=False, label="", prob=prob, width=image_size)

                if hm.shape[0] < 7:
                    win_size=3
                else:
                    win_size=None

                ssim_value = ssim(orig_hm, hm, data_range=255, multichannel=True, win_size=win_size)
                x_vals.append(beta)
                y_ssim.append(ssim_value)
                y_time.append(inc_inference_time)
                log_file.write(",".join([str(x) for x in [model_name, file_path, beta]])+"\n")
                log_file.flush()

                gc.collect()
                torch.cuda.empty_cache()

        ssim_curve_fits[model_name] = poly.polyfit(
            np.array(x_vals), np.array(y_ssim), 2)


        ax = plt.subplot(1,3,i)

        plt.scatter(x_vals, y_ssim)
        ax.plot(taus, [poly.Polynomial(ssim_curve_fits[model_name])(t) for t in taus], label='fit', color='orange')

        ax.set_title(model_name)
        #plt.xlabel(r'$\tau$')

        plt.grid()
        plt.xticks(taus, taus)

        if i == 1:
            plt.ylabel('SSIM')
            plt.legend(loc='lower right', ncol=1)


        i = i + 1

    plt.subplots_adjust(top=0.8)
    plt.savefig('../images/ssim_tau_'+dataset+'_test.pdf', bbox_inches='tight')


    plt.figure(figsize=(9,2.5))

    i = 1

    for model,model_name,weight_file in zip([VGG16, ResNet18, Inception3], ['VGG16', 'ResNet18', 'Inception3'],
        ['../../../code/python/vgg16_weights_ptch.h5', '../../../code/python/resnet18_weights_ptch.h5',
            '../../../code/python/inception3_weights_ptch.h5']):

        weights_data = load_dict_from_hdf5(weight_file, gpu=gpu)
        if dataset in ['oct', 'chest_xray']:
            temp_weights_data = load_dict_from_hdf5('../../../exps/'+dataset+'_'+model_name.lower()+'_ptch.h5', gpu=gpu)

        if model_name == 'VGG16':
            #continue
            image_size = 224
            if dataset in ['oct', 'chest_xray']:
                weights_data['fc8_W:0'] = temp_weights_data['fc8_W:0']
                weights_data['fc8_b:0'] = temp_weights_data['fc8_b:0']
        elif model_name == 'ResNet18':
            #continue
            image_size = 224
            if dataset in ['oct', 'chest_xray']:
                weights_data['fc:w'] = temp_weights_data['fc:w']
                weights_data['fc:b'] = temp_weights_data['fc:b']
        elif model_name == 'Inception3':
            #continue
            image_size = 299
            if dataset in ['oct', 'chest_xray']:
                weights_data['482.fc.weight'] = temp_weights_data['482.fc.weight']
                weights_data['483.fc.bias'] = temp_weights_data['483.fc.bias']


        y_ssim = []

        ssim_threshold = 0.9

        coeffs = np.flip(ssim_curve_fits[model_name], 0)
        coeffs[2] = coeffs[2] - ssim_threshold
        roots = np.roots(coeffs)

        if 0 < roots[0] <= 1.0 :
            ssim_tau = roots[0]
        else:
            ssim_tau = roots[1]

        ssim_tau = max(round((ssim_tau)*10)/10, 0.4)

        for name, files in zip(['train', 'test'], [train_files, test_files]):

            for file_path in files:
                x, prob, logit_index = inc_inference(model, file_path, 1.0, patch_size=patch_size, stride=stride,
                                     weights_data=weights_data, image_size=image_size)

                orig_hm = generate_heatmap(file_path, x, show=False, label="", prob=prob, width=image_size)

                x, prob, logit_index = inc_inference(model, file_path, ssim_tau, patch_size=patch_size, stride=stride,
                                 weights_data=weights_data, image_size=image_size)

                hm = generate_heatmap(file_path, x, show=False, label="", prob=prob, width=image_size)

                if hm.shape[0] < 7:
                    win_size=3
                else:
                    win_size=None

                ssim_value = ssim(orig_hm, hm, data_range=255, multichannel=True, win_size=win_size)
                y_ssim.append(ssim_threshold-ssim_value)

                log_file.write(",".join([str(x) for x in [model_name, file_path]])+"\n")
                log_file.flush()

            ax = plt.subplot(1,3,i)

            unique, counts = np.unique(y_ssim, return_counts=True)

            plt.plot(unique, np.cumsum(counts)/np.sum(counts)*100.0, label=name)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            #plt.xlabel(r'$SSIM_{target}-SSIM_{actual}$', fontsize=8)
            ax.xaxis.major.formatter._useMathText = True

        if i == 1:
            plt.ylabel('Cumulative Percent')
            plt.legend(loc='lower right', ncol=1)


        ax.set_title(r'$\tau$='+str(ssim_tau))
        plt.grid()

        i+=1

    plt.savefig('../images/ssim_cdf_'+dataset+'_test.pdf', bbox_inches='tight')    