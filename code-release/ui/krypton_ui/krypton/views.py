from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage

from .forms import *

import string
import random
import json
import sys
import time
import os
import cv2
from PIL import Image, ImageOps

# sys.path.append('/krypton/code-release/core/python')
sys.path.append('../../../code-release/core/python')


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch, gc
from commons import inc_inference, show_heatmap, full_inference_e2e
from imagenet_classes import class_names
from vgg16 import VGG16
from resnet18 import ResNet18
from inception3 import Inception3
from imagenet_classes import class_names as imagenet_class_names

def index(request):
	return render(request, 'index.html')

def selectedRegion(request):

	message = request.POST
	completeImage = False

	form = PhotoForm(request.POST,request.FILES)
	if form.is_valid():
		photo = form.save()
		data = {'is_valid': True}

	else:
		data = {'is_valid': False}
		print ("photo is not valid" );

	prev_path = '../../../code-release/ui/krypton_ui'

	url = photo.file.url
	curr_path = prev_path + url

	im = Image.open(curr_path)
	width, height = im.size

	model_class = message['model']
	mode = message['mode']
	c = float(message['color'])

	dataset = message['dataset']
	patch_size = (int)(float(message['patchSize']))
	stride_size = (int)(float(message['strideSize']))

	if model_class == "VGG16":
		model_class = VGG16
	elif model_class == "ResNet18":
		model_class = ResNet18
	elif model_class == "Inception":
		model_class = Inception3
	
	calibrated_h = 224
	calibrated_w = 224

	if message['x1'] == "":
		x1 = 0
		x2 = width
		y1 = 0
		y2 = height
		h = height
		w = width
		completeImage = True

	else:
		x1 = (int)(float(message['x1']))
		x2 = (int)(float(message['x2']))
		y1 = (int)(float(message['y1']))
		y2 = (int)(float(message['y2']))

		h = (int)(float(message['h']))
		w = (int)(float(message['w']))

		if model_class != Inception3:
			calibrated_x1 = (int)(x1 * 224 / width)
			calibrated_y1 = (int)(y1 * 224 / height)

			calibrated_w = (int)(w * 224 / width)
			calibrated_h = (int)(h * 224 / height)
			image_size = 224
		else:
			calibrated_x1 = (int)(x1 * 299 / width)
			calibrated_y1 = (int)(y1 * 299 / height)

			calibrated_w = (int)(w * 299 / width)
			calibrated_h = (int)(h * 299 / height)
			image_size = 299


	start_time = time.time()

	if completeImage:
		if mode == "exact":
			if model_class != Inception3:
				heatmap, prob, label = inc_inference(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x_size=224, y_size=224, image_size=224, gpu=True, c=c)
			else:
				heatmap, prob, label = inc_inference(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x_size=299, y_size=299, image_size=299, gpu=True, c=c)
		elif mode == "approximate":
			if model_class != Inception3:
				heatmap, prob, label = inc_inference(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=0.5, x_size=224, y_size=224, image_size=224, gpu=True, c=c)
			else:
				heatmap, prob, label = inc_inference(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=0.7, x_size=299, y_size=299, image_size=299, gpu=True, c=c)
		else:
			if model_class != Inception3:
				heatmap, prob, label = full_inference_e2e(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, batch_size=64, image_size=224, gpu=True, c=c)
			else:
				heatmap, prob, label = full_inference_e2e(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, batch_size=64, x_size=299, y_size=299, image_size=299, gpu=True, c=c)
	else:
		if mode == "exact":
			if model_class != Inception3:
				heatmap, prob, label = inc_inference(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x0=calibrated_y1, y0=calibrated_x1, x_size=calibrated_h, y_size=calibrated_w, image_size=224, gpu=True, c=c)
			else:
				heatmap, prob, label = inc_inference(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x0=calibrated_y1, y0=calibrated_x1, x_size=calibrated_h, y_size=calibrated_w, image_size=299, gpu=True, c=c)
		elif mode == "approximate":
			if model_class != Inception3:
				heatmap, prob, label = inc_inference(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=0.5, x0=calibrated_y1, y0=calibrated_x1, x_size=calibrated_h, y_size=calibrated_w, image_size=224, gpu=True, c=c)
			else:
				heatmap, prob, label = inc_inference(dataset, model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=0.7, x0=calibrated_y1, y0=calibrated_x1, x_size=calibrated_h, y_size=calibrated_w, image_size=299, gpu=True, c=c)


	end_time = time.time()

	heatmap_path = './media/photos/heatmap_' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20)) + '.png'
	low_prob = np.min(heatmap)
	high_prob = np.max(heatmap)

	plt.imsave(heatmap_path, heatmap, cmap=plt.cm.jet_r)
	img = Image.open(heatmap_path)
 	img = img.resize((w,h))
	img.save(heatmap_path)
	img = Image.open(heatmap_path)
	padding = (x1, y1, (width - x2), (height - y2))
	new_img = ImageOps.expand(img, padding)
	new_img.save(heatmap_path)

	actTime = round(end_time - start_time, 2)


	if dataset != 'imagenet':
		class_names = ['Positive', 'Negative']
	else:
		class_names = imagenet_class_names

	return JsonResponse({'url':url, 'heatmap_url': heatmap_path[1:], 'prediction': class_names[label], 'actual_time': actTime, 'low_prob':str(low_prob), 'high_prob':str(high_prob)})


def getTimeEstimate(request):

	file_path_time = '../../../code-release/ui/krypton_ui/krypton/PreTimeEstimation.txt'

	message = request.POST
	model_class = message['model']
	mode = message['mode']
	width = (int)(float(message['width']))
	height = (int)(float(message['height']))
	print "height: " + str(height)

	with open(file_path_time) as f:
		parameters = json.load(f)

	if model_class == "VGG":
		model_class = VGG16
		if mode == "approximate":
			intercept, slope = parameters['vgg16']['approx']['gpu'][0]['intercept'], parameters['vgg16']['approx']['gpu'][0]['slope']
		elif mode == "exact":
			intercept, slope = parameters['vgg16']['exact']['gpu'][0]['intercept'], parameters['vgg16']['exact']['gpu'][0]['slope']
		else:
			intercept, slope = parameters['vgg16']['naive']['gpu'][0]['intercept'], parameters['vgg16']['naive']['gpu'][0]['slope']
	elif model_class == "ResNet":
		model_class = ResNet18
		if mode == "approximate":
			intercept, slope = parameters['resnet18']['approx']['gpu'][0]['intercept'], parameters['resnet18']['approx']['gpu'][0]['slope']
		elif mode == "exact":
			intercept, slope = parameters['resnet18']['exact']['gpu'][0]['intercept'], parameters['resnet18']['exact']['gpu'][0]['slope']
		else:
			intercept, slope = parameters['resnet18']['naive']['gpu'][0]['intercept'], parameters['resnet18']['naive']['gpu'][0]['slope']
	elif model_class == "Inception":
		model_class = Inception3
		if mode == "approximate":
			intercept, slope = parameters['inception']['approx']['gpu'][0]['intercept'], parameters['inception']['approx']['gpu'][0]['slope']
		elif mode == "exact":
			intercept, slope = parameters['inception']['exact']['gpu'][0]['intercept'], parameters['inception']['exact']['gpu'][0]['slope']
		else:
			intercept, slope = parameters['inception']['naive']['gpu'][0]['intercept'], parameters['inception']['naive']['gpu'][0]['slope']

	patch_size = (int)(float(message['patchSize']))
	stride_size = (int)(float(message['strideSize']))

	calibrated_h = 224
	calibrated_w = 224

	if message['x1'] == "":
		x1 = 0
		x2 = width
		y1 = 0
		y2 = height
		h = height
		w = width
		completeImage = True

	else:
		x1 = (int)(float(message['x1']))
		x2 = (int)(float(message['x2']))
		y1 = (int)(float(message['y1']))
		y2 = (int)(float(message['y2']))

		h = (int)(float(message['h']))
		w = (int)(float(message['w']))

		if model_class != Inception3:
			calibrated_x1 = (int)(x1 * 224 / width)
			calibrated_y1 = (int)(y1 * 224 / height)

			calibrated_w = (int)(w * 224 / width)
			calibrated_h = (int)(h * 224 / height)
			image_size = 224
		else:
			calibrated_x1 = (int)(x1 * 299 / width)
			calibrated_y1 = (int)(y1 * 299 / height)

			calibrated_w = (int)(w * 299 / width)
			calibrated_h = (int)(h * 299 / height)
			image_size = 299


	real_estimate = time_estimate(slope, intercept, stride_size, patch_size, calibrated_w, calibrated_h)
	real_estimate = round(real_estimate, 2)
	return JsonResponse({'time_estimate':real_estimate})

def time_estimate(slope, intercept, stride, patch, width, height):
	return slope * (width - patch) * (height - patch) / stride / stride + intercept
