from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage

from .forms import *

import json
import sys
import time
import os
import cv2
from PIL import Image, ImageOps

sys.path.append('/krypton/code-release/core/python')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from commons import inc_inference, show_heatmap, full_inference_e2e
from imagenet_classes import class_names
from vgg16 import VGG16
from resnet18 import ResNet18
from inception3 import Inception3
from imagenet_classes import class_names

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


	file_path_time = '/krypton/code-release/ui/krypton_ui/krypton/updated-time-estimation.txt'
	prev_path = '/krypton/code-release/ui/krypton_ui'


	with open(file_path_time) as f:
		parameters = json.load(f)


	url = photo.file.url
	curr_path = prev_path + url

	im = Image.open(curr_path)
	width, height = im.size

	model_class = message['model']
	mode = message['mode']

	#do infernece with model with save much more time especially in the first try
	if model_class == "VGG":
		model_class = VGG16
		intercept, slope = parameters['vgg16'][0]['intercept'], parameters['vgg16'][0]['slope']
	elif model_class == "ResNet":
		model_class = ResNet18
		intercept, slope = parameters['resnet18'][0]['intercept'], parameters['resnet18'][0]['slope']
	elif model_class == "Inception":
		model_class = Inception3
		intercept, slope = parameters['inception'][0]['intercept'], parameters['inception'][0]['slope']

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

			calibrated_w = 299
			calibrated_h = 299
			image_size = 299


	real_estimate = time_estimate(slope, intercept, stride_size, patch_size, calibrated_w, calibrated_h)

	if mode == "naive":
		real_estimate = 16.3488
	elif mode == "approximate":
		real_estimate = 3.6081

	start_time = time.time()

	if completeImage:
		if mode == "exact":
			heatmap, prob, label = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, gpu=True)
		elif mode == "approximate":
			heatmap, prob, label = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=0.5, gpu=True)
		else:
			heatmap, prob, label = full_inference_e2e(model_class, curr_path, patch_size=patch_size, stride=stride_size, batch_size=64, gpu=True)

	else:
		if mode == "exact":
			heatmap, prob, label = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x0=calibrated_y1, y0=calibrated_x1, x_size=calibrated_h, y_size=calibrated_w, gpu=True)
		elif mode == "approximate":
			heatmap, prob, label = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=0.5, x0=calibrated_y1, y0=calibrated_x1, x_size=calibrated_h, y_size=calibrated_w, gpu=True)

	end_time = time.time()



	end_time = time.time()

	plt.imsave("./media/photos/heatmap.png", heatmap, cmap=plt.cm.jet_r)

	img = Image.open('./media/photos/heatmap.png')
 	img = img.resize((w,h), Image.ANTIALIAS)
	img.save('./media/photos/heatmap.png')
	img = Image.open('./media/photos/heatmap.png')
	padding = (x1, y1, (width - x2), (height - y2))
	new_img = ImageOps.expand(img, padding)
	new_img.save('./media/photos/heatmap.png')

	actTime = round(end_time - start_time, 2)

	estTime = round(real_estimate, 2)
	return JsonResponse({'url':url, 'heatmap_url': '/media/photos/heatmap.png', 'prediction': class_names[label], 'estimate_time': estTime, 'actual_time': actTime})


def time_estimate(slope, intercept, stride, patch, width, height):
	return slope * (width - patch) * (height - patch) / stride / stride + intercept
