from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage

from .forms import *
# from .models import Photo

import json
import sys
import time
import os
import cv2
from PIL import Image

#sys.path.append('/krypton/code-release/ui/krypton_ui/krypton')
sys.path.append('/krypton/code-release/core/python')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from commons import inc_inference, inc_inference_with_model, show_heatmap
from commons import full_inference_e2e 

from imagenet_classes import class_names
from vgg16 import VGG16
from resnet18 import ResNet18
from inception3 import Inception3

def index(request):
	return render(request, 'index.html')

def selectedRegion(request):
	message = request.POST
	print("SelectedRegion request detected!")

	# input_image = request.FILES
	# for line in input_image:
	# 	save_path = os.path.join(settings.MEDIA_ROOT, 'uploads', line)
	# 	print save_path
	# 	path = default_storage.save(save_path, line)
	# 	print path

	form = PhotoForm(request.POST,request.FILES)
	if form.is_valid():
		photo = form.save()
		data = {'is_valid': True}

	else:
		data = {'is_valid': False}
		print ("photo is not valid" );

	#print("start loading")
	file_path = '/krypton/code-release/ui/krypton_ui/krypton/updated-time-estimation.txt'

	with open(file_path) as f:
		print("loading")
		parameters = json.load(f)

	model_class = message['model']
	
	mode = message['mode']
	#exact, approximate, naive

	#do infernece with model with save much more time especially in the first try
	if model_class == "VGG":
		#model = VGG16(beta=1.0, gpu=True, n_labels=1000).eval()
		model_class = VGG16
		intercept, slope = parameters['vgg16'][0]['intercept'], parameters['vgg16'][0]['slope']
	elif model_class == 'ResNet':
		#model = ResNet18(beta=1.0, gpu=True, n_labels=1000).eval()
		model_class = ResNet18
		intercept, slope = parameters['resnet18'][0]['intercept'], parameters['resnet18'][0]['slope']
	elif model_class == 'Inception':
		#model = Inception3(beta=1.0, gpu=True, n_labels=1000).eval()
		model_class = Inception3
		intercept, slope = parameters['inception'][0]['intercept'], parameters['inception'][0]['slope']

	patch_size = (int)(float(message['patchSize']))
	stride_size = (int)(float(message['strideSize']))

	x1 = (int)(float(message['x1']))
	x2 = (int)(float(message['x2']))
	y1 = (int)(float(message['y1']))
	y2 = (int)(float(message['y2']))

	h = (int)(float(message['h']))
	w = (int)(float(message['w']))

	#naive mode only allows the entire
	if mode == "naive":
		w,h = 224, 224		
	
	estimated_time = time_estimate(slope, intercept, stride_size, patch_size, w, h)
	coeff = 1.0
	
	if model_class == VGG16:
		if stride_size == 16:
			coeff = 1.0
		elif stride_size == 8:
			coeff = 1.0
		elif stride_size == 4:
			coeff = 2.0
		elif stride_size == 2:
			coeff = 3.0
		else:
			coeff = 3.0
	elif model_class == ResNet18:
		if stride_size == 16:
			coeff = 1.0
		elif stride_size == 8:
			coeff = 2.0
		elif stride_size == 4:
			coeff = 3.0
		elif stride_size == 2:
			coeff = 4.0
		else:
			coeff = 5.0

	real_estimate = estimated_time / coeff

	'''
	we only display the estimated tiem for krypton exact
	when running krypton exact, we are verifying the accuracy of prediction
	when running naive, we can see how krypton improves from naive cnn
	when running approximation, we can see how approximation improves time cost
	'''

	print ('krypton exact estimated_time ' + str(real_estimate))

	prev_path = '/krypton/code-release/ui/krypton_ui'
	curr_path = prev_path + photo.file.url

	im = Image.open(curr_path)
	width, height = im.size

	calibrated_x1 = (int)(x1 * 224 / width)
	calibrated_y1 = (int)(y1 * 224 / height)

	calibrated_w = (int)(w * 224 / width)
	calibrated_h = (int)(h * 224 / height)

	start_time = time.time()
	
	if mode == "exact":
		heatmap, prob, label = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x0=calibrated_x1, y0=calibrated_y1, x_size=calibrated_w, y_size=calibrated_h, gpu=True)
	elif mode == "approximate":
		heatmap, prob, label = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=0.5, x0=0, y0=0, x_size=0, y_size=0, gpu=True)
	else:
		heatmap, prob, label = full_inference_e2e(model_class, curr_path, patch_size=patch_size, stride=stride_size, batch_size=64, gpu=True)

	end_time = time.time()

	print ("actual time " + str(end_time - start_time))
	print ("inference done")

	plt.imshow(heatmap)
	plt.savefig("./media/photos/heatmap.png")

	response = HttpResponse(content_type="image/png")
	img = Image.open('./media/photos/heatmap.png')
	img = img.resize((width,height), Image.ANTIALIAS)
	img.save('./media/photos/heatmap.png')
	img = Image.open('./media/photos/heatmap.png')
	print(img.size)
	img.save(response,'png')
	print label
	return response


def time_estimate(slope, intercept, stride, patch, width, height):
	return slope * (width - patch) * (height - patch) / stride / stride + intercept
