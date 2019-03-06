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

from commons import inc_inference, show_heatmap
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
		print photo.file.url
		# data = {'is_valid': True, 'name': photo.file.name, 'url': photo.file.url}
		data = {'is_valid': True}
		#image_file_path = '../../ui/krypton_ui/media/photos/animals.jpg'
		
	else:
		data = {'is_valid': False}
		print ("photo is not valid" );

	patch_size = (int)(float(message['patchSize']))
	stride_size = (int)(float(message['strideSize']))

	x1 = (int)(float(message['x1']))
	x2 = (int)(float(message['x2']))
	y1 = (int)(float(message['y1']))
	y2 = (int)(float(message['y2']))

	h = (int)(float(message['h']))
	w = (int)(float(message['w']))

	prev_path = '/krypton/code-release/ui/krypton_ui'
	curr_path = prev_path + photo.file.url
	#print(curr_path)

	print(curr_path)
	im = Image.open(curr_path)

	print("image opened")
	print(im)	

	width, height = im.size

	print('width ' + str(width))
	print('height ' + str(height))

	model_class = message['model']
	if model_class == "VGG":
		model_class = VGG16
	elif model_class == 'ResNet':
		model_class = ResNet18
	elif model_class == 'Inception':
		model_class = Inception3

	print ("ready for model")

	#our_model = model_class(beta=1.0, gpu=True, n_labels=1000).eval()
	#configure a time requirement for executing on a specific GPU
	print("model created");
	
	begin_time1 = time.time()
	_, _, _ = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x0=0, y0=0, x_size=224, y_size=224, gpu=True, c=0.0)
	end_time1 = time.time()
	
	t1 = end_time1 - begin_time1
	s1 = (224 - patch_size) * (224 - patch_size) * 1.0 / stride_size / stride_size
	
	print("part1 calculated")

	begin_time2 = time.time()
	_, _, _ = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x0=0, y0=0, x_size=100, y_size=100, gpu=True, c=0.0)
	end_time2 = time.time()

	t2 = end_time2 - begin_time2
	s2 = (100 - patch_size) * (100 - patch_size) * 1.0 / stride_size / stride_size

	slope = (t2 - t1) / (s2 - s1)
	intercept = t1 - slope * s1
	print ('slope ' + str(slope))
	print ('intercept ' + str(intercept))

	estimated_time = slope * (w - patch_size) * (h - patch_size) / stride_size / stride_size + intercept
	print ('estimated_time ' + str(estimated_time))
	
	'''
	slope, intercept = auto_configure(our_model, curr_path, stride_size, patch_size)
	print ('slope ' + str(slope))
	print ('intercept ' + str(intercept))

	estimate_time = time_estimate(slope, intercept, stride_size, patch_size, w, h)  	
	print ('estimated_time ' + str(estimated_time))
	'''
	calibrated_x1 = (int)(x1 * 224 / width) 
	calibrated_y1 = (int)(x2 * 224 / height)
	#x1 = (int)(float(message['x1']))
	
	calibrated_w = (int)(w * 224 / width)
	calibrated_h = (int)(h * 224 / height)

	heatmap, prob, label = inc_inference(model_class, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x0=calibrated_x1, y0=calibrated_y1, x_size=calibrated_w, y_size=calibrated_h, gpu=True)
	
	'''	
	slope, intercept = auto_configure(our_model, curr_patch, stride_size, patch_size)
	print('slope ' + str(slope))
	print('intercept ' + str(intercept))

	estimate_time = time_estimate(slope, intercept, stride_size, patch_size, w, h)
	print('time estimated ' + str(estimate_time))
  	
	heatmap, prob, label = inc_inference_with_model(our_model, curr_path, patch_size=patch_size, stride=stride_size, beta=1.0, x0=x1, y0=y1, x_size=w, y_size=h, gpu=True, c=0.0)
	'''

	print ("inference done")
	
	plt.imshow(heatmap)
	plt.savefig("./media/photos/heatmap.png")

	response = HttpResponse(content_type="image/png")
	img = Image.open('./media/photos/heatmap.png')
	img.save(response,'png')
	print label
	return response
	# plt.imshow(heatmap)
	# plt.savefig("heatmap.png")
	#
	# response = HttpResponse(mimetype="image/png")
	# img = Image.open("heatmap.png")
	# img.save(response,'png')



'''
estimate time according to linear function, alpha * (x_size - patch) * (y_size - patch) / (stride^2) + b = time
'''
'''

def auto_configure(model, image_file_path, stride, patch):
	print('start configuration')

	begin_time1 = time.time()
	a, b, c = inc_inference_with_model(model, image_file_path, patch_size=patch, stride=stride, beta=1.0, x0=0, y0=0, x_size=224, y_size=224, gpu=True, c=0.0)
	end_time1 = time.time()
	
	print('cc')
	y1 = end_time1 - begin_time1
	x1 = (224 - patch) * (224 - patch) * 1.0 / stride / stride
	print('1 done')

	begin_time2 = time.time()
	_, _, _ = inc_inference_with_model(model, image_file_path, patch_size=patch, stride=stride, beta=1.0, x0=0, y0=0, x_size=100, y_size=100, gpu=True, c=0.0)
	end_time2 = time.time()
	print('2 done')	

	y2 = end_time2 - begin_time2
	x2 = (100 - patch) * (100 - patch) * 1.0 / stride / stride

	alpha = (y2 - y1) / (x2 - x1)
	beta = y1 - alpha * x1
	return alpha, beta
	
 
def time_estimate(slope, intercept, stride, patch, width, height):
	return slope * (width - patch) * (height - patch) / stride / stride + intercept
'''

