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

sys.path.append('/Users/allenord/Documents/CSE291/project/krypton/code-release/core/python')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from commons import inc_inference, show_heatmap
from imagenet_classes import class_names
from vgg16 import VGG16


def index(request):
	# print("Index request detected!")
	return render(request, 'index.html')

# input is a request, output is a image
def selectedRegion(request):
	message = request.POST
	print("SelectedRegion request detected!")


	#
	# input_image = request.FILES
	#
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
		image_file_path = '../../ui/krypton_ui/media/photos/animals.jpg'
	else:
		data = {'is_valid': False}

	#return JsonResponse(data)


	patch_size = (int)(float(message['patchSize']))
	stride_size = (int)(float(message['strideSize']))

	x1 = (int)(float(message['x1']))
	x2 = (int)(float(message['x2']))
	y1 = (int)(float(message['y1']))
	y2 = (int)(float(message['y2']))

	h = (int)(float(message['h']))
	w = (int)(float(message['w']))

	model_class = message['model']
	if model_class == "VGG":
		model_class = VGG16



	heatmap, prob, label = inc_inference(model_class, image_file_path, patch_size=patch_size, stride=stride_size, beta=1.0, gpu=False, c=0.0)
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
