from django.http import HttpResponse
from django.http import JsonResponse
import json
from django.shortcuts import render

import sys
import time
sys.path.append('../core/python')

from commons import inc_inference, adaptive_drilldown, show_heatmap
from imagenet_classes import class_names
from vgg16 import VGG16

def index(request):
	# print("Index request detected!")
	return render(request, 'index.html')

# input is a request, output is a image
def selectedRegion(request):
	message = request.POST
	#print("SelectedRegion request detected!")
	#print(str(message))
	#print(request.FILES)
	
	input_image = request.FILES.image

	patch_size = (int)message.patchSize
	stride_size = (int)message.strideSize
	model = message.model

	if model == "VGG":
		model = "VGG16"

	x1 = (int)message.x1
	x2 = (int)message.x2
	y1 = (int)message.y1
	y2 = (int)message.y2

	begin_time = time.time()
	
	#image_file_path --> may need to edit to --> inputImage
	heatmap, prob, label = inc_inference(model, inputImage, patch_size=patch_size, stride=stride_size, beta=1.0, gpu=True, c=0.0)
	
	end_time = time.time()
	
	output_image = show_heatmap(inputImage, heatmap, label=class_names[label], prob=prob, width=224)
	time_elapsed = end_time - begin_time

	
	#return JsonResponse({'message': message})
