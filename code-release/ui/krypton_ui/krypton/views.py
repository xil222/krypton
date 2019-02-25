from django.http import HttpResponse
from django.http import JsonResponse
import json
from django.shortcuts import render

import sys
import time
# sys.path.append('../core/python')

# from commons import inc_inference, adaptive_drilldown, show_heatmap
# from imagenet_classes import class_names
# from vgg16 import VGG16
# sys.path.append('../')


def index(request):
	# print("Index request detected!")
	return render(request, 'index.html')

# input is a request, output is a image
def selectedRegion(request):
	message = request.POST
	print("SelectedRegion request detected!")

	#print(str(message))
	#print(request.FILES)
	#'django.utils.datastructures.MultiValueDict'
	#print(type(request.FILES))

	input_image = request.FILES

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
		model_class = "VGG16"

	# begin_time = time.time()

	# heatmap, prob, label = inc_inference(model_class, inputImage, patch_size, stride_size, batch_size=128, beta=1.0, x0=x1, y0=y1, image_size=224,
 #                      x_size=w, y_size=h, gpu=True, version='v1', n_labels=1000, weights_data=None, loader=None,
 #                      c=0.0):

	# end_time = time.time()
	# output_image = show_heatmap(inputImage, heatmap, label=class_names[label], prob=prob, width=224)
	# time_elapsed = end_time - begin_time

	# print(type(output_image))
	# print(output_image.shape)
	
	return JsonResponse({'message': message})



	