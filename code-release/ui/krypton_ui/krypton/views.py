from django.http import HttpResponse
from django.http import JsonResponse
import json
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')


def selectedRegion(request):
    print request.FILES
    message = request.POST
    return JsonResponse({'message': message})
