from __future__ import division
import math
import numpy as np
import torch
from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.densenet import densenet121
from torchvision.models.inception import Inception3
import sys
import random
import scipy.stats

sys.path.append('../../../code/python')

from models.mobilenet import mobilenet
from models.squeezenet import squeezenet1_0

from torch.autograd import Variable

import matplotlib.pyplot as plt

random.seed(45)
np.random.seed(45)

torch.set_num_threads(8)


def get_ivm_patch_coordinates(in_p_width, in_p_height, patch_x0, patch_y0, padding_x, padding_y,
                              filter_width, filter_height, stride_x, stride_y,
                              input_width, input_height, output_width, output_height, tau):

    out_width = min(math.ceil((filter_width + in_p_width - 1.0)/stride_x), output_width)
    out_height = min(math.ceil((filter_height + in_p_height - 1.0)/stride_y), output_height)
    
    if (out_width > round(tau*output_width)) or (out_height > round(tau*output_height)):
        temp_out_height = round(min(tau*output_height, out_height))
        temp_out_width = round(min(tau*output_width, out_width))
        
        new_in_p_height = temp_out_height * stride_y - filter_height + 1
        new_in_p_width = temp_out_width * stride_x - filter_width + 1
        
        remove_x = in_p_width - new_in_p_width
        remove_y = in_p_height - new_in_p_height
        
        in_p_width = new_in_p_width
        in_p_height = new_in_p_height
        
        patch_x0 += remove_x/2
        patch_y0 += remove_y/2

        out_width = temp_out_width
        out_height = temp_out_height
            
    out_x0 = max(math.ceil((patch_x0 + padding_x - filter_width + 1.0)/stride_x), 0)
    out_y0 = max(math.ceil((patch_y0 + padding_y - filter_height + 1.0)/stride_y), 0)
    
    in_x0 = out_x0 * stride_x - padding_x
    in_y0 = out_y0 * stride_y - padding_y
    
    in_r_width = filter_width + (out_width-1)*stride_x
    in_r_height = filter_height + (out_height-1)*stride_y   
    
    return int(in_x0), int(in_x0+in_r_width), int(in_y0), int(in_y0+in_r_height), int(out_x0), int(out_x0+out_width), int(out_y0), int(out_y0+out_height)


def calculate_ivm_flops(model, x, patch_size_x, patch_size_y, patch_x0, patch_y0, tau=1.0, debug=True, graph=None):
    if graph is None:
        trace, _ = torch.jit.trace(model, args=(x))
        torch.onnx._optimize_trace(trace)
        graph = trace.graph()

    nodes = list(graph.nodes())
    inputs = list(graph.inputs())

    coordinates = {inputs[0].uniqueName(): (patch_x0, patch_x0 + patch_size_x, patch_y0, patch_y0 + patch_size_y)}

    ops_count = 0
    counter = 0
    for node in nodes:
        if node.kind() in ['onnx::Conv', 'onnx::MaxPool', 'onnx::AveragePool']:
            kernel_height, kernel_width = node['kernel_shape']
            stride_x, stride_y = node['strides']

            if node.kind() == 'onnx::Conv':
                padding_x, _, padding_y, _ = node['pads']
                group = node['group']
            else:
                group = 1
                padding_x, _, padding_y,_ = node['pads']

            inputs = list(node.inputs())
            outputs = list(node.outputs())
            
            patch_x0, patch_x1, patch_y0, patch_y1 = coordinates[inputs[0].uniqueName()]

            _, in_channels, in_height, in_width = inputs[0].type().sizes()
            output_width = ((in_width - kernel_width + 2 * padding_x) // stride_x + 1)
            output_height = ((in_height - kernel_height + 2 * padding_y) // stride_y + 1)

            if node.kind() == 'onnx::Conv':
                out_channels, _, _, _ = inputs[1].type().sizes()

                _, _, _, _, patch_x0, patch_x1, patch_y0, patch_y1 = get_ivm_patch_coordinates(patch_x1 - patch_x0,
                                                                                               patch_y1 - patch_y0,
                                                                                               patch_x0,
                                                                                               patch_y0,
                                                                                               padding_x,
                                                                                               padding_y,
                                                                                               kernel_width,
                                                                                               kernel_height,
                                                                                               stride_x,
                                                                                               stride_y,
                                                                                               in_width,
                                                                                               in_height,
                                                                                               output_width,
                                                                                               output_height,
                                                                                               tau)

                temp = (patch_x1 - patch_x0) * (
                patch_y1 - patch_y0) * out_channels * (kernel_width ** 2) * in_channels / group

                if debug:
                    print (counter, "FLOPS:", node.kind(), temp / (1000 ** 2), (int(output_width), int(output_width)),
                           (patch_x0, patch_x1, patch_y0, patch_y1, int(out_channels)), ((int(kernel_width),
                                                                                                  int(kernel_width)),
                                                                                                 int(in_channels/group)))
                    counter += 1

                ops_count += temp
            else:

                _, _, _, _, patch_x0, patch_x1, patch_y0, patch_y1 = get_ivm_patch_coordinates(patch_x1 - patch_x0,
                                                                                               patch_y1 - patch_y0,
                                                                                               patch_x0,
                                                                                               patch_y0,
                                                                                               padding_x,
                                                                                               padding_y,
                                                                                               kernel_width,
                                                                                               kernel_height,
                                                                                               stride_x,
                                                                                               stride_y,
                                                                                               in_width,
                                                                                               in_height,
                                                                                               output_width,
                                                                                               output_height,
                                                                                               tau)

            coordinates[outputs[0].uniqueName()] = (patch_x0, patch_x1, patch_y0, patch_y1)
            

        elif node.kind() in ['onnx::Gemm']:
            inputs = list(node.inputs())
            outputs = list(node.outputs())
            
            _, in_width = inputs[0].type().sizes()
            out_width, _ = inputs[1].type().sizes()

            temp = in_width * out_width

            if debug:
                print ("FLOPS:", node.kind(), temp / (1000 ** 2))

            ops_count += temp
            coordinates[outputs[0].uniqueName()] = (-1, -1, -1, -1)

        else: # A conv operator comes with a proceeding Select
            # shape preserving operations
            inputs = list(node.inputs())
            outputs = list(node.outputs())
            
            if node.kind() in ['onnx::Concat']:  # DenseNet
                coordinates[outputs[0].uniqueName()] = coordinates[
                    inputs[-1].uniqueName()]  # last input is the most changed one
            elif node.kind() in ['onnx::Relu', 'onnx::BatchNormalization',
                                 'onnx::Dropout', 'onnx::Flatten', 'onnx::Add']:
                coordinates[outputs[0].uniqueName()] = coordinates[inputs[0].uniqueName()]
            else:
                #print(node.kind())
                pass

    return ops_count


def calculate_flops(model, x, debug=True, graph=None):

    if graph is None:
        trace, _ = torch.jit.trace(model, args=(x,))
        torch.onnx._optimize_trace(trace)
        graph = trace.graph()

    nodes = list(graph.nodes())

    ops_count = 0
    counter = 0
    for node in nodes:
        
        if node.kind() in ['onnx::Conv', 'onnx::MaxPool', 'onnx::AveragePool']:
            kernel_width, kernel_height = node['kernel_shape']
            stride_x, stride_y = node['strides']

            if node.kind() == 'onnx::Conv':
                padding_x, _, padding_y, _ = node['pads']
                group = node['group']
            else:
                padding_x, _, padding_y, _ = node['pads']

            inputs = list(node.inputs())

            if node.kind() == 'onnx::Conv':
                _, in_channels, in_x, in_y = inputs[0].type().sizes()
                out_channels, _, _, _ = inputs[1].type().sizes()

                output_width = ((in_x - kernel_width + 2 * padding_x) // stride_x + 1)
                output_height = ((in_y - kernel_height + 2 * padding_y) // stride_y + 1)
                temp = (output_width * output_height) * out_channels * (kernel_width * kernel_height) * in_channels / group

                if debug:
                    print (counter, "FLOPS:", node.kind(), out_channels, temp / (1000 ** 2))
                    counter += 1

                ops_count += temp

        elif node.kind() in ['onnx::Gemm']:
            inputs = list(node.inputs())
            _, in_width = inputs[0].type().sizes()
            out_width, _ = inputs[1].type().sizes()

            temp = in_width * out_width

            if debug:
                print (counter, "FLOPS:", node.kind(), temp / (1000 ** 2))
                counter += 1
                
            ops_count += temp

    return ops_count


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    if math.isnan(h):
        h = 0
    return m, h