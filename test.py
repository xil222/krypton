import torch
import time
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module

from _ext import inc_conv_lib


class IncConvFunctionV1(Function):

    def __init__(self, padding, stride):
        self.padding = padding
        self.stride = stride

    def forward(self, in_tensor, weights, out_tensor):
        inc_conv_lib.inc_conv_v1(in_tensor, weights, out_tensor, self.padding, self.stride)
        return out_tensor


class IncConvModule(Module):

    def forward(self, in_tensor, weights, out_tensor, padding, stride, version=0):
	if version == 0:
        	return IncConvFunctionV1(padding, stride)(in_tensor, weights, out_tensor)



for _ in range(3):

    module = IncConvModule()
    in_tensor = torch.FloatTensor(256, 3, 227, 227).fill_(1.0)
    weights = torch.FloatTensor(64, 3, 3, 3).fill_(1.0)
    in_tensor, weights, = in_tensor.cuda(), weights.cuda()

    m = torch.nn.Conv2d(3, 64, 3, padding=1, stride=1).cuda()
    m.weight.data = weights
    m.bias.data.fill_(0)

    in_tensor, weights = Variable(in_tensor), Variable(weights)

    torch.cuda.synchronize()
    prev_time = time.time()
    for i in range(5):
       out_tensor = m(in_tensor)
    torch.cuda.synchronize()
    print('Pytorch: ' + str(time.time() - prev_time))

    torch.cuda.synchronize()
    prev_time = time.time()
    for i in range(5):
        module(in_tensor, weights, out_tensor, 1, 1, version=0)
    torch.cuda.synchronize()
    print('inc conv v1: ' + str(time.time() - prev_time))
