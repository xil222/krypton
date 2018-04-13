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


class IncConvFunctionV2(Function):

    def __init__(self, padding, stride):
        self.padding = padding
        self.stride = stride

    def forward(self, in_tensor, weights, out_tensor):
        inc_conv_lib.inc_conv_v2(in_tensor, weights, out_tensor, self.padding, self.stride)
        return out_tensor


class IncConvFunctionV3(Function):

    def __init__(self, padding, stride):
        self.padding = padding
        self.stride = stride

    def forward(self, in_tensor, weights, out_tensor):
        inc_conv_lib.inc_conv_v3(in_tensor, weights, out_tensor, self.padding, self.stride)
        return out_tensor


class IncConvModule(Module):

    def forward(self, in_tensor, weights, out_tensor, padding, stride, version=1):
        if version == 1:
                return IncConvFunctionV1(padding, stride)(in_tensor, weights, out_tensor)
        elif version == 2:
                return IncConvFunctionV2(padding, stride)(in_tensor, weights, out_tensor)
        elif version == 3:
                return IncConvFunctionV3(padding, stride)(in_tensor, weights, out_tensor)


for _ in range(3):

    module = IncConvModule()
    in_tensor = torch.FloatTensor(64, 64, 227, 227).fill_(1.0)
    weights = torch.FloatTensor(64, 64, 3, 3).fill_(1.0)
    in_tensor, weights, = in_tensor.cuda(), weights.cuda()

    m = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1).cuda()
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
        module(in_tensor, weights, out_tensor, 1, 1, version=1)
    torch.cuda.synchronize()
    print('inc conv v1: ' + str(time.time() - prev_time))

    #torch.cuda.synchronize()
    #prev_time = time.time()
    #for i in range(5):
    #    module(in_tensor, weights, out_tensor, 1, 1, version=2)
    #torch.cuda.synchronize()
    #print('inc conv v2: ' + str(time.time() - prev_time))

    torch.cuda.synchronize()
    prev_time = time.time()
    for i in range(5):
        module(in_tensor, weights, out_tensor, 1, 1, version=3)
    torch.cuda.synchronize()
    print('inc conv v3: ' + str(time.time() - prev_time))

