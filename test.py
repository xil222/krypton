import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module

from _ext import inc_conv_lib


class IncConvFunction(Function):

    def __init__(self, padding, stride):
        self.padding = padding
        self.stride = stride

    def forward(self, input, weights, output):
        inc_conv_lib.inc_conv(input, weights, output, self.padding, self.stride)
        return output


class IncConvModule(Module):

    def forward(self, input, weights, padding, stride):
        z_in, y_in, x_in, _ = input.shape
        z_w, y_w, x_w, _ = weights.shape
        out_width = (x_in - x_w + 2 * padding) // stride + 1
        output = torch.FloatTensor(z_in, z_w, out_width, out_width)
        output = Variable(output)
        output = output.cuda()

        return IncConvFunction(padding, stride)(input, weights, output)


module = IncConvModule()
input = torch.FloatTensor(1, 3, 227, 227).fill_(1.0)
weights = torch.FloatTensor(64, 3, 3, 3).fill_(1.0)
input, weights = Variable(input), Variable(weights)

input, weights, = input.cuda(), weights.cuda()
print(module(input, weights, 1, 1))
