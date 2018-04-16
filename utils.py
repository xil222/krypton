from torch.autograd import Function
from torch.nn.modules.module import Module

from cuda._ext import inc_conv_lib

class IncConvFunction(Function):

    def __init__(self, padding, stride, version, p_row_start=0, p_col_start=0, p_height=0, p_width=0):
        self.padding = padding
        self.stride = stride
        self.version = version
        self.p_row_start = p_row_start;
        self.p_col_start = p_col_start;
        self.p_height = p_height;
        self.p_width = p_width;

    def forward(self, in_tensor, weights, biases, out_tensor):
        if self.version == 1:
            inc_conv_lib.inc_conv_v1(in_tensor, weights, biases, out_tensor, self.padding, self.stride)
        elif self.version == 3:
            inc_conv_lib.inc_conv_v3(in_tensor, weights, biases, out_tensor, self.padding, self.stride, self.p_row_start, self.p_col_start, self.p_height, self.p_width)

        return out_tensor


class IncConvModule(Module):
    def forward(self, in_tensor, weights, biases, out_tensor, padding, stride, version=1, p_row_start=0, p_col_start=0, p_height=0, p_width=0):
        return IncConvFunction(padding, stride, version, p_row_start, p_col_start, p_height, p_width)(in_tensor, weights, biases, out_tensor)

