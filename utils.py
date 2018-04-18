from torch.autograd import Function
from torch.nn.modules.module import Module

from cuda._ext import inc_conv_lib


class IncConvFunction(Function):

    def __init__(self, padding, stride, p_height, p_width, version):
        self.padding = padding
        self.stride = stride
        self.version = version
        self.p_height = p_height
        self.p_width = p_width

    def forward(self, in_tensor, weights, biases, out_tensor, patch_location_tensor):
        if self.version == 1:
            inc_conv_lib.inc_conv_v1(in_tensor, weights, biases, out_tensor, self.padding, self.stride)
        elif self.version == 2:
            inc_conv_lib.inc_conv_v2(in_tensor, weights, biases, out_tensor, self.padding, self.stride)
        elif self.version == 3:
            inc_conv_lib.inc_conv_v3(in_tensor, weights, biases, out_tensor,
                                     patch_location_tensor, self.padding, self.stride, self.p_height, self.p_width)
        elif self.version == 4:
            inc_conv_lib.inc_conv_v4(in_tensor, weights, biases, out_tensor,
                                     patch_location_tensor, self.padding, self.stride, self.p_height, self.p_width)

        return out_tensor


class IncConvModule(Module):
    def forward(self, in_tensor, weights, biases, out_tensor, patch_location_tensor,
                padding, stride, p_height=0, p_width=0, version=1):
        return IncConvFunction(padding, stride, p_height, p_width, version)(in_tensor, weights, biases, out_tensor, patch_location_tensor)

