import math
import torch
import h5py
from torch.autograd import Function, Variable
from torch.nn.modules.module import Module

import sys
sys.path.append('../')
from cuda._ext import inc_conv_lib


class IncMaxPoolFunction(Function):

    def __init__(self, padding, stride, k_size, p_height, p_width, patch_growth_threshold):
        super(IncMaxPoolFunction, self).__init__()
        self.padding = padding
        self.stride = stride
        self.k_size = k_size
        self.p_height = p_height
        self.p_width = p_width
        self.patch_growth_threshold = patch_growth_threshold

    def forward(self, in_tensor, out_tensor, patch_location_tensor):
        inc_conv_lib.inc_max_pool_v1(in_tensor, out_tensor, patch_location_tensor, self.padding, self.stride,
                                     self.k_size, self.p_height, self.p_width, self.patch_growth_threshold)
        return out_tensor, patch_location_tensor


class IncMaxPoolModule(Module):

    def __init__(self, in_tensor, out_tensor, padding, stride, k_size, patch_growth_threshold):
        super(IncMaxPoolModule, self).__init__()
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor
        self.padding = padding
        self.stride = stride
        self.k_size = k_size
        self.patch_growth_threshold = patch_growth_threshold

    def forward(self, patch_location_tensor, p_height=0, p_width=0):
        # FIXME Logic duplicated in both CUDA and Python
        out_p_height = int(math.ceil((p_height+self.k_size-1) * 1.0 / self.stride))
        out_p_width = int(math.ceil((p_width+self.k_size-1) * 1.0 / self.stride))
        return IncMaxPoolFunction(self.padding, self.stride, self.k_size, p_height, p_width, self.patch_growth_threshold)(self.in_tensor,  self.out_tensor, patch_location_tensor),\
               (out_p_height, out_p_width)


class IncConvFunction(Function):

    def __init__(self, padding, stride, p_height, p_width, patch_growth_threshold):
        super(IncConvFunction, self).__init__()
        self.padding = padding
        self.stride = stride
        self.p_height = p_height
        self.p_width = p_width
        self.patch_growth_threshold = patch_growth_threshold

    def forward(self, in_tensor, weights, biases, out_tensor, patch_location_tensor):
        inc_conv_lib.inc_conv_v4(in_tensor, weights, biases, out_tensor,
                                 patch_location_tensor, self.padding, self.stride, self.p_height, self.p_width,
                                 self.patch_growth_threshold)

        return out_tensor, patch_location_tensor


class IncConvModule(Module):

    def __init__(self, in_tensor, weights, biases, out_tensor, padding, stride, k_size, patch_growth_threshold):
        super(IncConvModule, self).__init__()
        self.in_tensor = in_tensor
        self.weights = Variable(weights, requires_grad=False, volatile=True)
        self.biases = Variable(biases, requires_grad=False, volatile=True)
        self.out_tensor = out_tensor
        self.padding = padding
        self.stride = stride
        self.k_size = k_size
        self.patch_growth_threshold = patch_growth_threshold

    def forward(self, patch_location_tensor, p_height=0, p_width=0):
        # FIXME Logic duplicated in both CUDA and Python
        out_p_height = int(math.ceil((p_height+self.k_size-1) * 1.0 / self.stride))
        out_p_width = int(math.ceil((p_width+self.k_size-1) * 1.0 / self.stride))
        return IncConvFunction(self.padding, self.stride, p_height, p_width, self.patch_growth_threshold)(self.in_tensor, self.weights, self.biases, self.out_tensor,
                                                                            patch_location_tensor), (out_p_height, out_p_width)


def load_dict_from_hdf5(filename, cuda=True):
    with h5py.File(filename, 'r') as h5file:
        return __recursively_load_dict_contents_from_group(h5file, '/', cuda)


def __recursively_load_dict_contents_from_group(h5file, path, cuda=True):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = torch.from_numpy(item.value)
            if cuda:
                ans[key] = ans[key].cuda()

        elif isinstance(item, h5py._hl.group.Group, cuda):
            ans[key] = __recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
