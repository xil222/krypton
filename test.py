#!/usr/bin/env python
import time
import torch
import numpy as np
from torch.autograd import Variable
from utils import IncConvModule

batch_size = 128
in_channels = 64
in_size = 224

out_channels = 64
out_size = 224

k_size = 3

p_size = 14

for iteration in range(4):

    module = IncConvModule()
    in_tensor = torch.FloatTensor(batch_size, in_channels, in_size, in_size).uniform_(0, 10)
    weights = torch.FloatTensor(out_channels, in_channels, k_size, k_size).uniform_(0, 10)
    biases = torch.FloatTensor(out_channels).uniform_(0, 1)

    in_tensor, weights, biases = in_tensor.cuda(), weights.cuda(), biases.cuda()

    m = torch.nn.Conv2d(in_channels, out_channels, k_size, padding=1, stride=1).cuda()
    m.weight.data = weights
    m.bias.data = biases

    in_tensor, weights, biases = Variable(in_tensor), Variable(weights), Variable(biases)

    torch.cuda.synchronize()
    prev_time = time.time()
    for i in range(5):
        out_tensor = m(in_tensor)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    print('Pytorch: ' + str(time.time() - prev_time))

    patch_location_tensor = Variable(
        torch.from_numpy(
            np.array([(in_size - p_size) // 2, (in_size - p_size) // 2] * batch_size,
                     dtype=np.int32)).cuda())

    for v in [4]:
        torch.cuda.synchronize()
        prev_time = time.time()
        for i in range(5):
            # have to move data at each iteration
            patch_location_tensor = Variable(
                torch.from_numpy(
                    np.array([(in_size - p_size) // 2, (in_size - p_size) // 2] * batch_size,
                             dtype=np.int32)).cuda())
            (out_tensor, _), _ = module(in_tensor, weights, biases, out_tensor, patch_location_tensor, 1, 1, k_size,
                                         p_height=p_size, p_width=p_size, version=v)
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        print('inc conv v' + str(v) + ': ' + str(time.time() - prev_time))
