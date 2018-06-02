#!/usr/bin/env python
# coding=utf-8

import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms
import torch.nn.functional as F

from inception3 import Inception3
from commons import inc_convolution_bn, inc_max_pool, inc_add
from imagenet_classes import class_names


class IncrementalInception3V2(nn.Module):

    def __init__(self, in_tensor, beta=1.0, gpu=True):
        super(IncrementalInception3V2, self).__init__()

        # performing initial full inference
        full_model = Inception3()
        full_model.eval()
        self.gpu = gpu
        if self.cuda:
            in_tensor = in_tensor.cuda()

        self.initial_result = full_model.forward_materialized(in_tensor).cpu().data.numpy()
        self.full_model = full_model
        self.beta = beta

        torch.cuda.empty_cache()

    def forward(self, x, locations, p_height, p_width):        
        return self.full_model.forward_inc_v2(x, locations, p_height, p_width, self.beta)


if __name__ == "__main__":
    batch_size = 1
    patch_size = 16
    input_size = 299

    image_patch = torch.cuda.FloatTensor(3, patch_size, patch_size).fill_(0)

    x_loc = random.sample(range(0, input_size - patch_size), batch_size)
    y_loc = random.sample(range(0, input_size - patch_size), batch_size)
    patch_locations = zip(x_loc, y_loc)
    patch_locations = [((299-patch_size)//2, (299-patch_size)//2)]

    loader = transforms.Compose([transforms.Resize([299, 299]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')
    images = loader(images)

    images = images.unsqueeze(0)
    images = images.repeat(batch_size, 1, 1, 1)

    for i,(x,y) in enumerate(patch_locations):
        images[i, :, x:x+patch_size, y:y+patch_size] = image_patch

    full_model = Inception3()
    full_model.eval()
    y = full_model(images.cuda())

    patch_locations = torch.from_numpy(np.array(patch_locations, dtype=np.int32))

    inc_model = IncrementalInception3V2(images, beta=1.0)

    inc_model.eval()
    x = inc_model(images, patch_locations, patch_size, patch_size)
    # print(class_names[np.argmax(x.data.cpu().numpy()[0, :])])

    temp = (y - x).cpu().data.numpy()
    import matplotlib.pyplot as plt
    plt.imshow(temp[0,0,:,:])
    plt.colorbar()
    plt.show()
    print(np.max(np.abs(temp)))