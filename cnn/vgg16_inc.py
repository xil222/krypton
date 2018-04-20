#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from imagenet_classes import class_names
from commons import load_dict_from_hdf5


class IncrementalVGG16(nn.Module):

    def __init__(self, init_weights=True):
        super(IncrementalVGG16, self).__init__()
