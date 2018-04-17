#!/usr/bin/env python

import time
import torch
from torch.autograd import Variable


in_tensor = Variable(torch.FloatTensor(128, 64,  7, 7).fill_(1.0).cuda())
m = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1).cuda()
m(in_tensor)
time.sleep(4)

in_tensor = Variable(torch.FloatTensor(128, 64,  14, 14).fill_(1.0).cuda())
m = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1).cuda()
m(in_tensor)
time.sleep(4)

in_tensor = Variable(torch.FloatTensor(128, 64,  28, 28).fill_(1.0).cuda())
m = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1).cuda()
m(in_tensor)
time.sleep(4)

in_tensor = Variable(torch.FloatTensor(128, 64,  56, 56).fill_(1.0).cuda())
m = torch.nn.Conv2d(64, 64, 3, padding=1, stride=1).cuda()
m(in_tensor)
time.sleep(4)