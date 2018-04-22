import random

import time
import torch
from vgg16 import VGG16
from vgg16_inc import IncrementalVGG16
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from imagenet_classes import class_names
from torch.autograd import Variable

# testing incremental inference for VGG16
if __name__ == "__main__":
    batch_size = 64
    patch_size = 4
    input_size = 224

    x_loc = random.sample(range(0, input_size - patch_size), batch_size)
    y_loc = random.sample(range(0, input_size - patch_size), batch_size)
    patch_locations = zip(x_loc, y_loc)

    loader = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    images = Image.open('./dog_resized.jpg')
    images = loader(images)
    images = images.repeat(batch_size, 1, 1, 1)

    images_var = Variable(images, requires_grad=False, volatile=True).cuda()

    full_model = VGG16()
    full_model.eval()

    inc_model = IncrementalVGG16(full_model, images_var, patch_growth_threshold=0.25)
    inc_model.eval()

    patch_locations_var = Variable(torch.from_numpy(np.array(patch_locations, dtype=np.int32)), volatile=True).cuda()
    for i, (x, y) in enumerate(patch_locations):
        images_var[i, :, x:x + patch_size, y:y + patch_size] = torch.from_numpy(
            np.zeros(shape=(3, patch_size, patch_size), dtype=np.float32))

    torch.cuda.empty_cache()
    x = full_model(images_var)
    torch.cuda.synchronize()
    prev_time = time.time()
    for i in range(5):
        x = full_model(images_var)
    torch.cuda.synchronize()
    full_time = time.time() - prev_time

    full_softmax = x.data.cpu().numpy()[0, :]
    print("Full Inference: " + class_names[np.argmax(full_softmax)])


    torch.cuda.empty_cache()
    x = inc_model(images_var, patch_locations_var, patch_size, patch_size)
    torch.cuda.synchronize()
    prev_time = time.time()
    for i in range(5):
        x = inc_model(images_var, patch_locations_var, patch_size, patch_size)
    torch.cuda.synchronize()
    inc_time = time.time() - prev_time

    inc_softmax = x.data.cpu().numpy()[0, :]
    print("Incremental Inference: " + class_names[np.argmax(inc_softmax)])

    print("Speedup: " + str(full_time/inc_time), " L2 Norm: " + str(np.linalg.norm(inc_softmax-full_softmax)))
