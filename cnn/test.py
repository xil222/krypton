import gc
import time

import matplotlib.pyplot as plt
import torch

from commons import full_inference_e2e, inc_inference_e2e
from vgg16_inc import IncrementalVGG16

# testing incremental inference for VGG16
if __name__ == "__main__":
    image_file_path = "./dog_resized.jpg"
    interested_logit_index = 208

    # torch.cuda.synchronize()
    # prev_time = time.time()
    # outputs = full_inference_e2e(image_file_path, 4, 1, interested_logit_index)
    # torch.cuda.synchronize()
    # full_inference_time = time.time() - prev_time
    # print("Full Inference Time: " + str(full_inference_time))
    #
    # plt.imshow(outputs, cmap=plt.cm.rainbow_r, vmin=.75, vmax=.95, interpolation='none')
    # plt.colorbar()
    # plt.savefig('full_inf_heatmap.png')
    #
    # gc.collect()
    # torch.cuda.empty_cache()

    torch.cuda.synchronize()
    prev_time = time.time()
    outputs = inc_inference_e2e(IncrementalVGG16, image_file_path, 4, 1, interested_logit_index, beta=0.33)
    print(outputs[110, 110])
    torch.cuda.synchronize()
    inc_inference_time = time.time() - prev_time
    print("Incremental Inference Time: " + str(inc_inference_time))

    plt.imshow(outputs, cmap=plt.cm.rainbow_r, vmin=.75, vmax=.95, interpolation='none')
    plt.colorbar()
    plt.savefig('inc_inf_heatmap.png')
