int inc_conv(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases, THCudaTensor * out_tensor,
        THCudaIntTensor * patch_location_tensor, int padding, int stride, int p_height, int p_width,
        float patch_growth_threshold);

int inc_max_pool(THCudaTensor * in_tensor, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor,
        int padding, int stride, int k_size, int p_height, int p_width, float beta);