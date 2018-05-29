int inc_conv_relu(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor, int padding, int stride, int p_height, int p_width, float patch_growth_threshold);

int inc_conv_bn(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * bn_mean, THCudaTensor * bn_var, THCudaTensor * bn_weights,THCudaTensor * bn_biases, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor, int padding_x, int padding_y, int stride, int p_height, int p_width, float beta, int relu, float eps);

int inc_max_pool(THCudaTensor * in_tensor, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor, int padding, int stride, int k_size, int p_height, int p_width, float beta);

int inc_avg_pool(THCudaTensor * in_tensor, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor,
 int padding, int stride, int k_size, int p_height, int p_width,
 float beta);

int inc_add(THCudaTensor * in_tensor1, THCudaTensor * in_tensor2, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor, int p_height, int p_width, int relu);

int update_output_locations(THCudaIntTensor * patch_location_tensor, int padding_y, int padding_x, int stride, int k_size_y, int k_size_x, int p_height, int p_width, int in_size, int out_size, float beta);

int inc_conv_relu2(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor, int out_size, int padding, int stride, int p_height, int p_width, float beta);

int inc_max_pool2(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,  THCudaIntTensor * patch_location_tensor, int out_size, int padding, int stride, int k_size, int p_height, int p_width, float beta);