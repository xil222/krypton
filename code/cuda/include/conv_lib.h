int inc_convolution(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases,
 THCudaTensor * out_tensor, THCudaIntTensor * location_tensor, int padding_x, int padding_y,
  int stride_x, int stride_y, int p_height, int p_width, float beta);

int batch_normalization(THCudaTensor * in_tensor,  THCudaTensor * bn_mean, THCudaTensor * bn_var, THCudaTensor * bn_weights, THCudaTensor * bn_biases, float eps);

void inc_add(THCudaTensor * in_tensor1, THCudaTensor * location_tensor1, THCudaTensor * premat_tensor2, THCudaTensor * in_tensor2, THCudaTensor * location_tensor2);

int inc_max_pool(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,  THCudaIntTensor * location_tensor,
 int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y, int p_height, int p_width, float beta);

int full_projection(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,
 THCudaIntTensor * location_tensor, int p_height, int p_width);