int inc_convolution(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases,
 THCudaTensor * out_tensor, THCudaIntTensor * location_tensor, int padding_y, int padding_x,
  int stride_y, int stride_x, int p_height, int p_width, float beta);

int batch_normalization(THCudaTensor * in_tensor,  THCudaTensor * bn_mean, THCudaTensor * bn_var, THCudaTensor * bn_weights, THCudaTensor * bn_biases, float eps);

void inc_add(THCudaTensor * in_tensor1, THCudaTensor * location_tensor1, THCudaTensor * premat_tensor2, THCudaTensor * in_tensor2, THCudaTensor * location_tensor2);

void inc_stack(THCudaTensor * out_tensor, int out_channels, int start_channel, THCudaIntTensor * out_location, THCudaTensor * in_tensor, THCudaIntTensor * in_location, THCudaTensor * premat_tensor);

int inc_max_pool(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,  THCudaIntTensor * location_tensor,
 int padding_y, int padding_x, int stride_y, int stride_x, int k_size_y, int k_size_x, int p_height, int p_width, float beta);

int inc_avg_pool(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,  THCudaIntTensor * location_tensor,
 int padding_y, int padding_x, int stride_y, int stride_x, int k_size_y, int k_size_x, int p_height, int p_width, float beta);

void full_projection(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,
 THCudaIntTensor * location_tensor, int p_height, int p_width);

void calc_bbox_coordinates(int batch_size, THCudaIntTensor * loc_out_tensor, THCudaIntTensor * loc_tensor1, THCudaIntTensor * loc_tensor2);