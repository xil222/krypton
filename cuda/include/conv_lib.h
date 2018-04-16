int inc_conv_v1(THCudaTensor *input, THCudaTensor *weights, THCudaTensor *biases, THCudaTensor *output, int padding, int stride);
int inc_conv_v3(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor *biases, THCudaTensor * out_tensor, int padding, int stride, int p_row_start, int p_col_start, int p_width, int p_height);

//int inc_conv_v2(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * out_tensor, int padding, int stride);
//int inc_conv_v4(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * out_tensor, int padding, int stride);
