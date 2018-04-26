#ifndef _CONV_CUDA
#define _CONV_CUDA

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void update_output_locations_gpu(int batch, int* ptr_location, int size, int padding, int stride, int k_size,
    int in_p_height, int in_p_width, bool patch_growing);

void cudnn_mem_copy_gpu(int batch, int channels, int size, int padding, int stride, float *in_ptr, float* out_ptr, int * ptr_location, int in_p_height, int in_p_width);

void inc_max_pool_gpu(float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int out_size, int channels,
    int batch, int padding, int stride, int k_size, int * ptr_location, int out_p_height, int out_p_width);

void relu_fused_mem_copy_gpu(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_biases, int * ptr_location, int batch, int p_height, int p_width, int channels, int size);


#ifdef __cplusplus
}
#endif

#endif
