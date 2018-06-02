#ifndef _CONV_CUDA
#define _CONV_CUDA

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void update_output_locations_gpu(int* ptr_location, int batch, int size_x, int size_y, int padding_x, int padding_y, int stride_x,
    int stride_y, int k_size_x, int k_size_y , int in_p_height, int in_p_width, bool patch_growing);

void add_bias_gpu(float *ptr_out_tensor, float * ptr_biases, int batch, int p_height, int p_width, int channels);

void extract_input_volume_gpu(float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location,
 int batch, int channels, int in_height, int in_width, int padding_x, int padding_y, int stride_x, int stride_y,
 int p_height, int p_width, int in_p_height, int in_p_width, int k_size_x, int k_size_y, int patch_growing);

#ifdef __cplusplus
}
#endif

#endif
