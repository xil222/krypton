#ifndef _CONV_CUDA
#define _CONV_CUDA

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void update_output_locations_gpu(int* ptr_location, int batch, int in_height, int in_width, int out_height, int out_width, int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y , int out_p_height, int out_p_width, int remove_x, int remove_y);
    
void add_bias_gpu(float *ptr_out_tensor, float * ptr_biases, int batch, int p_height, int p_width, int channels);
    
void extract_input_volume_gpu(float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location, int batch, int channels, int in_height, int in_width, int out_height, int out_width, int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y, int p_height, int p_width, int in_p_height, int in_p_width, int out_p_height, int out_p_width, int remove_x, int remove_y);

void inc_max_pool_gpu(float* ptr_premat_tensor, float* ptr_in_tensor, float* ptr_out_tensor, int * ptr_location, int batch, int channels, int in_height, int in_width, int out_height, int out_width, int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y, int p_height, int p_width, int out_p_height, int out_p_width, int remove_x, int remove_y);
    
void full_projection_gpu(float * ptr_premat_tensor, float * ptr_in_tensor, float * ptr_out_tensor, int * ptr_location, int batch, int channels, int in_height, int in_width, int p_height, int p_width);
    
#ifdef __cplusplus
}
#endif

#endif
