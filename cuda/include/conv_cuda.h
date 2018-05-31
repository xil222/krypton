#ifndef _CONV_CUDA
#define _CONV_CUDA

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void update_output_locations_gpu(int batch, int* ptr_location, int size, int padding_x, int padding_y, int stride, int k_size_x, int k_size_y, int in_p_height, int in_p_width, bool patch_growing);

void cudnn_mem_copy_gpu(int batch, int channels, int size, int padding_x, int padding_y, int stride, float *in_ptr, float* out_ptr, int * ptr_location, int in_p_height, int in_p_width);

void inc_max_pool_gpu(float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int out_size, int channels, int batch, int padding, int stride, int k_size, int * ptr_location, int out_p_height, int out_p_width);
    
void inc_avg_pool_gpu(float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int out_size, int channels,
    int batch, int padding, int stride, int k_size, int * ptr_location, int out_p_height, int out_p_width);
    
void inc_add_gpu(float * ptr_in_tensor1, float * ptr_in_tensor2, float * ptr_out_tensor, int * ptr_location_tensor, int channels, int batch, int size, int p_height, int p_width, bool relu);
    
void relu_fused_mem_copy_gpu(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_biases, int * ptr_location, int batch, int p_height, int p_width, int channels, int size);

void bn_fused_mem_copy_gpu(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_bn_mean, float * ptr_bn_var, float * ptr_bn_weights, float * ptr_bn_biases, int * ptr_location, int batch, int p_height, int p_width, int channels, int size, bool relu, float eps);

    
    
    
//experimental
void cudnn_mem_copy_gpu2(int batch, int channels, int size, int padding_x, int padding_y, int stride, float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location, int p_size, int in_p_height, int in_p_width, int k_size_x, int k_size_y);
       
void relu_add_bias_gpu(float *ptr_out_tensor, float * ptr_biases, int batch, int p_height, int p_width, int channels);

void inc_max_pool_gpu2(float* ptr_premat_tensor, float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int out_size, int in_p_size, int channels, int batch, int padding, int stride, int k_size, int * ptr_location, int out_p_height, int out_p_width);  
    
void final_full_projection_gpu(float * ptr_premat_tensor, float * ptr_in_tensor, float * ptr_out_tensor, int * ptr_location, int batch, int channels, int in_size, int p_height, int p_width);    
    
#ifdef __cplusplus
}
#endif

#endif
