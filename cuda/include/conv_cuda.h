#ifndef _CONV_CUDA
#define _CONV_CUDA

#ifdef __cplusplus
extern "C" {
#endif

float* cuda_make_array(float* ptr, size_t size);
void cuda_free_array(float* ptr);

void im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col);
void inc_im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col, int p_row_start, int p_col_start, int p_height, int p_width);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda, float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc);

void gemv_conv_gpu(float *ptr_weights, float *workspace, float *ptr_out_tensor, int in_size, int in_channels, int out_size, int out_channels, int k_size);
void inc_conv_mem_copy_gpu(float *c, float *ptr_out_tensor, int p_row_start, int p_col_start, int p_height, int p_width, int channels, int size);

void premat_mem_copy_gpu(int size, int channels, int batch, float *data_out_ptr, float *premat_ptr);

void cudnn_mem_copy_gpu(int batch, int channels, int size, int padding, int stride, float *in_ptr, float* out_ptr, int * ptr_location, int in_p_height, int in_p_width);
void inc_conv_mem_copy_gpu_v2(float *ptr_temp_tensor, float *ptr_out_tensor, float * biases, int * ptr_location, int batch, int p_height, int p_width, int channels, int size);

//void batch_dp_gemm_conv_gpu(int in_channels, int in_size, int k_size, int out_size, int padding, int stride, float * ptr_input, float * ptr_weights, float * ptr_output, int groups, int batch, int m, int k, int n, float * workspace);

//void batched_inc_conv_dp_gpu(int batch, float *workspace, float *c, float * ptr_in_tensor, float *ptr_out_tensor, 
// float *ptr_weights, int p_row_start, int p_col_start, int p_width,
// int p_height, int k_size, int in_size, int in_channels, int out_size, int out_channels, int padding, int stride);

//nhwc
//void img_mem_copy_gpu_nhwc(out_size, out_channels, batch, ptr_out_tensor, ptr_out_tensor);

#ifdef __cplusplus
}
#endif

#endif
