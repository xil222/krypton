#ifndef _CONV_CUDA
#define _CONV_CUDA

#ifdef __cplusplus
extern "C" {
#endif

float* cuda_make_array(float* ptr, size_t size);
void cuda_free_array(float* ptr);

void im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col);
void inc_im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col, int p_row_start, int p_col_start, int p_height, int p_width);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda, float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc);

void batch_dp_gemm_conv_gpu(int in_channels, int in_size, int k_size, int out_size, int padding, int stride, float * ptr_input, float * ptr_weights, float * ptr_output, int groups, int batch, int m, int k, int n, float * workspace);

void gemv_conv_gpu(float *ptr_weights, float *workspace, float *ptr_out_tensor, int in_size, int in_channels, int out_size, int out_channels, int k_size);
void inc_conv_mem_copy_gpu(float *c, float *ptr_out_tensor, int p_row_start, int p_col_start, int p_height, int p_width, int channels, int size);
#ifdef __cplusplus
}
#endif

#endif
