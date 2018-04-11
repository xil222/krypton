#ifndef _CONV_CUDA
#define _CONV_CUDA

#ifdef __cplusplus
extern "C" {
#endif

float* cuda_make_array(float* ptr, size_t size);
void cuda_free_array(float* ptr);
void im2col_gpu(float *im, int channels, int height, int width, int ksize, int stride, int pad, float *data_col);
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float *A_gpu, int lda, float *B_gpu, int ldb, float BETA, float *C_gpu, int ldc);

#ifdef __cplusplus
}
#endif

#endif