#include <THC/THC.h>
#include <stdio.h>
#include "conv_cuda.h"

extern THCState *state;

//Basic MCMK Convolution Implementation using im2col + GEMM
int inc_conv_v1(THCudaTensor *in_tensor, THCudaTensor *weights, THCudaTensor *out_tensor, int padding, int stride)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights  = THCudaTensor_data(NULL, weights);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);

    int i,j;

    int groups = 1;
    int batch = in_tensor->size[0];

    int m = out_tensor->size[1]/groups;
    int k = weights->size[3] * weights->size[2] * weights->size[1]/groups;
    int n = out_tensor->size[3] * out_tensor->size[2];

    float * workspace = NULL;
    workspace  = cuda_make_array(workspace, ((size_t)n)*((size_t)k)*sizeof(float));

    for(i = 0; i < batch; i++){
        for(j = 0; j < groups; j++){
            im2col_gpu(ptr_in_tensor + i * in_tensor->size[1] * in_tensor->size[2] * in_tensor->size[3], in_tensor->size[1], in_tensor->size[2],
                    in_tensor->size[3], weights->size[3], stride, padding, workspace);
            float * a = ptr_weights + j*weights->size[0]/groups;
            float * b = workspace;
            float * c = ptr_out_tensor;
            gemm_gpu(0, 0, m,n, k, 1., a, k, b, n, 0., c+i*m*n, n);
        }
    }

    cuda_free_array(workspace);

    return 1;
}


//Change Aware MCMK Convolution implementation(im2col + GEMM)
int inc_conv_v3(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * out_tensor, int padding, int stride)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights  = THCudaTensor_data(NULL, weights);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);

    int i;

    int batch = in_tensor->size[0];
    int in_channels = in_tensor->size[1];
    int in_size = in_tensor->size[2];
    int out_channels = out_tensor->size[1];
    int out_size = out_tensor->size[2];
    int k_size = weights->size[2];

    int p_row_start = 10;
    int p_col_start = 10;
    int p_width = 56;
    int p_height = 56;

    float * workspace = NULL;
    workspace  = cuda_make_array(workspace, ((size_t)p_width*p_height)*((size_t)k_size*k_size*in_channels)*sizeof(float));

    float * c = NULL;
    c = cuda_make_array(c, ((size_t)p_width*p_height)*((size_t)out_channels)*sizeof(float));

    img_mem_copy_gpu(out_size, out_channels, batch, ptr_out_tensor, ptr_out_tensor);

    for(i = 0; i < batch; i++){
        inc_im2col_gpu(ptr_in_tensor + i * in_channels * in_size * in_size, in_channels, in_size,
                in_size, k_size, stride, padding, workspace, p_row_start, p_col_start, p_height, p_width);

        float * a = ptr_weights;
        float * b = workspace;

        int m = out_channels;
        int k = k_size * k_size * in_channels;
        int n = p_width * p_height;

        gemm_gpu(0, 0, m, n, k, 1., a, k, b, n, 0., c, n);

        inc_conv_mem_copy_gpu(c, ptr_out_tensor, p_row_start, p_col_start, p_height, p_width, out_channels, out_size);
    }

    cuda_free_array(c);
    cuda_free_array(workspace);
    return 1;
}


////Trying Dynamic Parallelism for Batch Convolution(dp + im2col + GEMM)
//int inc_conv_v2(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * out_tensor, int padding, int stride)
//{
//    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
//    float * ptr_weights  = THCudaTensor_data(NULL, weights);
//    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);
//
//    int groups = 1;
//    int batch = in_tensor->size[0];
//
//    int m = out_tensor->size[1]/groups;
//    int k = weights->size[3] * weights->size[2] * weights->size[1]/groups;
//    int n = out_tensor->size[3] * out_tensor->size[2];
//
//    float * workspace = NULL;
//
//    workspace  = cuda_make_array(workspace, ((size_t)batch)*((size_t)n)*((size_t)k)*sizeof(float));
//    batch_dp_gemm_conv_gpu(in_tensor->size[1], in_tensor->size[2], weights->size[2], weights->size[0], padding,
//     stride, ptr_in_tensor, ptr_weights, ptr_out_tensor, groups, batch, m, k, n, workspace);
//
//    cuda_free_array(workspace);
//    return 1;
//}


////Change Aware MCMK Convolution with Dynamic Parallelism(DP + im2col + GEMM)
//int inc_conv_v4(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * out_tensor, int padding, int stride)
//{
//    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
//    float * ptr_weights  = THCudaTensor_data(NULL, weights);
//    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);
//
//    int batch = in_tensor->size[0];
//    int in_channels = in_tensor->size[1];
//    int in_size = in_tensor->size[2];
//    int out_channels = out_tensor->size[1];
//    int out_size = out_tensor->size[2];
//    int k_size = weights->size[2];
//
//    int p_row_start = 100;
//    int p_col_start = 100;
//    int p_width = 32;
//    int p_height = 32;
//
//    float * workspace = NULL;
//    workspace  = cuda_make_array(workspace, ((size_t) batch * (p_width*p_height) * (k_size*k_size*in_channels))*sizeof(float));
//
//    float * c = NULL;
//    c = cuda_make_array(c, (size_t)batch * p_width*p_height * out_channels*sizeof(float));
//
//    batched_inc_conv_dp_gpu(batch, workspace, c, ptr_in_tensor, ptr_out_tensor,
//        ptr_weights, p_row_start, p_col_start, p_width,
//        p_height, k_size, in_size, in_channels, out_size, out_channels, padding, stride);
//
//    cuda_free_array(c);
//    cuda_free_array(workspace);
//    return 1;
//}
