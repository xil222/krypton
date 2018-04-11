#include <THC/THC.h>
#include <stdio.h>
#include "conv_cuda.h"

extern THCState *state;

//im2_col + cublas GEMM
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
            gemm_gpu(0, 0, m,n, k, 1., a, k, b, n, 1., c+i*m*n, n);
        }
    }

    cuda_free_array(workspace);

    return 1;
}


//im2_col + cublas GEMM + Dynamic Parallelism
int inc_conv_v2(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * out_tensor, int padding, int stride)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights  = THCudaTensor_data(NULL, weights);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);

    int groups = 1;
    int batch = in_tensor->size[0];

    int m = out_tensor->size[1]/groups;
    int k = weights->size[3] * weights->size[2] * weights->size[1]/groups;
    int n = out_tensor->size[3] * out_tensor->size[2];

    float * workspace = NULL;

    workspace  = cuda_make_array(workspace, ((size_t)batch)*((size_t)n)*((size_t)k)*sizeof(float));
    //batch_conv_dp_parent_gpu(in_tensor->size[1], in_tensor->size[2], weights->size[2], weights->size[0], padding, stride, ptr_in_tensor, ptr_weights, ptr_out_tensor, groups, batch, m, k, n, workspace);

    return 1;
}
