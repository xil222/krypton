#include <THC/THC.h>
#include <stdio.h>
#include "conv_cuda.h"

extern THCState *state;

int inc_conv(THCudaTensor *input, THCudaTensor *weights, THCudaTensor *output, int padding, int stride)
{
    float* ptr_input    = THCudaTensor_data(NULL, input);
	float* ptr_weights  = THCudaTensor_data(NULL, weights);
	float* ptr_output   = THCudaTensor_data(NULL, output);

    int i,j;

    int groups = 1;
    int batch = input->size[0];

    int m = input->size[1]/groups;
    int k = weights->size[3] * weights->size[2] * weights->size[1]/groups;
    int n = output->size[3] * output->size[2];

    float* workspace;
    workspace = cuda_make_array(workspace, ((size_t)n)*((size_t)k)*sizeof(float));

    for(i = 0; i < batch; i++){
        for(j = 0; j < groups; j++){
            im2col_gpu(ptr_input + i * input->size[1] * input->size[2] * input->size[3], input->size[1], input->size[2],
                input->size[3], weights->size[3], stride, padding, workspace);
            float * a = ptr_weights + j*weights->size[0]/groups;
            float * b = workspace;
            float * c = ptr_output;
            gemm_gpu(0, 0, m,n, k, 1., a, k, b, n, 1., c+i*m*n, n);
        }
    }

    cuda_free_array(workspace);

    return 1;
}