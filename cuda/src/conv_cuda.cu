#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "cuda.h"
#include "curand.h"
#include "cublas_v2.h"

#include "conv_cuda.h"

#define BLOCK 512

__global__ void cudnn_mem_copy_gpu_kernel(int num_kernels, int batch, int channels, int size, int stride,
 int padding, float *in_ptr,
 float* out_ptr, int * ptr_location, int p_height, int p_width)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index < num_kernels)
    {
        int h_index = index / p_width;
        int channel_in = h_index / p_height;

        out_ptr += index;

        for(int i=0; i<batch; i++)
        {
            int w_out = *(ptr_location+i*2+1)*stride - padding + index % p_width;
            int h_out = *(ptr_location+i*2)*stride - padding + h_index % p_height;

            if(w_out >= size || w_out < 0 || h_out >= size || h_out < 0)
            {
                *out_ptr = 0;
            }
            else
            {
                *out_ptr = in_ptr[channel_in*size*size+h_out*size+w_out];
            }
            out_ptr += p_width*p_height*channels;
            in_ptr += size*size*channels;
        }
    }
}

void cudnn_mem_copy_gpu(int batch, int channels, int size, int padding, int stride, float *in_ptr, float* out_ptr, int * ptr_location, int in_p_height, int in_p_width)
{
    int num_kernels = in_p_height*in_p_width*channels;
    cudnn_mem_copy_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(num_kernels, batch, channels, size, padding, stride, in_ptr, out_ptr, ptr_location, in_p_height, in_p_width);
}

__global__ void update_output_locations_gpu_kernel(int num_kernels, int * ptr_location, int size, int padding,
    int stride, int k_size, int in_p_height, int in_p_width, bool patch_growing)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < num_kernels)
    {
        int current_y0 = ptr_location[index*2];
        int current_x0 = ptr_location[index*2+1];

        int out_size = (size - k_size + 2*padding)/stride + 1;
        int out_p_width = (in_p_width-k_size)/stride + 1;
        
        if(patch_growing)
        {
            current_y0 = max((int)ceil((padding + current_y0-k_size + 1.0)/stride), 0);
            current_x0 = max((int)ceil((padding + current_x0-k_size + 1.0)/stride), 0);
        }
        else
        {
            current_y0 = floor(current_y0*out_size/(float)size);
            current_x0 = floor(current_x0*out_size/(float)size);
        }

        if(current_y0 + out_p_width > out_size)
        {
            current_y0 = out_size - out_p_width;
        }
        if(current_x0 + out_p_width > out_size)
        {
            current_x0 = out_size - out_p_width;
        }

        ptr_location[index*2] = current_y0;
        ptr_location[index*2+1] = current_x0;
    }
}

void update_output_locations_gpu(int batch, int* ptr_location, int size, int padding, int stride,
    int k_size, int in_p_height, int in_p_width, bool patch_growing)
{
    int num_kernels = batch;
    update_output_locations_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
        num_kernels, ptr_location, size, padding, stride, k_size, in_p_height, in_p_width, patch_growing);
}

__global__ void inc_max_pool_gpu_kernel(int n, float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int out_size, int channels,
    int batch, int padding, int stride, int k_size, int * ptr_location, int out_p_height, int out_p_width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n)
    {
        int j = index % out_p_width;
        index /= out_p_width;
        int i = index % out_p_height;
        index /= out_p_height;
        int k = index % channels;
        index /= channels;
        int b = index;

        int w_offset = -padding;
        int h_offset = -padding;

        i = i + ptr_location[2*b];
        j = j + ptr_location[2*b+1];

        int out_index = j + out_size*(i+out_size*(k+channels*b));
        float max = -INFINITY;

        for(int l=0; l<k_size; l++)
        {
            for(int m=0; m<k_size; m++)
            {
                int cur_h = h_offset + i*stride + l;
                int cur_w = w_offset + j*stride + m;
                int index = cur_w + in_size*(cur_h + in_size*(k + b*channels));
                int valid = (cur_h >= 0 && cur_h < in_size && cur_w >= 0 && cur_w < in_size);
                float val = (valid != 0) ? ptr_in_tensor[index] : -INFINITY;
                max   = (val > max) ? val   : max;
            }
        }

        ptr_out_tensor[out_index] = max;
    }
}

void inc_max_pool_gpu(float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int out_size, int channels,
    int batch, int padding, int stride, int k_size, int * ptr_location, int out_p_height, int out_p_width)
{
    size_t n =  out_p_height * out_p_width * channels * batch;
    inc_max_pool_gpu_kernel<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(n, ptr_in_tensor, ptr_out_tensor,
        in_size, out_size, channels, batch, padding, stride, k_size, ptr_location, out_p_height, out_p_width);
}

__global__ void relu_fused_mem_copy_gpu_kernel(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_biases, int * ptr_location, int p_height,
        int p_width, int channels, int size, int batch, int n)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < n)
    {
        int h_index = index / p_width;
        int channel_in = h_index / p_height;

        float *in_data_ptr = ptr_temp_tensor + index;

        float temp = 0.0;
        for(int i=0; i<batch; i++)
        {
            int w_out = *(ptr_location+i*2+1) + index % p_width;
            int h_out = *(ptr_location+i*2) + h_index % p_height;

            //adding bias and performing ReLU activation
            temp = *in_data_ptr + ptr_biases[channel_in];
            ptr_out_tensor[channel_in*size*size+h_out*size+w_out] =  temp > 0 ? temp : 0;

            in_data_ptr += p_width*p_height*channels;
            ptr_out_tensor += size*size*channels;
        }
    }
}

void relu_fused_mem_copy_gpu(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_biases, int * ptr_location, int batch, int p_height, int p_width, int channels, int size)
{
    int num_kernels = p_height * p_width * channels;
    relu_fused_mem_copy_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(ptr_temp_tensor, ptr_out_tensor, ptr_biases, ptr_location, p_height, p_width, channels, size, batch, num_kernels);
}