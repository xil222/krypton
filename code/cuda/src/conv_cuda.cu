#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "cuda.h"
#include "curand.h"
#include "cublas_v2.h"
#include "conv_cuda.h"

#define BLOCK 512

__global__ void update_output_locations_gpu_kernel(int num_kernels, int * ptr_location, int in_height, int in_width, int out_height, int out_width, int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y, int out_p_height, int out_p_width, int remove_x, int remove_y)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < num_kernels)
    {
        int current_y0 = ptr_location[index*2] + remove_y/2;
        int current_x0 = ptr_location[index*2+1] + remove_x/2;

        current_y0 = max((int)ceil((padding_y + current_y0-k_size_y + 1.0)/stride_x), 0);
        current_x0 = max((int)ceil((padding_x + current_x0-k_size_x + 1.0)/stride_y), 0);

        if(current_y0 + out_p_height > out_height)
        {
            current_y0 = out_height - out_p_height;
        }
        if(current_x0 + out_p_width > out_width)
        {
            current_x0 = out_width - out_p_width;
        }

        ptr_location[index*2] = current_y0;
        ptr_location[index*2+1] = current_x0;
    }
}

void update_output_locations_gpu(int* ptr_location, int batch, int in_height, int in_width, int out_height, int out_width, int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y , int out_p_height, int out_p_width, int remove_x, int remove_y)
{
    update_output_locations_gpu_kernel<<<(batch+BLOCK-1)/BLOCK, BLOCK>>>(
        batch, ptr_location, in_height, in_width, out_height, out_width, padding_x, padding_y, stride_x, stride_y, k_size_x, k_size_y, out_p_height, out_p_width, remove_x, remove_y);
}

__global__ void add_bias_gpu_kernel(float *ptr_out_tensor, float * ptr_biases, int p_height,
        int p_width, int channels, int batch, int n)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < n)
    {
        int w_out = index % p_width;
        int h_index = index / p_width;
        int h_out = h_index % p_height;
        int channel_in = h_index / p_height;

        #pragma unroll 4
        for(int i=0; i<batch; i++)
        {
            ptr_out_tensor[channel_in*p_height*p_width+h_out*p_width+w_out] += ptr_biases[channel_in];
            ptr_out_tensor += p_width*p_height*channels;
        }
    }
}

void add_bias_gpu(float *ptr_out_tensor, float * ptr_biases, int batch, int p_height, int p_width, int channels)
{
    int num_kernels = p_height * p_width * channels;
    add_bias_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(ptr_out_tensor, ptr_biases, p_height, p_width, channels, batch, num_kernels);
}

__global__ void extract_input_volume_gpu_kernel(int num_kernels, float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location, int batch, int channels, int in_height, int in_width, int out_height, int out_width, int stride_x, int stride_y, int padding_x, int padding_y,  int p_height, int p_width, int in_p_height, int in_p_width, int out_p_height, int out_p_width, int k_size_x, int k_size_y, int remove_x, int remove_y)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index < num_kernels)
    {
        int w = index % in_p_width;
        int h_index = index / in_p_width;
        int h = h_index % in_p_height;
        int channel_in = h_index / in_p_height;

        out_ptr += index;
                
        #pragma unroll 4
        for(int i=0; i<batch; i++)
        {
            int current_y0 = ptr_location[i*2] + remove_y/2;
            int current_x0 = ptr_location[i*2+1] + remove_x/2;

            int new_y0, new_x0;

            new_y0 = max((int)ceil((padding_y + current_y0-k_size_y + 1.0)/stride_y), 0);
            new_x0 = max((int)ceil((padding_x + current_x0-k_size_x + 1.0)/stride_x), 0);

            
            if(new_y0 + out_p_height > out_height)
            {
                new_y0 = out_height - out_p_height;
            }
            if(new_x0 + out_p_width > out_width)
            {
                new_x0 = out_width - out_p_width;
            }
                
            int w_out = new_x0*stride_x - padding_x + w;
            int h_out = new_y0*stride_y - padding_y + h;

            if(w_out < 0 || w_out >= in_width || h_out < 0 || h_out >= in_height)
            {
                *out_ptr = 0;
            }
            else
            {
                current_y0 -= remove_y/2;
                current_x0 -= remove_x/2;
                
                if((w_out < current_x0) || (w_out >= (current_x0 + p_width + remove_x))
                   || (h_out < current_y0) || (h_out >= (current_y0 + p_height + remove_y)))
                {
                    *out_ptr = premat_ptr[channel_in*in_height*in_width+h_out*in_width+w_out];
                }
                else
                {
                    w_out -= current_x0;
                    h_out -= current_y0;
                    
                    *out_ptr = in_ptr[channel_in*(p_height+remove_y)*(p_width+remove_x)+h_out*(p_width+remove_x)+w_out];
                }
            }
            out_ptr += in_p_width*in_p_height*channels;
            in_ptr += (p_height+remove_y)*(p_width+remove_x)*channels;
        }
        
    }
}

void extract_input_volume_gpu(float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location,
 int batch, int channels, int in_height, int in_width, int out_height, int out_width,
 int padding_x, int padding_y, int stride_x, int stride_y, int p_height, int p_width,
 int in_p_height, int in_p_width, int out_p_height, int out_p_width, int k_size_x, int k_size_y, int remove_x, int remove_y)
{
    int num_kernels = in_p_height*in_p_width*channels;
    extract_input_volume_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(num_kernels, premat_ptr, in_ptr, out_ptr,
             ptr_location, batch, channels, in_height, in_width, out_height, out_width,
             padding_x, padding_y, stride_x, stride_y,  p_height, p_width,
             in_p_height, in_p_width, out_p_height, out_p_width, k_size_x, k_size_y, remove_x, remove_y);
}



__global__ void inc_max_pool_gpu_kernel(int n, float* ptr_premat_tensor, float* ptr_in_tensor, float* ptr_out_tensor, int * ptr_location, int batch, int channels, int in_height, int in_width, int out_height, int out_width, int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y, int p_height, int p_width, int out_p_height, int out_p_width, int remove_x, int remove_y)
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

        int w_offset = -padding_x;
        int h_offset = -padding_y;
        
        int current_y0 = ptr_location[b*2] + remove_y/2;
        int current_x0 = ptr_location[b*2+1] + remove_x/2;

        int new_y0, new_x0;

        new_y0 = max((int)ceil((padding_y + current_y0-k_size_y + 1.0)/stride_y), 0);
        new_x0 = max((int)ceil((padding_x + current_x0-k_size_x + 1.0)/stride_x), 0);
            
        if(new_y0 + out_p_height > out_height)
        {
            new_y0 = out_height - out_p_height;
        }
        if(new_x0 + out_p_width > out_width)
        {
            new_x0 = out_width - out_p_width;
        }

        i = i + new_y0;
        j = j + new_x0;

        float max = -INFINITY;

        current_y0 -= remove_y/2;
        current_x0 -= remove_x/2;        
        
        #pragma unroll 3
        for(int l=0; l<k_size_y; l++)
        {
            #pragma unroll 3
            for(int m=0; m<k_size_x; m++)
            {
                int cur_h = h_offset + i*stride_y + l;
                int cur_w = w_offset + j*stride_x + m;

                float val;
                int idx, valid;
                if(cur_w<current_x0 || cur_h<current_y0 || cur_w>=(current_x0+p_width+remove_x) || cur_h>=(current_y0+p_height+remove_y))
                {
                    idx = cur_w + in_width*(cur_h + in_height*k);
                    valid = (cur_h >= 0 && cur_h < in_height && cur_w >= 0 && cur_w < in_width);
                    val = (valid != 0) ? ptr_premat_tensor[idx] : -INFINITY;
                }
                else
                {
                    cur_h -= current_y0;
                    cur_w -= current_x0;
                    
                    idx = cur_w + (p_width+remove_x)*(cur_h + (p_height+remove_y)*(k + b*channels));
                    val = ptr_in_tensor[idx];
                }
                max   = (val > max) ? val   : max;
            }
        }

        ptr_out_tensor[blockIdx.x * blockDim.x + threadIdx.x] = max;
    }
}

void inc_max_pool_gpu(float* ptr_premat_tensor, float* ptr_in_tensor, float* ptr_out_tensor, int * ptr_location, int batch, int channels, int in_height, int in_width, int out_height, int out_width,  int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y, int p_height, int p_width, int out_p_height, int out_p_width, int remove_x, int remove_y)
{
    size_t n =  out_p_height * out_p_width * channels * batch;
    inc_max_pool_gpu_kernel<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(n, ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor, ptr_location,
        batch, channels, in_height, in_width, out_height, out_width, padding_x, padding_y, stride_x, stride_y,
        k_size_x, k_size_y, p_height, p_width, out_p_height, out_p_width, remove_x, remove_y);
}

__global__ void  full_projection_gpu_kernel(int n, float * ptr_premat_tensor, float * ptr_in_tensor,
 float * ptr_out_tensor, int * ptr_location, int batch, int channels, int in_height, int in_width, int p_height, int p_width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n)
    {
        int w = index % in_width;
        int h_index = index / in_width;
        int h = h_index % in_height;
        int c_index = h_index / in_height;
        int c = c_index % channels;
        int b = c_index / channels;
        
        int patch_x0 = ptr_location[b*2+1];
        int patch_y0 = ptr_location[b*2];
        
        if(w < patch_x0 || w >= patch_x0 + p_width || h < patch_y0 || h >= patch_y0 + p_height)
        {
            ptr_out_tensor[index] = ptr_premat_tensor[w + h*in_width + c*in_width*in_height];
        }
        else
        {
            w = w - patch_x0;
            h = h - patch_y0;
            
            ptr_out_tensor[index] = ptr_in_tensor[w + h*p_width + c*p_width*p_height + b*p_width*p_height*channels];
        }
    }
}

void full_projection_gpu(float * ptr_premat_tensor, float * ptr_in_tensor, float * ptr_out_tensor, int * ptr_location, int batch, int channels, int in_height, int in_width, int p_height, int p_width)
{
    size_t n =  in_height * in_width * channels * batch;
    full_projection_gpu_kernel<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(n, ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor,
     ptr_location, batch, channels, in_height, in_width, p_height, p_width);
}
