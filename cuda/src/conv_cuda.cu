#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "cuda.h"
#include "curand.h"
#include "cublas_v2.h"
#include "conv_cuda.h"

#define BLOCK 512

__global__ void update_output_locations_gpu_kernel(int num_kernels, int * ptr_location, int size_x, int size_y,
    int padding_x, int padding_y, int stride_x, int stride_y, int k_size_x, int k_size_y,
    int in_p_height, int in_p_width, bool patch_growing)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < num_kernels)
    {
        int current_y0 = ptr_location[index*2];
        int current_x0 = ptr_location[index*2+1];

        int out_size_x = (size_x - k_size_x + 2*padding_x)/stride_x + 1;
        int out_size_y = (size_y - k_size_y + 2*padding_y)/stride_y + 1;

        int out_p_width = (in_p_width-k_size_x)/stride_x + 1;
        int out_p_height = (in_p_height-k_size_y)/stride_y + 1;

        if(patch_growing)
        {
            current_y0 = max((int)ceil((padding_y + current_y0-k_size_y + 1.0)/stride_x), 0);
            current_x0 = max((int)ceil((padding_x + current_x0-k_size_x + 1.0)/stride_y), 0);
        }
        else
        {
            current_y0 = round(current_y0*out_size_y/(float)size);
            current_x0 = round(current_x0*out_size_x/(float)size);
        }

        if(current_y0 + out_p_height > out_size_y)
        {
            current_y0 = out_size_y - out_p_height;
        }
        if(current_x0 + out_p_width > out_size_x)
        {
            current_x0 = out_size_x - out_p_width;
        }

        ptr_location[index*2] = current_y0;
        ptr_location[index*2+1] = current_x0;
    }
}

void update_output_locations_gpu(int* ptr_location, int batch, int size_x, int size_y, int padding_x, int padding_y, int stride_x,
    int stride_y, int k_size_x, int k_size_y , int in_p_height, int in_p_width, bool patch_growing)
{
    update_output_locations_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
        batch, ptr_location, size_x, size_y, padding_x, padding_y, stride_x, stride_y, k_size_x, k_size_y,
        in_p_height, in_p_width, patch_growing);
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

        float temp = 0.0;
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

__global__ void extract_input_volume_gpu_kernel(int num_kernels, float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location,
 int batch, int channels, int in_height, int in_width, int stride_x, int stride_y,
 int padding_x, int padding_y,  int p_height, int p_width, int in_p_height, int in_p_width, int k_size_x, int k_size_y, int patch_growing)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index < num_kernels)
    {
        int w = index % in_p_width;
        int h_index = index / in_p_width;
        int h = h_index % in_p_height;
        int channel_in = h_index / in_p_height;

        out_ptr += index;

        //output size
        int out_size_x = (in_width - k_size_x + 2*padding_x)/stride_x + 1;
        int out_size_y = (in_height - k_size_y + 2*padding_y)/stride_y + 1;

        //output patch size
        int out_p_width = (in_p_width-k_size_x)/stride_x + 1;
        int out_p_height = (in_p_height-k_size_y)/stride_y + 1;
                
        #pragma unroll 4
        for(int i=0; i<batch; i++)
        {
            int current_y0 = ptr_location[i*2];
            int current_x0 = ptr_location[i*2+1];

            int new_y0, new_x0;
            if(patch_growing)
            {
                new_y0 = max((int)ceil((padding_y + current_y0-k_size_y + 1.0)/stride_y), 0);
                new_x0 = max((int)ceil((padding_x + current_x0-k_size_x + 1.0)/stride_x), 0);
            }
            else
            {
                new_y0 = round(current_y0*out_size_y/(float)in_height);
                new_x0 = round(current_x0*out_size_x/(float)in_width);
            }

            
            if(new_y0 + out_p_height > out_size_y)
            {
                new_y0 = out_size_y - out_p_height;
            }
            if(new_x0 + out_p_width > out_size_x)
            {
                new_x0 = out_size_x - out_p_width;
            }
                
            int w_out = new_x0*stride_x - padding_x + w;
            int h_out = new_y0*stride_y - padding_y + h;

            if(w_out < 0 || w_out >= in_width || h_out < 0 || h_out >= in_height)
            {
                *out_ptr = 0;
            }
            else
            {
                if((w_out < current_x0) || (w_out >= (current_x0 + p_width))
                   || (h_out < current_y0) || (h_out >= (current_y0 + p_height)))
                {
                    *out_ptr = premat_ptr[channel_in*in_size_x*in_size_y+h_out*in_size_x+w_out];
                }
                else
                {
                    w_out -= current_x0;
                    h_out -= current_y0;
                    *out_ptr = in_ptr[channel_in*p_height*p_width+h_out*p_width+w_out];
                }
            }
            out_ptr += in_p_width*in_p_height*channels;
            in_ptr += p_height*p_width*channels;
        }
        
    }
}

void extract_input_volume_gpu(float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location,
 int batch, int channels, int in_height, int in_width, int padding_x, int padding_y, int stride_x, int stride_y,
 int p_height, int p_width, int in_p_height, int in_p_width, int k_size_x, int k_size_y, int patch_growing)
{
    int num_kernels = in_p_height*in_p_width*channels;
    cudnn_mem_copy_gpu_kernel2<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(num_kernels, premat_ptr, in_ptr, out_ptr,
             ptr_location, batch, channels, in_height, in_width, padding_x, padding_y, stride_x, stride_y,  p_height, p_width,
             in_p_height, in_p_width, k_size_x, k_size_y, patch_growing);
}



__global__ void inc_max_pool_gpu_kernel(int n, float* ptr_premat_tensor, float* ptr_in_tensor, float* ptr_out_tensor, int * ptr_location,
 int in_height, int in_width, int in_p_height, int in_p_width, int channels, int batch, int padding_x, int padding_y,
  int stride_x, int stride_y, int k_size_x, int k_size_y, int out_p_height, int out_p_width, int patch_growing)
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
        
        int current_y0 = ptr_location[b*2];
        int current_x0 = ptr_location[b*2+1];

        int out_size_x = (in_width - k_size_x + 2*padding_x)/stride_x + 1;
        int out_size_y = (in_height - k_size_y + 2*padding_y)/stride_y + 1;

        int new_y0, new_x0;
        if(patch_growing)
        {
            new_y0 = max((int)ceil((padding_y + current_y0-k_size_x + 1.0)/stride_y), 0);
            new_x0 = max((int)ceil((padding_x + current_x0-k_size_y + 1.0)/stride_x), 0);
        }
        else
        {
            new_y0 = round(current_y0*out_size_y/(float)in_size);
            new_x0 = round(current_x0*out_size_x/(float)in_size);
        }
            
        if(new_y0 + out_p_height > out_size_y)
        {
            new_y0 = out_size_y - out_p_height;
        }
        if(new_x0 + out_p_width > out_size_x)
        {
            new_x0 = out_size_x - out_p_width;
        }

        i = i + new_y0;
        j = j + new_x0;

        float max = -INFINITY;

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
                if(cur_w<current_x0 || cur_h<current_y0 || cur_w>=current_x0+in_p_width || cur_h>=current_y0+in_p_width)
                {
                    idx = cur_w + in_size*(cur_h + in_size*k);
                    valid = (cur_h >= 0 && cur_h < in_size && cur_w >= 0 && cur_w < in_size);
                    val = (valid != 0) ? ptr_premat_tensor[idx] : -INFINITY;
                }
                else
                {
                    cur_h -= current_y0;
                    cur_w -= current_x0;
                    idx = cur_w + in_p_width*(cur_h + in_p_height*(k + b*channels));
                    val = ptr_in_tensor[idx];
                }
                max   = (val > max) ? val   : max;
            }
        }

        ptr_out_tensor[blockIdx.x * blockDim.x + threadIdx.x] = max;
    }
}

void inc_max_pool_gpu(float* ptr_premat_tensor, float* ptr_in_tensor, float* ptr_out_tensor, int * ptr_location, int in_height, int in_width,
 int in_p_height, int in_p_width, int channels, int batch, int padding_x, int padding_y, int stride_x, int stride_y,
 int k_size_x, int k_size_y, int out_p_height, int out_p_width, int patch_growing)
{
    size_t n =  out_p_height * out_p_width * channels * batch;
    inc_max_pool_gpu_kernel<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(n, ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor, ptr_location,
        in_height, in_width, in_p_height, in_p_width, channels, batch, padding_x, padding_y, stride_x, stride_y,
        k_size_x, k_size_y, out_p_height, out_p_width, patch_growing);
}

__global__ void  full_projection_gpu_kernel(int n, float * ptr_premat_tensor, float * ptr_in_tensor,
 float * ptr_out_tensor, int * ptr_location, int batch, int channels, int in_size, int p_height, int p_width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n)
    {
        int w = index % in_size;
        int h_index = index / in_size;
        int h = h_index % in_size;
        int c_index = h_index / in_size;
        int c = c_index % channels;
        int b = c_index / channels;
        
        int patch_x0 = ptr_location[b*2+1];
        int patch_y0 = ptr_location[b*2];
        
        if(w < patch_x0 || w >= patch_x0 + p_width || h < patch_y0 || h >= patch_y0 + p_height)
        {
            ptr_out_tensor[index] = ptr_premat_tensor[w + h*in_size + c*in_size*in_size];
        }
        else
        {
            w = w - patch_x0;
            h = h - patch_y0;
            
            ptr_out_tensor[index] = ptr_in_tensor[w + h*p_width + c*p_width*p_height + b*p_width*p_height*channels];
        }
    }
}

void full_projection_gpu(float * ptr_premat_tensor, float * ptr_in_tensor, float * ptr_out_tensor, int * ptr_location,
 int batch, int channels, int in_size, int p_height, int p_width)
{
    size_t n =  in_size * in_size * channels * batch;
    full_projection_gpu_kernel<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(n, ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor,
     ptr_location, batch, channels, in_size, p_height, p_width);
}
