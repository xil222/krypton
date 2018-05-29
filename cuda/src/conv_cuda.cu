#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "cuda.h"
#include "curand.h"
#include "cublas_v2.h"

#include "conv_cuda.h"

#define BLOCK 512

__global__ void cudnn_mem_copy_gpu_kernel(int num_kernels, int batch, int channels, int size, int stride,
 int padding_x, int padding_y, float *in_ptr,
 float* out_ptr, int * ptr_location, int p_height, int p_width)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index < num_kernels)
    {
        int h_index = index / p_width;
        int channel_in = h_index / p_height;

        out_ptr += index;

        #pragma unroll 4
        for(int i=0; i<batch; i++)
        {
            int w_out = *(ptr_location+i*2+1)*stride - padding_x + index % p_width;
            int h_out = *(ptr_location+i*2)*stride - padding_y + h_index % p_height;

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

void cudnn_mem_copy_gpu(int batch, int channels, int size, int padding_x, int padding_y, int stride, float *in_ptr, float* out_ptr, int * ptr_location, int in_p_height, int in_p_width)
{
    int num_kernels = in_p_height*in_p_width*channels;
    cudnn_mem_copy_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(num_kernels, batch, channels, size, padding_x, padding_y, stride, in_ptr, out_ptr, ptr_location, in_p_height, in_p_width);
}

__global__ void update_output_locations_gpu_kernel(int num_kernels, int * ptr_location, int size, int padding_x, int padding_y,
    int stride, int k_size_x, int k_size_y, int in_p_height, int in_p_width, bool patch_growing)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < num_kernels)
    {
        int current_y0 = ptr_location[index*2];
        int current_x0 = ptr_location[index*2+1];

        int out_size_x = (size - k_size_x + 2*padding_x)/stride + 1;
        int out_size_y = (size - k_size_y + 2*padding_y)/stride + 1;

        int out_p_width = (in_p_width-k_size_x)/stride + 1;
        int out_p_height = (in_p_height-k_size_y)/stride + 1;

        if(patch_growing)
        {
            current_y0 = max((int)ceil((padding_y + current_y0-k_size_y + 1.0)/stride), 0);
            current_x0 = max((int)ceil((padding_x + current_x0-k_size_x + 1.0)/stride), 0);
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

void update_output_locations_gpu(int batch, int* ptr_location, int size, int padding_x, int padding_y, int stride,
    int k_size_x, int k_size_y , int in_p_height, int in_p_width, bool patch_growing)
{
    int num_kernels = batch;
    update_output_locations_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK, BLOCK>>>(
        num_kernels, ptr_location, size, padding_x, padding_y, stride, k_size_x, k_size_y, in_p_height, in_p_width, patch_growing);
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

        #pragma unroll 3
        for(int l=0; l<k_size; l++)
        {
            #pragma unroll 3
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


__global__ void inc_avg_pool_gpu_kernel(int n, float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int out_size, int channels,
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
        float sum = 0;
        int count = 0;

        #pragma unroll 3
        for(int l=0; l<k_size; l++)
        {
            #pragma unroll 3
            for(int m=0; m<k_size; m++)
            {
                int cur_h = h_offset + i*stride + l;
                int cur_w = w_offset + j*stride + m;
                int index = cur_w + in_size*(cur_h + in_size*(k + b*channels));
                int valid = (cur_h >= 0 && cur_h < in_size && cur_w >= 0 && cur_w < in_size);
                float val = (valid != 0) ? ptr_in_tensor[index] : 0;
                sum = sum + val;
                count = count + 1;
            }
        }

        ptr_out_tensor[out_index] = sum/count;
    }
}

void inc_avg_pool_gpu(float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int out_size, int channels,
    int batch, int padding, int stride, int k_size, int * ptr_location, int out_p_height, int out_p_width)
{
    size_t n =  out_p_height * out_p_width * channels * batch;
    inc_avg_pool_gpu_kernel<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(n, ptr_in_tensor, ptr_out_tensor,
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
        #pragma unroll 4
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

__global__ void bn_fused_mem_copy_gpu_kernel(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_bn_mean,
 float * ptr_bn_var, float * ptr_bn_weights, float * ptr_bn_biases, int * ptr_location, int p_height,
  int p_width, int channels, int size, int batch, bool relu, float eps, int n)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < n)
    {
        int h_index = index / p_width;
        int channel_in = h_index / p_height;

        float *in_data_ptr = ptr_temp_tensor + index;

        float temp = 0.0;
        #pragma unroll 4
        for(int i=0; i<batch; i++)
        {
            int w_out = *(ptr_location+i*2+1) + index % p_width;
            int h_out = *(ptr_location+i*2) + h_index % p_height;

            //adding bias and performing ReLU activation
            temp = ptr_bn_weights[channel_in]*((*in_data_ptr) - ptr_bn_mean[channel_in])/sqrt(ptr_bn_var[channel_in] + eps) + ptr_bn_biases[channel_in];

            ptr_out_tensor[channel_in*size*size+h_out*size+w_out] = (temp < 0 && relu) ? 0 : temp;
            in_data_ptr += p_width*p_height*channels;
            ptr_out_tensor += size*size*channels;
        }
    }
}

void bn_fused_mem_copy_gpu(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_bn_mean, float * ptr_bn_var,
 float * ptr_bn_weights, float * ptr_bn_biases, int * ptr_location, int batch, int p_height, int p_width,
  int channels, int size, bool relu, float eps)
{
    int num_kernels = p_height * p_width * channels;
    bn_fused_mem_copy_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(ptr_temp_tensor, ptr_out_tensor, ptr_bn_mean, ptr_bn_var, ptr_bn_weights, ptr_bn_biases,
         ptr_location, p_height, p_width, channels, size, batch, relu,
          eps, num_kernels);
}

__global__ void inc_add_gpu_kernel(float * ptr_in_tensor1, float * ptr_in_tensor2, float * ptr_out_tensor, int * ptr_location, int channels, int batch, int size, int p_height, int p_width, bool relu, int n)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < n)
    {
        int h_index = index / p_width;
        int channel_in = h_index / p_height;
       
        float temp = 0.0;
        #pragma unroll 4
        for(int i=0; i<batch; i++)
        {
            int w_out = *(ptr_location+i*2+1) + index % p_width;
            int h_out = *(ptr_location+i*2) + h_index % p_height;

            int correct_index = channel_in*size*size+h_out*size+w_out;
            
            temp = ptr_in_tensor1[correct_index] + ptr_in_tensor2[correct_index];
            
            ptr_out_tensor[correct_index] =  (temp < 0 && relu) ? 0 : temp;
            //ptr_out_tensor[correct_index] = 100;
            
            ptr_in_tensor1 += size*size*channels;
            ptr_in_tensor2 += size*size*channels;
            ptr_out_tensor += size*size*channels;
        }
    }
}

void inc_add_gpu(float * ptr_in_tensor1, float * ptr_in_tensor2, float * ptr_out_tensor, int * ptr_location_tensor, int channels, int batch, int size, int p_height, int p_width, bool relu)
{
    int num_kernels = p_height * p_width * channels;
    inc_add_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,BLOCK>>>(ptr_in_tensor1, ptr_in_tensor2, ptr_out_tensor, ptr_location_tensor, channels, batch, size, p_height, p_width, relu, num_kernels);
}


/************Experimental***************/


__global__ void cudnn_mem_copy_gpu_kernel2(int num_kernels, int batch, int channels, int size, int stride,
 int padding_x, int padding_y, float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location, int p_height, int p_width)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index < num_kernels)
    {
        int h_index = index / p_width;
        int channel_in = h_index / p_height;

        out_ptr += index;

        #pragma unroll 4
        for(int i=0; i<batch; i++)
        {
            int w_out = *(ptr_location+i*2+1)*stride - padding_x + index % p_width;
            int h_out = *(ptr_location+i*2)*stride - padding_y + h_index % p_height;

            if(w_out >= size || w_out < 0 || h_out >= size || h_out < 0)
            {
                *out_ptr = 0;
            }
            else
            {
                if(w_out < *(ptr_location+i*2+1)*stride || w_out >= *(ptr_location+i*2+1)*stride+p_width ||
                  h_out < *(ptr_location+i*2)*stride || h_out >= *(ptr_location+i*2)*stride+p_height)
                {
                    *out_ptr = premat_ptr[channel_in*size*size+h_out*size+w_out];
                }
                else
                {
                    *out_ptr = in_ptr[channel_in*size*size+h_out*size+w_out];
                }
            }
            out_ptr += p_width*p_height*channels;
            in_ptr += size*size*channels;
        }
    }
}

void cudnn_mem_copy_gpu2(int batch, int channels, int size, int padding_x, int padding_y, int stride, float *premat_ptr, float *in_ptr, float* out_ptr, int * ptr_location, int in_p_height, int in_p_width)
{
    int num_kernels = in_p_height*in_p_width*channels;
    cudnn_mem_copy_gpu_kernel2<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(num_kernels, batch, channels, size, padding_x, padding_y, stride, premat_ptr, in_ptr, out_ptr, ptr_location, in_p_height, in_p_width);
}

__global__ void relu_add_bias_gpu_kernel(float *ptr_out_tensor, float * ptr_biases, int p_height,
        int p_width, int channels, int batch, int n)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < n)
    {
        int h_index = index / p_width;
        int channel_in = h_index / p_height;

        float temp = 0.0;
        #pragma unroll 4
        for(int i=0; i<batch; i++)
        {
            int w_out = index % p_width;
            int h_out = h_index % p_height;

            //adding bias and performing ReLU activation
            temp = ptr_out_tensor[channel_in*p_height*p_width+h_out*p_width+w_out] + ptr_biases[channel_in];
            ptr_out_tensor[channel_in*p_height*p_width+h_out*p_width+w_out] =  temp > 0 ? temp : 0;

            ptr_out_tensor += p_width*p_height*channels;
        }
    }
}

void relu_add_bias_gpu(float *ptr_out_tensor, float * ptr_biases, int batch, int p_height, int p_width, int channels)
{
    int num_kernels = p_height * p_width * channels;
    relu_add_bias_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(ptr_out_tensor, ptr_biases, p_height, p_width, channels, batch, num_kernels);
}

__global__ void inc_max_pool_gpu_kernel2(int n, float* ptr_premat_tensor, float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int in_p_size, int channels, int batch, int padding, int stride, int k_size, int * ptr_location,
                                         int out_p_height, int out_p_width)
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

        //i = i + ptr_location[2*b];
        //j = j + ptr_location[2*b+1];

        //int out_index = j + p_width*(i+p_height*(k+channels*b));
        float max = -INFINITY;

        #pragma unroll 3
        for(int l=0; l<k_size; l++)
        {
            #pragma unroll 3
            for(int m=0; m<k_size; m++)
            {
                int cur_h = h_offset + i*stride + l;
                int cur_w = w_offset + j*stride + m;

                float val;
                int idx, valid;
                
                if(cur_w<0 || cur_h<0 || cur_h>=in_p_size || cur_w>=in_p_size)
                {
                    cur_h += ptr_location[2*b]*stride;
                    cur_w += ptr_location[2*b+1]*stride;
                    idx = cur_w + in_size*(cur_h + in_size*k);
                    valid = (cur_h >= 0 && cur_h < in_size && cur_w >= 0 && cur_w < in_size);
                    val = (valid != 0) ? ptr_premat_tensor[idx] : -INFINITY;
                }
                else
                {
                    idx = cur_w + in_p_size*(cur_h + in_p_size*(k + b*channels));
                    valid = (cur_h >= 0 && cur_h < in_p_size && cur_w >= 0 && cur_w < in_p_size);
                    val = (valid != 0) ? ptr_in_tensor[idx] : -INFINITY;
                }
                max   = (val > max) ? val   : max;
            }
        }

        ptr_out_tensor[blockIdx.x * blockDim.x + threadIdx.x] = max;
    }
}

void inc_max_pool_gpu2(float* ptr_premat_tensor, float* ptr_in_tensor, float* ptr_out_tensor, int in_size, int in_p_size, int channels, int batch, int padding, int stride, int k_size, int * ptr_location, int out_p_height, int out_p_width)
{
    size_t n =  out_p_height * out_p_width * channels * batch;
    inc_max_pool_gpu_kernel2<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(n, ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor,
        in_size, in_p_size, channels, batch, padding, stride, k_size, ptr_location, out_p_height, out_p_width);
}
