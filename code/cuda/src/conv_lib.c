#include <THC/THC.h>
#include <stdio.h>
#include <sys/time.h>
#include "conv_cuda.h"

#include <cuda.h>
#include <cudnn.h>
#include <stdbool.h>

extern THCState *state;

float min(float a, float b)
{
    return a > b ? b : a;
}

float max(float a, float b)
{
    return a > b ? a : b;
}

int cuda_get_device(void)
{
    int n = 0;
    cudaGetDevice(&n);
    return n;
}

cudnnHandle_t cudnn_handle(void)
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

float * get_temp_in_tensor(size_t size)
{
    static float * tmp_in_tensor = NULL;
    static size_t current_size = 64*1024*1024*sizeof(float);
    if(tmp_in_tensor == NULL || current_size < size)
    {
        if(tmp_in_tensor != NULL) cudaFree(tmp_in_tensor);
        if(size > current_size) current_size = size;
        cudaMalloc((void**)&tmp_in_tensor, current_size);
    }
    return tmp_in_tensor;
}

float * get_cudnn_workspace(size_t size)
{
    static float * cudnn_workspace = NULL;
    static size_t current_size = 64*1024*1024*sizeof(float);
    if(cudnn_workspace == NULL || current_size < size)
    {
        if(cudnn_workspace != NULL) cudaFree(cudnn_workspace);
        if(size > current_size) current_size = size;
        cudaMalloc((void**)&cudnn_workspace, current_size);
    }
    return cudnn_workspace;
}

int inc_convolution(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases,
 THCudaTensor * out_tensor, THCudaIntTensor * location_tensor, int padding_x, int padding_y,
  int stride_x, int stride_y, int p_height, int p_width, float beta)
{
    float * ptr_premat_tensor = THCudaTensor_data(NULL, premat_tensor);
    float * ptr_in_tensor = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights = THCudaTensor_data(NULL, weights);
    float * ptr_biases = THCudaTensor_data(NULL, biases);
    float * ptr_out_tensor = THCudaTensor_data(NULL, out_tensor);

    int * ptr_location = THCudaIntTensor_data(NULL, location_tensor);

    int batch = in_tensor->size[0];
    int in_channels = in_tensor->size[1];
    int in_height = premat_tensor->size[2];
    int in_width = premat_tensor->size[3];
    
    int out_channels = out_tensor->size[1];

    int k_height = weights->size[2];
    int k_width = weights->size[3];

    cudnnHandle_t cudnn = cudnn_handle();
 
    int out_width = (in_width - k_width + 2*padding_x)/stride_x + 1;
    int out_height = (in_height - k_height + 2*padding_y)/stride_y + 1;
        
    int remove_x = 0;
    int remove_y = 0;
    if(((p_width) > round(in_width * beta)) || ((p_height) > round(in_height * beta)))
    {
        int temp_p_height = (int)round(min(beta*in_height, p_height));
        int temp_p_width = (int)round(min(beta*in_width, p_width));
        
        if((p_height-temp_p_height)%2 != 0)
        {
            temp_p_height -= 1;
        }
        if((p_width-temp_p_width)%2 != 0)
        {
            temp_p_width -= 1;
        }
        
        remove_x = p_width - temp_p_width;
        remove_y = p_height - temp_p_height;
        
        p_height = temp_p_height;
        p_width = temp_p_width;
    }

    int out_p_height = min((int)ceil((p_height+k_height-1)*1.0/stride_y), out_height);
    int out_p_width = min((int)ceil((p_width+k_width-1)*1.0/stride_x), out_width);
    
    int in_p_height = k_height + (out_p_height-1)*stride_y;
    int in_p_width = k_width + (out_p_width-1)*stride_x;

    //temp input tensor
    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, in_channels, in_p_width, in_p_height);

    float * temp_in_tensor;
    temp_in_tensor = get_temp_in_tensor(batch*in_channels*in_p_width*in_p_height*sizeof(float));

    extract_input_volume_gpu(ptr_premat_tensor, ptr_in_tensor, temp_in_tensor, ptr_location, batch, in_channels,
     in_height, in_width, out_height, out_width, stride_x, stride_y, padding_x, padding_y, p_height, p_width, in_p_height,
     in_p_width, out_p_height, out_p_width, k_width, k_height, remove_x, remove_y);

    update_output_locations_gpu(ptr_location, batch, in_height, in_width, out_height, out_width, padding_x, padding_y,
              stride_x, stride_y, k_width, k_height, out_p_height, out_p_width, remove_x, remove_y);
    
    //filter tensor
    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, k_height, k_width);

    //convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, stride_y, stride_x, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    //output descriptor
    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &batch, &out_channels, &out_p_height, &out_p_width);
    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, out_channels, out_p_height, out_p_width);
    
    //convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;//= CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    //if(p_width <= 6) algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    size_t ws_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);

    float * ws_data = NULL;
    if(ws_size>0)
    {
        ws_data = get_cudnn_workspace(ws_size);
    }

    float ALPHA = 1.f;
    float BETA = 0.f;

    cudnnConvolutionForward(cudnn, &ALPHA, in_desc, temp_in_tensor,
                filt_desc, ptr_weights, conv_desc, algo, ws_data, ws_size, &BETA,
                out_desc, ptr_out_tensor);
    
    add_bias_gpu(ptr_out_tensor, ptr_biases, batch, out_p_height, out_p_width, out_channels);

    return out_p_height*1000+out_p_width;
}

void batch_normalization(THCudaTensor * in_tensor,  THCudaTensor * bn_mean, THCudaTensor * bn_var, THCudaTensor * bn_weights, THCudaTensor * bn_biases, float eps)
{
    float * ptr_in_tensor = THCudaTensor_data(NULL, in_tensor);    
    float * ptr_bn_mean = THCudaTensor_data(NULL, bn_mean);
    float * ptr_bn_var = THCudaTensor_data(NULL, bn_var);
    float * ptr_bn_weights = THCudaTensor_data(NULL, bn_weights);
    float * ptr_bn_biases = THCudaTensor_data(NULL, bn_biases);
    
    int batch = in_tensor->size[0];
    int in_channels = in_tensor->size[1];
    int in_height = in_tensor->size[2];
    int in_width = in_tensor->size[3];
    
    bn_gpu(ptr_in_tensor, ptr_bn_mean, ptr_bn_var, ptr_bn_weights, ptr_bn_biases, eps, batch, in_height, in_width, in_channels);
}

void inc_add(THCudaTensor * in_tensor1, THCudaIntTensor * location_tensor1, THCudaTensor * premat_tensor2, THCudaTensor * in_tensor2, THCudaIntTensor * location_tensor2)
{
    float * ptr_in_tensor1 = THCudaTensor_data(NULL, in_tensor1);
    float * ptr_premat_tensor2 = THCudaTensor_data(NULL, premat_tensor2);
    float * ptr_in_tensor2 = THCudaTensor_data(NULL, in_tensor2);
    
    int * ptr_location1 = THCudaIntTensor_data(NULL, location_tensor1);
    int * ptr_location2 = THCudaIntTensor_data(NULL, location_tensor2);
    
    int batch = in_tensor1->size[0];
    int channels = in_tensor1->size[1];
    int in_height1 = in_tensor1->size[2];
    int in_width1 = in_tensor1->size[3];

    int in_height2 = in_tensor2->size[2];
    int in_width2 = in_tensor2->size[3];
    
    int premat_height = premat_tensor2->size[2];
    int premat_width = premat_tensor2->size[3];

    inc_add_gpu(ptr_in_tensor1, ptr_location1, ptr_premat_tensor2, ptr_in_tensor2, ptr_location2, batch, channels, in_height1,
               in_width1, in_height2, in_width2, premat_height, premat_width);
}


int inc_max_pool(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,  THCudaIntTensor * location_tensor, int padding_x, int padding_y, int stride_x, int stride_y, int k_width, int k_height, int p_height, int p_width, float beta)
{
    float * ptr_premat_tensor = THCudaTensor_data(NULL, premat_tensor);
    float * ptr_in_tensor = THCudaTensor_data(NULL, in_tensor);
    float * ptr_out_tensor = THCudaTensor_data(NULL, out_tensor);    
    int * ptr_location = THCudaIntTensor_data(NULL, location_tensor);

    int batch = in_tensor->size[0];
    int in_channels = in_tensor->size[1];
    int in_height = premat_tensor->size[2];
    int in_width = premat_tensor->size[3];

    int out_width = (in_width - k_width + 2*padding_x)/stride_x + 1;
    int out_height = (in_height - k_height + 2*padding_y)/stride_y + 1;

    int remove_x = 0;
    int remove_y = 0;
    if(((p_width) > round(in_width * beta)) || ((p_height) > round(in_height * beta)))
    {
        int temp_p_height = (int)round(min(beta*in_height, p_height));
        int temp_p_width = (int)round(min(beta*in_width, p_width));
        
        if((p_height-temp_p_height)%2 != 0)
        {
            temp_p_height -= 1;
        }
        if((p_width-temp_p_width)%2 != 0)
        {
            temp_p_width -= 1;
        }
        
        remove_x = p_width - temp_p_width;
        remove_y = p_height - temp_p_height;
        
        p_height = temp_p_height;
        p_width = temp_p_width;
    }
    
    int out_p_height = min((int)ceil((p_height+k_height-1)*1.0/stride_y), out_height);
    int out_p_width = min((int)ceil((p_width+k_width-1)*1.0/stride_x), out_width);   
    
    inc_max_pool_gpu(ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor, ptr_location, batch, in_channels, in_height, in_width, out_height, out_width, padding_x, padding_y, stride_x, stride_y, k_width, k_height, p_height, p_width, out_p_height, out_p_width, remove_x, remove_y);

    update_output_locations_gpu(ptr_location, batch, in_height, in_width, out_height, out_width, padding_x, padding_y,
         stride_x, stride_y, k_width, k_height, out_p_height, out_p_width, remove_x, remove_y);
    
    return out_p_height*1000+out_p_width;
}


int full_projection(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,
 THCudaIntTensor * location_tensor, int p_height, int p_width)
{
    float * ptr_premat_tensor = THCudaTensor_data(NULL, premat_tensor);
    float * ptr_in_tensor = THCudaTensor_data(NULL, in_tensor);
    float * ptr_out_tensor = THCudaTensor_data(NULL, out_tensor);    
    int * ptr_location = THCudaIntTensor_data(NULL, location_tensor);

    int batch = in_tensor->size[0];
    int in_channels = in_tensor->size[1];
    int in_height = premat_tensor->size[2];
    int in_width = premat_tensor->size[3];
    
    full_projection_gpu(ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor, ptr_location, batch, in_channels, in_height, in_width, p_height, p_width);
        
    return 0;
}

//     int out_p_height = min((int)ceil((p_height+k_height-1)*1.0/stride_y), out_height);
//     int out_p_width = min((int)ceil((p_width+k_width-1)*1.0/stride_x), out_width);
    
//     int remove_x=0;    
//     int remove_y=0;    
//     if(((out_p_width) > round(out_width * beta)) || ((out_p_height) > round(out_height * beta)))
//     {
//         int out_p_width_temp = (int)round(min(beta*out_width, out_p_width));
//         int out_p_height_temp = (int)round(min(beta*out_height, out_p_height));
        
//         remove_x = (out_p_width-out_p_width_temp)*stride_x;
//         remove_y = (out_p_height-out_p_height_temp)*stride_y;

//         p_height -= remove_y;
//         p_width -= remove_x;        
        
//         out_p_width = out_p_width_temp;
//         out_p_height = out_p_height_temp;        
//     }

//     int in_p_height = k_height + (out_p_height-1)*stride_y;
//     int in_p_width = k_width + (out_p_width-1)*stride_x;

//     if((p_height > round(in_height * beta) || p_width > round(in_width * beta)) && p_height >= 3 && p_width >= 3)
//     {
//         remove_x = max(2, ((p_width - (int)round(in_width * beta))/2)*2);
//         remove_y = max(2, ((p_height - (int)round(in_height * beta))/2)*2);
        
//         p_height -= remove_y;
//         p_width -= remove_x;
        
//         out_p_height = min((int)ceil(((p_height)+k_height-1)*1.0/stride_y), out_height);
//         out_p_width = min((int)ceil(((p_width)+k_width-1)*1.0/stride_x), out_width);        
//     }