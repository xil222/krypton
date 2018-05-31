#include <THC/THC.h>
#include <stdio.h>
#include <sys/time.h>
#include "conv_cuda.h"

#include <cuda.h>
#include <cudnn.h>
#include <stdbool.h>

extern THCState *state;

int min(int a, int b)
{
    return a > b ? b : a;
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

float * get_temp_out_tensor(size_t size)
{
    static float * tmp_out_tensor = NULL;
    static size_t current_size = 64*1024*1024*sizeof(float);
    if(tmp_out_tensor == NULL || current_size < size)
    {
        if(tmp_out_tensor != NULL) cudaFree(tmp_out_tensor);
        if(size > current_size) current_size = size;
        cudaMalloc((void**)&tmp_out_tensor, current_size);
    }
    return tmp_out_tensor;
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

int inc_conv_relu2(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases,
 THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor, int out_size, int padding,
  int stride, int p_height, int p_width, float beta)
{
    float * ptr_premat_tensor = THCudaTensor_data(NULL, premat_tensor);
    float * ptr_in_tensor = THCudaTensor_data(NULL, in_tensor);    
    
    float * ptr_weights = THCudaTensor_data(NULL, weights);
    float * ptr_biases = THCudaTensor_data(NULL, biases);
    
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);
    
    float * ptr_out_tensor = THCudaTensor_data(NULL, out_tensor);
    
    int batch = in_tensor->size[0];

    int in_channels = in_tensor->size[1];
    int in_size = premat_tensor->size[2];
    
    int out_channels = out_tensor->size[1];
    
    int k_size = weights->size[2];

    int n = batch;

    cudnnHandle_t cudnn = cudnn_handle();
 
    int temp_p_height = min((int)ceil((p_height+k_size-1)*1.0/stride), out_size);
    int temp_p_width = min((int)ceil((p_width+k_size-1)*1.0/stride), out_size);

    int out_p_height = 0;
    int out_p_width = 0;

    bool patch_growing = 1;
    if(temp_p_height > round(out_size * beta))
    {
        out_p_height = (int)ceil(p_height*1.0/stride);
        out_p_width = (int)ceil(p_width*1.0/stride);
        
        patch_growing = 0;
    }
    else
    {
        out_p_height = temp_p_height;
        out_p_width = temp_p_width;
    }

      
    int in_p_height = k_size + (out_p_height-1)*stride;
    int in_p_width = k_size + (out_p_width-1)*stride;

    //temp input tensor
    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, in_channels, in_p_width, in_p_height);

    float * temp_in_tensor;
    temp_in_tensor = get_temp_in_tensor(n*in_channels*in_p_width*in_p_height*sizeof(float));

    cudnn_mem_copy_gpu2(batch, in_channels, in_size, stride, padding, padding, ptr_premat_tensor, ptr_in_tensor,
     temp_in_tensor, ptr_location, p_height, in_p_height, in_p_width, k_size, k_size, patch_growing);    

    update_output_locations_gpu(batch, ptr_location, in_size, padding, padding, stride, k_size, k_size, in_p_height, in_p_width,
     patch_growing);
    
    //filter tensor
    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, k_size, k_size);

    //convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    //output descriptor
    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &n, &out_channels, &out_p_height, &out_p_width);
    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, out_channels, out_p_height, out_p_width);
    
    //convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;// = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    if(p_width <= 6) algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

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
    
    relu_add_bias_gpu(ptr_out_tensor, ptr_biases, batch, out_p_height, out_p_width, out_channels);
    
    return out_p_height*1000+out_p_width;
}

int inc_max_pool2(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,  THCudaIntTensor * patch_location_tensor, int out_size, int padding, int stride, int k_size, int p_height, int p_width, float beta)
{
    float * ptr_premat_tensor = THCudaTensor_data(NULL, premat_tensor);
    float * ptr_in_tensor = THCudaTensor_data(NULL, in_tensor);
    float * ptr_out_tensor = THCudaTensor_data(NULL, out_tensor);    
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);

    int batch = in_tensor->size[0];

    int in_channels = in_tensor->size[1];
    int in_size = premat_tensor->size[2];
    
    int temp_p_height = min((int)ceil((p_height+k_size-1)*1.0/stride), out_size);
    int temp_p_width = min((int)ceil((p_width+k_size-1)*1.0/stride), out_size);
    
    int out_p_height = 0;
    int out_p_width = 0;

    bool patch_growing = 1;
    if(temp_p_height > round(out_size * beta))
    {
        out_p_height = (int)ceil(p_height*1.0/stride);
        out_p_width = (int)ceil(p_width*1.0/stride);
        
        patch_growing = 0;
    }
    else
    {
        out_p_height = temp_p_height;
        out_p_width = temp_p_width;
    }

    int in_p_height = k_size + (out_p_height-1)*stride;
    int in_p_width = k_size + (out_p_width-1)*stride;

    inc_max_pool_gpu2(ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor, in_size, out_size, p_height, in_channels, batch, padding, stride, k_size, ptr_location, out_p_height, out_p_width, patch_growing);

    update_output_locations_gpu(batch, ptr_location, in_size, padding, padding, stride, k_size, k_size, in_p_height, in_p_width,
     patch_growing);
    
    return out_p_height*1000+out_p_width;
}


int final_full_projection(THCudaTensor * premat_tensor, THCudaTensor * in_tensor, THCudaTensor * out_tensor,  THCudaIntTensor * patch_location_tensor, int p_height, int p_width)
{
    float * ptr_premat_tensor = THCudaTensor_data(NULL, premat_tensor);
    float * ptr_in_tensor = THCudaTensor_data(NULL, in_tensor);
    float * ptr_out_tensor = THCudaTensor_data(NULL, out_tensor);    
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);

    int batch = in_tensor->size[0];

    int in_channels = in_tensor->size[1];
    int in_size = premat_tensor->size[2];
    
    final_full_projection_gpu(ptr_premat_tensor, ptr_in_tensor, ptr_out_tensor, ptr_location, batch, in_channels, in_size, p_height, p_width);
        
    return 0;
}


int inc_conv_relu(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases, THCudaTensor * out_tensor,
 THCudaIntTensor * patch_location_tensor, int padding, int stride, int p_height, int p_width,
 float beta)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights  = THCudaTensor_data(NULL, weights);
    float * ptr_biases = THCudaTensor_data(NULL, biases);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);

    int batch = in_tensor->size[0];

    int in_channels = in_tensor->size[1];
    int in_size = in_tensor->size[2];

    int out_channels = out_tensor->size[1];
    int out_size = out_tensor->size[2];

    int k_size = weights->size[2];

    int n = batch;

    cudnnHandle_t cudnn = cudnn_handle();

    
    int temp_p_height = min((int)ceil((p_height+k_size-1)*1.0/stride), out_size);
    int temp_p_width = min((int)ceil((p_width+k_size-1)*1.0/stride), out_size);

    
    int out_p_height = 0;
    int out_p_width = 0;

    bool patch_growing = 1;
    if(temp_p_height > round(out_size * beta))
    {
        out_p_height = round(out_size * beta);
        out_p_width = round(out_size * beta);
        
        patch_growing = 0;
    }
    else
    {
        out_p_height = temp_p_height;
        out_p_width = temp_p_width;
    }

    int in_p_height = k_size + (out_p_height-1)*stride;
    int in_p_width = k_size + (out_p_width-1)*stride;

    update_output_locations_gpu(batch, ptr_location, in_size, padding, padding, stride, k_size, k_size, in_p_height, in_p_width,
     patch_growing);

    //temp input tensor
    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, in_channels, in_p_width, in_p_height);

    float * temp_in_tensor;
    temp_in_tensor = get_temp_in_tensor(n*in_channels*in_p_width*in_p_height*sizeof(float));

    cudnn_mem_copy_gpu(batch, in_channels, in_size, stride, padding, padding, ptr_in_tensor, temp_in_tensor, ptr_location, in_p_height, in_p_width);

    //filter tensor
    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, k_size, k_size);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);


    //temp output tensor
    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &n, &out_channels, &out_p_height, &out_p_width);
    float * temp_out_tensor;
    temp_out_tensor = get_temp_out_tensor(n*out_channels*out_p_height*out_p_width*sizeof(float));

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, out_channels, out_p_height, out_p_width);


    //convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;// = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    if(p_width <= 6) algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

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
                out_desc, temp_out_tensor);

    relu_fused_mem_copy_gpu(temp_out_tensor, ptr_out_tensor, ptr_biases, ptr_location, batch, out_p_height, out_p_width, out_channels, out_size);

    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyTensorDescriptor(in_desc);

    return out_p_height*1000+out_p_width;
}

int inc_conv_bn(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * bn_mean, THCudaTensor * bn_var,
 THCudaTensor * bn_weights,THCudaTensor * bn_biases, THCudaTensor * out_tensor,
 THCudaIntTensor * patch_location_tensor, int padding_y, int padding_x, int stride, int p_height, int p_width,
 float beta, int relu, float eps)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights  = THCudaTensor_data(NULL, weights);
    
    float * ptr_bn_mean  = THCudaTensor_data(NULL, bn_mean);
    float * ptr_bn_var  = THCudaTensor_data(NULL, bn_var);
    float * ptr_bn_weights  = THCudaTensor_data(NULL, bn_weights);
    float * ptr_bn_biases = THCudaTensor_data(NULL, bn_biases);
    
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);

    int batch = in_tensor->size[0];

    int in_channels = in_tensor->size[1];
    int in_size = in_tensor->size[2];

    int out_channels = out_tensor->size[1];
    int out_size = out_tensor->size[2];

    int k_size_x = weights->size[3];
    int k_size_y = weights->size[2];

    int n = batch;

    cudnnHandle_t cudnn = cudnn_handle();

    int temp_p_height = min((int)ceil((p_height+k_size_y-1)*1.0/stride), out_size);
    int temp_p_width = min((int)ceil((p_width+k_size_x-1)*1.0/stride), out_size);
    
    int out_p_height = 0;
    int out_p_width = 0;

    bool patch_growing = 1;
    if(temp_p_height > round(out_size * beta))
    {
        out_p_height = round(out_size * beta);
        patch_growing = 0;
    }
    else
    {
        out_p_height = temp_p_height;
    }
    if(temp_p_width > round(out_size * beta))
    {
        out_p_width = round(out_size * beta);
        patch_growing = 0;
    }
    else
    {
        out_p_width = temp_p_width;
    }

    int in_p_height = k_size_y + (out_p_height-1)*stride;
    int in_p_width = k_size_x + (out_p_width-1)*stride;
    
    update_output_locations_gpu(batch, ptr_location, in_size, padding_x, padding_y, stride, k_size_x, k_size_y, in_p_height, in_p_width, patch_growing);

    //temp input tensor
    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, in_channels, in_p_height, in_p_width);

    float * temp_in_tensor;
    temp_in_tensor = get_temp_in_tensor(n*in_channels*in_p_width*in_p_height*sizeof(float));

    cudnn_mem_copy_gpu(batch, in_channels, in_size, stride, padding_x, padding_y, ptr_in_tensor, temp_in_tensor, ptr_location, in_p_height, in_p_width);

    //filter tensor
    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, k_size_y, k_size_x);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);


    //temp output tensor  
    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &n, &out_channels, &out_p_height, &out_p_width);
    float * temp_out_tensor;
    temp_out_tensor = get_temp_out_tensor(n*out_channels*out_p_height*out_p_width*sizeof(float));

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, out_channels, out_p_height, out_p_width);


    //convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;// = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    if(p_width <= 6) algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

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
                out_desc, temp_out_tensor);

    bn_fused_mem_copy_gpu(temp_out_tensor, ptr_out_tensor, ptr_bn_mean, ptr_bn_var, ptr_bn_weights, ptr_bn_biases,
     ptr_location, batch, out_p_height, out_p_width, out_channels, out_size, relu, eps);

    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyTensorDescriptor(in_desc);

    return out_p_height*1000+out_p_width;
}

int inc_max_pool(THCudaTensor * in_tensor, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor,
 int padding, int stride, int k_size, int p_height, int p_width,
 float beta)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);

    int batch = in_tensor->size[0];

    int in_channels = in_tensor->size[1];
    int in_size = in_tensor->size[2];

    int out_size = out_tensor->size[2];

    int temp_p_height = min((int)ceil((p_height+k_size-1)*1.0/stride), out_size);
    int temp_p_width = min((int)ceil((p_width+k_size-1)*1.0/stride), out_size);

    int out_p_height = 0;
    int out_p_width = 0;

    bool patch_growing = 1;
    if(temp_p_height > round(out_size * beta))
    {   
        out_p_height = round(out_size * beta);
        out_p_width = round(out_size * beta);   
        patch_growing = 0;
    }
    else
    {
        out_p_height = temp_p_height;
        out_p_width = temp_p_width;
    }

    int in_p_height = k_size + (out_p_height-1)*stride;
    int in_p_width = k_size + (out_p_width-1)*stride;

    update_output_locations_gpu(batch, ptr_location, in_size, padding, padding, stride, k_size, k_size, in_p_height, in_p_width,
     patch_growing);

    inc_max_pool_gpu(ptr_in_tensor, ptr_out_tensor, in_size, out_size, in_channels, batch, padding, stride, k_size,
        ptr_location, out_p_height, out_p_width);

    return out_p_height*1000+out_p_width;
}

int inc_avg_pool(THCudaTensor * in_tensor, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor,
 int padding, int stride, int k_size, int p_height, int p_width,
 float beta)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);

    int batch = in_tensor->size[0];

    int in_channels = in_tensor->size[1];
    int in_size = in_tensor->size[2];

    int out_size = out_tensor->size[2];

    int temp_p_height = min((int)ceil((p_height+k_size-1)*1.0/stride), out_size);
    int temp_p_width = min((int)ceil((p_width+k_size-1)*1.0/stride), out_size);

    int out_p_height = 0;
    int out_p_width = 0;

    bool patch_growing = 1;
    if(temp_p_height > round(out_size * beta))
    {   
        out_p_height = round(out_size * beta);
        out_p_width = round(out_size * beta);   
        patch_growing = 0;
    }
    else
    {
        out_p_height = temp_p_height;
        out_p_width = temp_p_width;
    }

    int in_p_height = k_size + (out_p_height-1)*stride;
    int in_p_width = k_size + (out_p_width-1)*stride;

    update_output_locations_gpu(batch, ptr_location, in_size, padding, padding, stride, k_size, k_size, in_p_height, in_p_width,
     patch_growing);

    inc_avg_pool_gpu(ptr_in_tensor, ptr_out_tensor, in_size, out_size, in_channels, batch, padding, stride, k_size,
        ptr_location, out_p_height, out_p_width);

    return out_p_height*1000+out_p_width;
}


int inc_add(THCudaTensor * in_tensor1, THCudaTensor * in_tensor2, THCudaTensor * out_tensor, THCudaIntTensor * patch_location_tensor, int p_height, int p_width, int relu)
{
    float * ptr_in_tensor1 = THCudaTensor_data(NULL, in_tensor1);
    float * ptr_in_tensor2 = THCudaTensor_data(NULL, in_tensor2);
    float * ptr_out_tensor = THCudaTensor_data(NULL, out_tensor);
    
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);
    
    int batch = out_tensor->size[0];
    int channels = out_tensor->size[1];
    int size = out_tensor->size[2];

    inc_add_gpu(ptr_in_tensor1, ptr_in_tensor2, ptr_out_tensor, ptr_location, channels, batch, size, p_height, p_width, relu);
    return 0;
}

int update_output_locations(THCudaIntTensor * patch_location_tensor, int padding_y, int padding_x, int stride, int k_size_y, int k_size_x, int p_height, int p_width, int in_size, int out_size, float beta)
{
    int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);
    int batch = patch_location_tensor->size[0];

    int temp_p_height = min((int)ceil((p_height+k_size_y-1)*1.0/stride), out_size);
    int temp_p_width = min((int)ceil((p_width+k_size_x-1)*1.0/stride), out_size);
    
    int out_p_height = 0;
    int out_p_width = 0;

    bool patch_growing = 1;
    if(temp_p_height > round(out_size * beta))
    {
        out_p_height = round(out_size * beta);
        patch_growing = 0;
    }
    else
    {
        out_p_height = temp_p_height;
    }
    if(temp_p_width > round(out_size * beta))
    {
        out_p_width = round(out_size * beta);
        patch_growing = 0;
    }
    else
    {
        out_p_width = temp_p_width;
    }

    int in_p_height = k_size_y + (out_p_height-1)*stride;
    int in_p_width = k_size_x + (out_p_width-1)*stride;
    
    update_output_locations_gpu(batch, ptr_location, in_size, padding_x, padding_y, stride, k_size_x, k_size_y, in_p_height, in_p_width, patch_growing);
    
    return out_p_height*1000+out_p_width;
}
