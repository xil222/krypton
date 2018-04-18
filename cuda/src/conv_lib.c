#include <THC/THC.h>
#include <stdio.h>
#include <sys/time.h>
#include "conv_cuda.h"

#include <cuda.h>
#include <cudnn.h>

extern THCState *state;

int cuda_get_device(void)
{
    int n = 0;
    //cudaError_t status = cudaGetDevice(&n);
    //cuda_check_error(status);
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
        //printf("%z\n",size);
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
        //printf("%z\n",size);
    }
    return tmp_out_tensor;
}

int inc_conv_v4(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases, THCudaTensor * out_tensor,
 THCudaIntTensor * patch_location_tensor, int padding, int stride, int p_height, int p_width)
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

    //temp input tensor
    int in_p_height = p_height*stride + k_size-1;
    int in_p_width = p_width*stride + k_size-1;

    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, in_channels, in_p_width, in_p_height);

    float * temp_in_tensor;
    //cudaMalloc((void **)&temp_in_tensor, n*in_channels*in_p_width*in_p_height*sizeof(float));
    temp_in_tensor = get_temp_in_tensor(n*in_channels*in_p_width*in_p_height*sizeof(float));
    
    cudnn_mem_copy_gpu(batch, in_channels, in_size, stride, padding, ptr_in_tensor, temp_in_tensor, ptr_location, in_p_height, in_p_width);

    //filter tensor
    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, k_size, k_size);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);


    //temp output tensor
    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &n, &out_channels, &p_height, &p_width);
    float * temp_out_tensor;
    //cudaMalloc((void **)&temp_out_tensor, n*out_channels*p_height*p_width*sizeof(float));
    temp_out_tensor = get_temp_out_tensor(n*out_channels*p_height*p_width*sizeof(float));

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, out_channels, p_height, p_width);


    //convolution algorithm
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    //cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    if(p_width <= 14) algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    size_t ws_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);

    float * ws_data = NULL;
    if(ws_size>0)cudaMalloc((void **)&ws_data, ws_size);

    float alpha = 1.f;
    float beta = 0.f;

    cudnnConvolutionForward(cudnn, &alpha, in_desc, temp_in_tensor,
                filt_desc, ptr_weights, conv_desc, algo, ws_data, ws_size, &beta,
                out_desc, temp_out_tensor);

    //add_bias_gpu(temp_out_tensor, ptr_biases, batch, out_channels, p_width*p_height);
    inc_conv_mem_copy_gpu_v2(temp_out_tensor, ptr_out_tensor, ptr_location, batch, p_height, p_width, out_channels, out_size);

    if(ws_size>0)cudaFree(ws_data);
    //cudaFree(temp_in_tensor);
    //cudaFree(temp_out_tensor);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyTensorDescriptor(in_desc);

    return 0;
}


int inc_conv_v1(THCudaTensor *in_tensor, THCudaTensor *weights, THCudaTensor *biases, THCudaTensor *out_tensor, int padding, int stride)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights  = THCudaTensor_data(NULL, weights);
    float * ptr_biases = THCudaTensor_data(NULL, biases);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);

    int batch = in_tensor->size[0];

    int m = out_tensor->size[1];
    int k = weights->size[3] * weights->size[2] * weights->size[1];
    int n = out_tensor->size[3] * out_tensor->size[2];

    float * workspace = NULL;
    //workspace  = cuda_make_array(workspace, ((size_t)n)*((size_t)k)*sizeof(float));
    cudaMalloc((void **)&workspace, ((size_t)n)*((size_t)k)*sizeof(float));

    for(int i = 0; i < batch; i++){
        im2col_gpu(ptr_in_tensor + i * in_tensor->size[1] * in_tensor->size[2] * in_tensor->size[3], in_tensor->size[1], in_tensor->size[2],
                in_tensor->size[3], weights->size[3], stride, padding, workspace);
        float * a = ptr_weights;
        float * b = workspace;
        float * c = ptr_out_tensor;
        gemm_gpu(0, 0, m,n, k, 1., a, k, b, n, 0., c+i*m*n, n);
    }

    //cuda_free_array(workspace);
    cudaFree(workspace);

    add_bias_gpu(ptr_out_tensor, ptr_biases, batch, out_tensor->size[1], out_tensor->size[2]*out_tensor->size[3]); 

    return 0;
}


//Using CUDNN
int inc_conv_v2(THCudaTensor *in_tensor, THCudaTensor *weights, THCudaTensor *biases, THCudaTensor *out_tensor, int padding, int stride)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights  = THCudaTensor_data(NULL, weights);
    float * ptr_biases = THCudaTensor_data(NULL, biases);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);

    int batch = in_tensor->size[0];

    int in_channels = in_tensor->size[1];
    int in_size = in_tensor->size[2];

    int out_channels = out_tensor->size[1];
    int out_size = out_tensor->size[2];

    int k_size = weights->size[2];

    int n = batch;

    cudnnHandle_t cudnn = cudnn_handle();
    //cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, in_channels, in_size, in_size);

    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, k_size, k_size);

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &n, &out_channels, &out_size, &out_size);

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, out_channels, out_size, out_size);

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    size_t ws_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);

    float * ws_data = NULL;
    if(ws_size>0)
    {
        cudaMalloc((void **)&ws_data, ws_size);
    }

    float alpha = 1.f;
    float beta = 0.f;

    //for(int i = 0; i < batch; i++){
    //    cudnnConvolutionForward(cudnn, &alpha, in_desc, ptr_in_tensor + i * in_channels * in_size * in_size,
    //            filt_desc, ptr_weights, conv_desc, algo, ws_data, ws_size, &beta,
    //            out_desc, ptr_out_tensor + i * out_channels * out_size * out_size);
    //}

    cudnnConvolutionForward(cudnn, &alpha, in_desc, ptr_in_tensor,
            filt_desc, ptr_weights, conv_desc, algo, ws_data, ws_size, &beta,
            out_desc, ptr_out_tensor);

    if(ws_size>0)
    {
        cudaFree(ws_data);
    }
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    //cudnnDestroy(cudnn);

    add_bias_gpu(ptr_out_tensor, ptr_biases, batch, out_tensor->size[1], out_tensor->size[2]*out_tensor->size[3]);

    return 0;
}

//Change Aware MCMK Convolution implementation(im2col + GEMM)
int inc_conv_v3(THCudaTensor * in_tensor, THCudaTensor * weights, THCudaTensor * biases, THCudaTensor * out_tensor,
 THCudaIntTensor * patch_location_tensor, int padding, int stride, int p_height, int p_width)
{
    float * ptr_in_tensor    = THCudaTensor_data(NULL, in_tensor);
    float * ptr_weights  = THCudaTensor_data(NULL, weights);
    float * ptr_biases = THCudaTensor_data(NULL, biases);
    float * ptr_out_tensor   = THCudaTensor_data(NULL, out_tensor);
    //int * ptr_location = THCudaIntTensor_data(NULL, patch_location_tensor);

    int p_row_start = 0;
    int p_col_start = 0;

    int batch = in_tensor->size[0];
    int in_channels = in_tensor->size[1];
    int in_size = in_tensor->size[2];
    int out_channels = out_tensor->size[1];
    int out_size = out_tensor->size[2];
    int k_size = weights->size[2];

    float * workspace = NULL;
    float * c = NULL;

    //memory allocation
    //workspace  = cuda_make_array(workspace, ((size_t)p_width*p_height)*((size_t)k_size*k_size*in_channels)*sizeof(float));
    //c = cuda_make_array(c, ((size_t)p_width*p_height)*((size_t)out_channels)*sizeof(float));

    cudaMalloc((void **)&workspace, ((size_t)p_width*p_height)*((size_t)k_size*k_size*in_channels)*sizeof(float));
    cudaMalloc((void **)&c, ((size_t)p_width*p_height)*((size_t)out_channels)*sizeof(float));

    //pre_mat memory copying
    //premat_mem_copy_gpu(out_size, out_channels, batch, ptr_out_tensor, ptr_out_tensor);

    for(int i = 0; i < batch; i++){
        inc_im2col_gpu(ptr_in_tensor + i * in_channels * in_size * in_size, in_channels, in_size,
                in_size, k_size, stride, padding, workspace, p_row_start, p_col_start, p_height, p_width);

        float * a = ptr_weights;
        float * b = workspace;

        int m = out_channels;
        int k = k_size * k_size * in_channels;
        int n = p_width * p_height;

        gemm_gpu(0, 0, m, n, k, 1., a, k, b, n, 0., c, n);
        add_bias_gpu(ptr_out_tensor, ptr_biases, 1, out_channels, p_width*p_height);

        //gemm output copying
        inc_conv_mem_copy_gpu(c, ptr_out_tensor, p_row_start, p_col_start, p_height, p_width, out_channels, out_size);
    }

    //memory freeing
    //cuda_free_array(c);
    //cuda_free_array(workspace);

    cudaFree(workspace);
    cudaFree(c);

    return 0;
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
