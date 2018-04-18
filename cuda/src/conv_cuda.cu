#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "cuda.h"
#include "curand.h"
#include "cublas_v2.h"

#include "conv_cuda.h"

#define BLOCK 512

typedef enum{
    RELU
} ACTIVATION;

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

void cuda_check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    }
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

void cublas_check_error(cublasStatus_t status){
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char *s = cublasGetErrorString(status);
        printf("CUBLAS Error: %s\n", s);
        assert(0);
    }
}

__host__ __device__ dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
#if defined(__CUDA_ARCH__)
        // Device code here
        x = ceilf((float)sqrtf((float)k));
#else
        // Host code here
        x = ceil(sqrt(k));
#endif
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 dimGrid(x, y, 1);
    return dimGrid;
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    cuda_check_error(status);
    return n;
}

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    cuda_check_error(cudaPeekAtLastError());
}

float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    cuda_check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        cuda_check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free_array(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    cuda_check_error(status);
}

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = index % width_col;
        int h_index = index / width_col;
        int h_out = h_index % height_col;
        int channel_in = h_index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        float* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void im2col_gpu(float *im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_col){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col);
}

__global__ void inc_im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col, int p_row_start, int p_col_start, int p_height, int p_width) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x){
        int w_out = p_col_start + index % p_width;
        int h_index = index / p_width;
        int h_out = p_row_start + h_index % p_height;
        int channel_in = h_index / p_height;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;

        float* data_col_ptr = data_col;
        //data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        data_col_ptr += (channel_out * p_height + h_out) * p_width + w_out;


        const float* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;

                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;

                //data_col_ptr += height_col * width_col;
                data_col_ptr += p_height * p_width;
            }
        }
    }
}

void inc_im2col_gpu(float *im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_col, int p_row_start, int p_col_start, int p_height, int p_width){
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * p_height * p_width;
    inc_im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(
                num_kernels, im, height, width, ksize, pad,
                stride, height_col,
                width_col, data_col, p_row_start, p_col_start, p_height, p_width);
}


void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cublasStatus_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    cublas_check_error(status);
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    int num = n*size*batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    cuda_check_error(cudaPeekAtLastError());
}

__device__ float relu_activate_kernel(float x){return x*(x>0);}

__device__ float activate_kernel(float x, ACTIVATION a)
{
    switch(a){
        case RELU:
            return relu_activate_kernel(x);
    }
    return 0;
}

__global__ void activate_array_kernel(float *x, int n, ACTIVATION a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = activate_kernel(x[i], a);
}

void activate_array_gpu(float *x, int n, ACTIVATION a)
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
    cuda_check_error(cudaPeekAtLastError());
}

__global__ void batch_conv_dp_child_kernel(int in_c, int in_w, int filter_w, int out_w, int padding, int stride, float * ptr_input, float * ptr_weights, float * ptr_output, int batch, int groups, int m, int k, int n, float * workspace)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < batch){
        int j;
        workspace = workspace + i*n*k*sizeof(float);
        for(j = 0; j < groups; j++){
            float * im = ptr_input + i * in_c * in_w * in_w;
            int height_col = (in_w + 2 * padding - filter_w) / stride + 1;
            int width_col = height_col;
            int num_kernels = in_c * height_col * width_col;
            im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
                BLOCK>>>(
                        num_kernels, im, in_w, in_w, filter_w, padding,
                        stride, height_col,
                        width_col, workspace);

            float * a = ptr_weights + j*out_w/groups;
            float * b = workspace;
            float * c = ptr_output;

            float ALPHA = 1.0;
            float BETA = 0.0;

            cublasHandle_t handle;
            cublasCreate(&handle);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &ALPHA, b, n, a, k, &BETA, c+i*m*n, n);
            cublasDestroy(handle);
        }

    }
}

__global__ void batch_conv_dp_parent_kernel(int in_c, int in_w, int filter_w, int out_w, int padding, int stride, float * ptr_input, float * ptr_weights, float * ptr_output, int batch, int groups, int m, int k, int n, float * workspace)
{
    dim3 dimGrid = cuda_gridsize(batch);
    batch_conv_dp_child_kernel<<<dimGrid, BLOCK>>>(in_c, in_w, filter_w, out_w, padding, stride, ptr_input, ptr_weights, ptr_output, groups, batch, m, k, n, workspace);
}

void batch_dp_gemm_conv_gpu(int in_channels, int in_size, int k_size, int out_size, int padding, int stride, float * ptr_input, float * ptr_weights, float * ptr_output, int groups, int batch, int m, int k, int n, float * workspace)
{
    batch_conv_dp_parent_kernel<<<1,1>>>(in_channels, in_size, k_size, out_size, padding, stride, ptr_input, ptr_weights, ptr_output, groups, batch, m, k, n, workspace);
    cuda_check_error(cudaPeekAtLastError());
}


__global__ void inc_conv_mem_copy_gpu_kernel(float *c, float *ptr_out_tensor, int p_row_start, int p_col_start, int p_height,
        int p_width, int channels, int size, int n)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < n)
    {
        int col = p_col_start + index % p_width;
        int row = p_row_start + index / p_width;

        float *in_data_ptr = c + index;
        float *out_data_ptr = ptr_out_tensor + row * size + col;

        for(int i=0; i<channels; i++)
        {
            *out_data_ptr = *in_data_ptr;

            in_data_ptr += p_width*p_height;
            out_data_ptr += size*size;
        }
    }
}

void inc_conv_mem_copy_gpu(float *c, float *ptr_out_tensor, int p_row_start, int p_col_start, int p_height, int p_width, int channels, int size)
{
    int num_kernels = p_height * p_width;
    inc_conv_mem_copy_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(c, ptr_out_tensor, p_row_start, p_col_start, p_height, p_width, channels, size, num_kernels);
    cuda_check_error(cudaPeekAtLastError());
}

__global__ void batched_inc_conv_dp_child_kernel(cublasHandle_t handle, int batch, float *workspace, float *c, float * ptr_in_tensor, float *ptr_out_tensor,
        float *ptr_weights, int p_row_start, int p_col_start, int p_width,
        int p_height, int k_size, int in_size, int in_channels, int out_size, int out_channels, int padding, int stride)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < batch){
        workspace = workspace + i * (p_width*p_height) * (k_size*k_size*in_channels) * sizeof(float);
        c = c + i * (p_width*p_height) * out_channels * sizeof(float);

        int height_col = (in_size + 2 * padding - k_size) / stride + 1;
        int width_col = (in_size + 2 * padding - k_size) / stride + 1;
        int num_kernels = in_channels * p_height * p_width;
        inc_im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(
                    num_kernels, ptr_in_tensor, in_size, in_size, k_size, padding,
                    stride, height_col,
                    width_col, workspace, p_row_start, p_col_start, p_height, p_width);
        cudaDeviceSynchronize();

        float * a = ptr_weights;
        float * b = workspace;

        int m = out_channels;
        int k = k_size * k_size * in_channels;
        int n = p_width * p_height;

        //cublasHandle_t handle = NULL;
        //cublasCreate(&handle);
        float ALPHA = 1.0;
        float BETA = 0.0;
        cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N,
                CUBLAS_OP_N, n, m, k, &ALPHA, b, n, a, k, &BETA, c, n);
        //cublasDestroy(handle);
        cudaDeviceSynchronize();

        num_kernels = p_height * p_width;
        inc_conv_mem_copy_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(c, ptr_out_tensor, p_row_start, p_col_start, p_height, p_width, out_channels, out_size, num_kernels);
    }
}

__global__ void batched_inc_conv_dp_parent_kernel(int batch, float *workspace, float *c, float * ptr_in_tensor, float *ptr_out_tensor,
        float *ptr_weights, int p_row_start, int p_col_start, int p_width,
        int p_height, int k_size, int in_size, int in_channels, int out_size, int out_channels, int padding, int stride)
{
    dim3 dimGrid = cuda_gridsize(batch);
    cublasHandle_t handle;
    cublasCreate(&handle);
    batched_inc_conv_dp_child_kernel<<<dimGrid, BLOCK>>>(handle, batch, workspace, c, ptr_in_tensor, ptr_out_tensor,
                ptr_weights, p_row_start, p_col_start, p_width,
                            p_height, k_size, in_size, in_channels, out_size, out_channels, padding, stride);
    cublasDestroy(handle);
}

void batched_inc_conv_dp_gpu(int batch, float *workspace, float *c, float * ptr_in_tensor, float *ptr_out_tensor,
        float *ptr_weights, int p_row_start, int p_col_start, int p_width,
        int p_height, int k_size, int in_size, int in_channels, int out_size, int out_channels, int padding, int stride)
{
    batched_inc_conv_dp_parent_kernel<<<1,1>>>(batch, workspace, c, ptr_in_tensor, ptr_out_tensor,
            ptr_weights, p_row_start, p_col_start, p_width,
            p_height, k_size, in_size, in_channels, out_size, out_channels, padding, stride);
    cuda_check_error(cudaPeekAtLastError());
}

__global__ void premat_mem_copy_gpu_kernel(const int n, int size, int channels, int batch, float *data_out_ptr, float* premat_ptr)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < n)
    {
        data_out_ptr += index;
        for (int i=0; i<batch; i++){
        *data_out_ptr = premat_ptr[index];
            data_out_ptr += size * size * channels;
        }
    }
}

void premat_mem_copy_gpu(int size, int channels, int batch, float *data_out_ptr, float *premat_ptr)
{
    int num_kernels = size * size * channels;
    premat_mem_copy_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(num_kernels, size, channels, batch, data_out_ptr, premat_ptr);
    cuda_check_error(cudaPeekAtLastError());
}

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

__global__ void print_array(float * ptr, int size)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index < 1)
    {
        for(int i=0; i<size; i++)
        {
            for(int j=0; j<size; j++)
            {
                printf("%f,", *(ptr+i*size+j));
            }
            printf("\n");
        }
        printf("\n");
    }
}

void cudnn_mem_copy_gpu(int batch, int channels, int size, int padding, int stride, float *in_ptr, float* out_ptr, int * ptr_location, int in_p_height, int in_p_width)
{
    int num_kernels = in_p_height*in_p_width*channels;
    cudnn_mem_copy_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
            BLOCK>>>(num_kernels, batch, channels, size, padding, stride, in_ptr, out_ptr, ptr_location, in_p_height, in_p_width);
}

__global__ void inc_conv_mem_copy_gpu_v2_kernel(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_biases, int * ptr_location, int p_height,
        int p_width, int channels, int size, int batch, int n)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index < n)
    {
        int h_index = index / p_width;
        int channel_in = h_index / p_height;

        float *in_data_ptr = ptr_temp_tensor + index;

        for(int i=0; i<batch; i++)
        {
            int w_out = *(ptr_location+i*2+1) + index % p_width;
            int h_out = *(ptr_location+i*2) + h_index % p_height;

            //adding bias and performing ReLU activation
            float temp = *in_data_ptr + ptr_biases[channel_in];
            ptr_out_tensor[channel_in*size*size+h_out*size+w_out] =  temp > 0 ? temp : 0;

            in_data_ptr += p_width*p_height*channels;
            ptr_out_tensor += size*size*channels;
        }
    }
}

void inc_conv_mem_copy_gpu_v2(float *ptr_temp_tensor, float *ptr_out_tensor, float * ptr_biases, int * ptr_location, int batch, int p_height, int p_width, int channels, int size)
{
    int num_kernels = p_height * p_width * channels;
    inc_conv_mem_copy_gpu_v2_kernel<<<(num_kernels+BLOCK-1)/BLOCK,
        BLOCK>>>(ptr_temp_tensor, ptr_out_tensor, ptr_biases, ptr_location, p_height, p_width, channels, size, batch, num_kernels);
    cuda_check_error(cudaPeekAtLastError());
}