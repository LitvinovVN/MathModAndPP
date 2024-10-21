#pragma once

#include <iostream>

/*#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif*/

template<typename T>
__global__ void kernel_sum(T* dev_arr, size_t length, T* result)
{
    extern __shared__ T shared_array[];

    //printf("\nkernel_sum: length = %ld\n", length);
    const int tid = threadIdx.x + blockDim.x * blockIdx.x;
    //printf("\nkernel_sum: tid = %ld\n", tid);    
    const int number_of_threads = gridDim.x * blockDim.x;
    int n_elem_per_thread = length / (gridDim.x * blockDim.x);

    if(tid == 0)
    {
        printf("\nkernel_sum: number_of_threads = %ld\n", number_of_threads);
        printf("\nkernel_sum: n_elem_per_thread = %ld\n", n_elem_per_thread);
    }

    int block_start_idx = n_elem_per_thread * blockIdx.x * blockDim.x;
    int thread_start_idx = block_start_idx
            + threadIdx.x * n_elem_per_thread;
    int thread_end_idx = thread_start_idx + n_elem_per_thread;
    if(thread_end_idx > length) thread_end_idx = length;
    printf("\nkernel_sum: thread_start_idx = %ld\n", thread_start_idx);
    printf("\nkernel_sum: thread_end_idx = %ld\n", thread_end_idx);

    T localResult{0};
    
    for(size_t i = thread_start_idx; i < thread_end_idx; i++)
    {
        localResult += dev_arr[i];
    }
    printf("\nkernel_sum: localResult = %lf\n", localResult);
    //atomicAdd(result, localResult);
    shared_array[tid] = localResult;
    __syncthreads();
    
    if(tid == 0)
    {
        for(int i = 0; i < number_of_threads; i++)
        {
            *result += shared_array[i];
        }
         
    } 
    printf("\nkernel_sum: *result = %lf\n", *result);  
}

template<typename T>
class CudaHelper
{
public:
    
    // https://cuda-programming.blogspot.com/2013/01/threads-and-blocks-in-detail-in-cuda.html
    static T Sum(T* dev_arr, size_t length, unsigned blocksNum, unsigned threadsNum)
    {
        std::cout << "T Sum(" << dev_arr << ", "
                  << length << ", "<< blocksNum << ", "
                  << threadsNum << ")" <<std::endl;
        
        T result{0};
        T* dev_result;
        cudaMalloc(&dev_result, sizeof(T));
        //cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

        unsigned shared_mem_size = blocksNum*threadsNum*sizeof(T);
        std::cout << "shared_mem_size = " << shared_mem_size << std::endl;
        kernel_sum<<<blocksNum, threadsNum, shared_mem_size>>>(dev_arr, length, dev_result);

        cudaMemcpy(&result, dev_result, sizeof(T), cudaMemcpyDeviceToHost);

        std::cout << "Sum is " << result << std::endl;

        return result;
    }
};
